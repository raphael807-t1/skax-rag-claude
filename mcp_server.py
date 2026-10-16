"""
SK AX(skax.co.kr) RAG — Claude Code용 독립 MCP 서버.

다른 RAG(rag_docs, rag_skhynix_news 등)와 분리된 단일 목적 패키지다.
Qdrant(localhost:6333)의 'rag_skax' 컬렉션을 BGE-M3로 검색한다.

실행: python3 mcp_server.py (stdio) — 보통은 `claude mcp add`로 등록해서 Claude Code가 띄움.
사전조건:
  1. Docker로 Qdrant 떠 있어야 함.
  2. build_embeddings.py 로 최초 1회 임베딩 빌드해뒀어야 함 (README 참고).
  3. pip install -r requirements.txt

2026-10-16 업데이트 (Parent-Child):
- 기존엔 검색된 "청크"만 그대로 반환해서, 채용공고 자격요건처럼 긴 섹션이 chunk 경계에서
  끊겨 일부만 보이는 문제가 있었음(rag_skhynix_news 쪽에서 먼저 발견/수정한 것과 동일 문제).
- skax_articles_full.json을 Parent store로 같이 로드해서, 검색은 청크로 하되 결과는 해당
  "문서 전체"를 돌려주도록 바꿈. 같은 문서에서 여러 청크가 매칭돼도 결과엔 1번만 나옴(중복 제거).
"""
import json
import os
import re
import urllib.parse
from datetime import datetime, timedelta

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from mcp.server.mcpserver import MCPServer

COLLECTION = "rag_skax"
FULL_STORE_PATH = os.path.join(os.path.dirname(__file__), "skax_articles_full.json")
mcp = MCPServer("skax-rag")

_model = None
_client = None
_full_store = None


def _get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, device=device)
    return _model


def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient("localhost", port=6333)
    return _client


def _get_full_store():
    global _full_store
    if _full_store is None:
        with open(FULL_STORE_PATH, "r", encoding="utf-8") as f:
            _full_store = json.load(f)
    return _full_store


def _extract_query_keywords(query: str) -> list[str]:
    """질문에서 불용어와 조사를 제거하고 핵심 키워드 목록을 추출한다 (할루시네이션 방지 및 Title Boost용)."""
    stop_words = {
        "무엇인가요", "어떻게", "되나요", "어떤", "있는가", "알려주세요", "설명해주세요",
        "대해", "관련", "내용은", "주요", "현재", "구체적으로", "얼마인가요", "수치는", "정확히", "얼마", "얼마나", "얼마야", "대략", "뭐야", "뭐임", "뭔가요", "뭔지", "어때", "궁금해", "궁금", "소식", "뉴스", "알려줘", "요즘", "최근", "최신", "요새", "근황", "지금",
        "비교", "공통점", "차이점", "자격요건", "우대사항", "마감일", "마감", "상태는",
        "채용공고", "공고", "프로젝트", "사례", "지원", "영입", "모집", "sk", "ax", "skax",
        "분야", "전문가", "업무는", "필수", "조건", "역량", "사항", "다루었나요", "진행되었나요",
        "이용", "사내", "제공", "보도자료",
        "뉴스룸", "사례", "usecase", "story", "채용공고", "company", "manufacturing", "spec", "cloud",
        "finance", "commerce", "service", "services", "enterprisesolution", "esg", "insight",
        "assessment", "기타",
        # 2026-10-16 5차 피드백(블라인드 평가): "인재상"/"조직문화"/"복리후생"/"전형절차"/"직무소개"/"소개"는
        # 카테고리명이기도 하지만 그 자체가 질문의 핵심 내용 단어임(예: "SK AX 인재상이 뭐야") — 이걸 불용어로
        # 빼면 title_boost가 정답 페이지("인재상 (Talent)")를 못 알아보게 되는 역효과가 생김. 순수 메타
        # 라벨(보도자료/usecase/story처럼 내용과 무관한 분류어)만 불용어로 남기고 이것들은 되돌림.
    }
    raw_words = re.findall(r"[A-Za-z0-9가-힣]+", query)
    cleaned = []
    for w in raw_words:
        for suf in ["에서는", "에서", "으로는", "으로", "까지", "부터", "에게", "의", "은", "는", "이", "가", "을", "를", "에", "로", "과", "와", "도", "란"]:
            if w.endswith(suf) and len(w) > len(suf) + 1:
                w = w[:-len(suf)]
                break
        if len(w) >= 2 and w.lower() not in stop_words:
            cleaned.append(w)
    return cleaned


_RECENCY_KEYWORDS = ("요즘", "최근", "최신", "요새", "근황", "지금")
_RECENCY_WINDOW_DAYS = 540
# SK AX는 뉴스/사례 절대량이 skhynix보다 적어서 "최근"의 기준을 1.5년으로 넉넉하게 잡음.
_RECENCY_BOOST_WEIGHT = 0.0
_RECENCY_HALF_LIFE_DAYS = 540
# 2026-10-16 5차 피드백(블라인드 평가 34문항): 날짜 있는 뉴스/트렌드만 이 가산을 받다 보니, raw
# 유사도가 거의 같을 때 날짜 없는 채용/인재상/회사소개 문서가 구조적으로 항상 밀려났음("SK AX가
# 원하는 인재상" 질문에서 정답이 9위로 밀려 top_k=5 기본값으론 호출자에게 아예 안 들어옴). "최근/
# 요즘" 명시 질의가 아닌 평상시 가산은 통째로 끔(0.0). auto_recent 가중치(0.08)는 그대로 둠.


def _recency_factor(date_str: str, today) -> float:
    if not date_str:
        return 0.0
    import math

    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return 0.0
    days = max((today - d).days, 0)
    return math.exp(-days / _RECENCY_HALF_LIFE_DAYS)


def _extract_snippet(body: str, keywords: list[str], window: int = 1200) -> str:
    """full_text=False일 때 본문 앞부분이 아니라 쿼리 키워드가 실제로 등장하는 위치 주변을 잘라 반환한다."""
    if len(body) <= window:
        return body

    body_lower = body.lower()
    positions = []
    for kw in keywords:
        kw_norm = kw.lower()
        if len(kw_norm) < 2:
            continue
        idx = body_lower.find(kw_norm)
        if idx != -1:
            positions.append(idx)

    if not positions:
        return body[:window] + "...\n[중략, 전문은 위 URL 참고]"

    positions.sort()
    center = positions[len(positions) // 2]
    half = window // 2
    start = max(center - half, 0)
    end = min(start + window, len(body))
    start = max(end - window, 0)

    prefix = "[...앞부분 중략...]\n" if start > 0 else ""
    suffix = "\n[...뒷부분 중략, 전문은 위 URL 참고...]" if end < len(body) else ""
    return prefix + body[start:end] + suffix


_EXACT_TOKEN_RE = re.compile(
    r'"[^"]{2,30}"'
    r'|\d+(?:\.\d+)?%'
    r'|\b[A-Za-z]{1,4}[A-Z][A-Za-z0-9]{1,10}\b'
    r'|\b[A-Z]{2,6}\b'
)
# Dense 임베딩만으론 "45%", "sLLM" 같은 정확히 일치해야 하는 숫자/약어/고객사 이름을 못 잡을 수
# 있음 — 완전한 BM25/Sparse 하이브리드(재임베딩 필요) 대신, 쿼리에서 정확매칭 토큰을 뽑아 전체
# full_store 텍스트에 직접 문자열 검색(대소문자 구분)을 거는 가벼운 보완책.


_RESCUE_MAX_DOC_FREQ_RATIO = 0.05
# "AI"/"AX" 같은 흔한 약어가 토큰 패턴에 걸려 수백 개 문서에 매번 걸리고 결과도 항상 같은 몇
# 문서로 반복돼서 토큰만 먹는 문제가 있었음(2026-10-16 4차 피드백). 문서의 5% 넘게 나오는
# 토큰은 "정확매칭"으로서의 의미가 없다고 보고 보완검색에서 제외한다.


def _exact_match_rescue(
    query: str, full_store: dict, exclude_urls: set, category: str | None = None
) -> list[tuple[str, str]]:
    """category가 지정되면 그 카테고리 문서만 대상으로 한다 (2026-10-16 4차 피드백: 보완검색이
    category 필터를 무시하고 다른 카테고리 문서를 끼워넣는 문제가 있었음)."""
    tokens = {t.strip('"') for t in _EXACT_TOKEN_RE.findall(query)}
    if not tokens:
        return []

    pool = full_store
    if category:
        pool = {u: a for u, a in full_store.items() if a.get("category") == category}
        if not pool:
            return []

    total = len(pool)
    doc_freq = {t: 0 for t in tokens}
    hits_by_token = {t: [] for t in tokens}
    for url, art in pool.items():
        body = art.get("text", "")
        for tok in tokens:
            if tok in body:
                doc_freq[tok] += 1
                hits_by_token[tok].append(url)

    rare_tokens = [t for t in tokens if total and doc_freq[t] / total <= _RESCUE_MAX_DOC_FREQ_RATIO]

    found = []
    seen = set()
    for tok in rare_tokens:
        for url in hits_by_token[tok]:
            if url in exclude_urls or url in seen:
                continue
            found.append((url, tok))
            seen.add(url)
    return found


_INDUSTRY_WORDS = (
    "제조", "반도체", "배터리", "석유화학", "화학", "금융", "유통", "물류", "호텔", "교육",
    "통신", "미디어", "공공", "에너지", "건설", "플랜트", "병원", "제약", "자동차", "식품",
)
# Story/UseCase 제목에 산업명이 앞쪽에 붙는 패턴이 있어서, 재임베딩 없이 제목에서 바로 뽑아 쓴다.


def _industry_of(title: str) -> str | None:
    for w in _INDUSTRY_WORDS:
        if w in title:
            return w
    return None


@mcp.tool()
def search_skax(
    query: str,
    top_k: int = 5,
    full_text: bool = True,
    category: str | None = None,
    industry: str | None = None,
) -> str:
    """SK AX(SK그룹 IT서비스/AX 계열사) 자소서/면접 준비용 컬렉션에서 검색한다.
    www.skax.co.kr 사이트 전체(회사소개/서비스/산업별 솔루션/인사이트/채용/조직문화/컴플라이언스/
    윤리경영/공정거래/인권경영/동반성장/사회공헌/지속가능경영/환경경영/안전보건 등), 공식 뉴스룸
    보도자료(165건), 고객 사례(Story 42건 + UseCase 137건), 인사이트 리포트(트렌드 652건, 2021~),
    www.skcareers.com 채용공고(28건), skax.skcareers.com 채용 전용 서브사이트(직무소개/인재상/
    조직문화/복리후생/전형절차)까지 다룬다.
    (수집 시점 2026-10-12~16, 축적형 컬렉션 — 시점 스냅샷이라 채용공고 목록이 최신과 다를 수 있음.
    오래된 보도자료/사례는 SK AX의 과거 사명인 "SK㈜ C&C" 또는 "SK C&C"로 나올 수 있다 — 같은 회사다.)

    검색은 청크 단위로 하지만 결과는 해당 문서 "전문"(또는 매칭 부분 스니펫)을 제목/카테고리/원문
    URL과 함께 돌려준다(Parent-Child 구조). 같은 문서에서 나온 여러 청크가 매칭돼도 결과엔 1번만
    나온다(중복 제거).

    한계: 의미 임베딩 기반 검색이라 "8년 이상" 같은 정확한 수치/문구 조건으로 전체 채용공고를
    필터링하는 건 잘 안 될 수 있다(의미는 비슷해도 그 숫자가 본문에 있는지는 보장 못함) — 이런
    질문엔 category="채용공고"로 좁혀서 top_k를 크게 주고 본문을 직접 훑어보는 걸 권장한다.
    다만 "45%", "sLLM"처럼 숫자/약어/따옴표구문은 정확매칭 보완검색이 자동으로 같이 돌아서
    "[정확매칭]" 라벨로 결과 맨 뒤에 추가될 수 있다(최대 3건, 의미검색 순위와는 무관, category
    필터를 건 경우 그 카테고리 안에서만 찾음, 문서의 5% 넘게 나오는 흔한 토큰은 제외함).

    결과 맨 위 "(1위 순수 의미유사도=X)"는 각 항목 줄의 "최종점수"(타이틀매칭+최신성 가중치 포함,
    정렬 기준)와는 다른 수치다 — 가장 유사도 높은 문서 1건의 순수 코사인 유사도만 보여준다.

    다른 산업/카테고리 문서가 섞여 나오면 category로 좁혀서 다시 검색할 것. 실제 category 값:
    뉴스룸_보도자료, 사례_Story, 사례_UseCase, 채용공고, 채용_인재상, 채용_조직문화, 채용_복리후생,
    채용_전형절차, 채용_직무소개, 채용_소개, company, manufacturing, manufacturing-spec, cloud,
    finance, commerce, service, ax-services, enterprisesolution, esg-spec, insight, assessment,
    인사이트_트렌드(SK AX 자체 AI/업계 트렌드 분석 리포트), 기타.

    질문에 "요즘/최근/최신" 등이 들어있으면 자동으로 최근 자료를 우선 검색하고(날짜 없는 회사소개류
    정적 페이지는 이 가중치에서 제외됨), 그런 키워드가 없어도 비슷한 유사도면 최신 글이 약하게
    우선되도록 재정렬한다 — 오래된 "SK C&C" 시절 글이 부당하게 위로 오는 문제 방지.

    Args:
        query: 검색할 질문/키워드
        top_k: 반환할 "문서" 개수 (기본 5, 중복 제거 후 개수, 서버에서 최대 10으로 제한됨)
        full_text: True(기본)면 문서 전문을 그대로 반환. False면 문서당 쿼리 키워드가 실제로
            등장하는 부분 주변(약 1200자)만 스니펫으로 잘라 반환한다(앞부분이 아니라 매칭 위치 기준).
        category: 특정 카테고리로 좁히고 싶을 때 지정 (위 목록 중 하나). 모르면 None(전체 검색).
        industry: 제목에서 뽑은 산업명으로 더 좁히고 싶을 때 지정. 예: "제조", "반도체", "배터리",
            "석유화학", "화학", "금융", "유통", "물류", "호텔", "교육", "통신", "미디어", "공공",
            "에너지", "건설", "플랜트", "병원", "제약", "자동차", "식품". 주로 사례_Story/사례_UseCase
            제목에 산업명이 앞에 붙는 패턴을 이용한 것이라, 다른 카테고리엔 효과가 약할 수 있다.
    """
    _requested_top_k = top_k
    top_k = max(1, min(top_k, 10))
    # top_k 상한을 서버에서 강제함 — 호출하는 LLM이 근거 없이 큰 값을 줄 수 있고,
    # full_text=True와 큰 top_k가 같이 오면 토큰이 과도하게 늘어나는 문제 방지(2026-10-16).

    try:
        model = _get_model()
        client = _get_client()
        full_store = _get_full_store()
    except Exception as e:
        return f"[에러] 모델/Qdrant/Parent store 로드 실패: {e}\n- Docker에서 Qdrant가 떠 있는지 확인하세요: docker ps --filter name=qdrant"

    today = datetime.now()
    q_vec = np.array(model.encode([query])["dense_vecs"][0]).astype("float32").tolist()

    query_filter = None
    if category:
        query_filter = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category))])

    pool = max(top_k * 8, 40)
    try:
        hits = client.query_points(
            collection_name=COLLECTION, query=q_vec, query_filter=query_filter, limit=pool
        ).points
    except Exception as e:
        return f"[에러] 검색 실패: {e}\n- 컬렉션 '{COLLECTION}'이 존재하는지 확인하세요 (build_embeddings.py를 먼저 돌렸는지 확인)."

    if not hits:
        hint = f" (category='{category}' 필터 조건에 맞는 게 없을 수 있음)" if category else ""
        return f"검색 결과 없음 — 수집된 자료 범위 밖일 수 있습니다{hint}."

    q_keywords = _extract_query_keywords(query)
    auto_recent = any(k in query for k in _RECENCY_KEYWORDS)

    # 1. 문서별 최고 점수 수집 + 타이틀 매칭 부스트 + 최신성 가중치 적용
    doc_best_score = {}
    doc_raw_score = {}
    for h in hits:
        url = h.payload.get("url", "?")
        if url not in full_store:
            continue
        art = full_store[url]
        raw_score = h.score

        # 공백 제거 타이틀 매칭 (예: '디지털화폐' vs '디지털 화폐', 'O사', 'AI Agent' 등)
        title_norm = "".join(art.get("title", "").lower().split())
        title_matches = 0
        for kw in q_keywords:
            kw_norm = "".join(kw.lower().split())
            if len(kw_norm) >= 2 and kw_norm in title_norm:
                title_matches += 1
        title_boost = min(title_matches * 0.05, 0.15)

        # 2026-10-16 2차 피드백: 가산 상한이 의미유사도 차이(보통 0.01~0.05)보다 훨씬 커서 관련
        # 없는 문서를 끌어올리는 문제가 있었음 — auto_recent 가중치를 0.18→0.08로 낮춰 상한을 좁힘.
        recency_weight = 0.08 if auto_recent else _RECENCY_BOOST_WEIGHT
        recency_boost = recency_weight * _recency_factor(art.get("date", ""), today)
        final_score = raw_score + title_boost + recency_boost

        if url not in doc_best_score or final_score > doc_best_score[url]:
            doc_best_score[url] = final_score
            doc_raw_score[url] = raw_score

    # "요즘/최근" 질의면 날짜 없는 문서를 뒤로 보내고, 날짜 있는 문서 중에서도 윈도우 밖이면 제외(결과
    # 있을 때만). ⚠️ relevance pool을 추리기 "전에" 전체 후보에 먼저 적용해야 한다 — 순서를 바꾸면
    # (풀을 작게 자른 뒤 날짜 필터링) 교집합이 거의 비어버리는 문제가 있었다.
    all_candidate_urls = list(doc_raw_score.keys())
    if auto_recent:
        cutoff = (today - timedelta(days=_RECENCY_WINDOW_DAYS)).strftime("%Y-%m-%d")
        recent_urls = [u for u in all_candidate_urls if full_store[u].get("date", "") >= cutoff]
        if recent_urls:
            all_candidate_urls = recent_urls

    # industry 필터 (제목 기반 휴리스틱, Qdrant 재임베딩 없이 Python 단에서 후처리)
    if industry:
        all_candidate_urls = [
            u for u in all_candidate_urls if _industry_of(full_store[u].get("title", "")) == industry
        ]
        if not all_candidate_urls:
            return f"검색 결과 없음 — industry='{industry}' 조건에 맞는 문서가 없습니다. (주로 사례_Story/사례_UseCase 제목에서만 산업명을 인식함)"

    # 타이틀매칭/최신성 가산이 의미유사도 차이(보통 0.01~0.05)보다 커서 관련 없는 문서를 끌어올리는
    # 문제가 있었음(2026-10-16 피드백) — 그래서 "순수 의미유사도 상위 풀"을 먼저 추리고, 가산점은
    # 그 풀 안에서만 재정렬에 쓴다(풀 밖 문서는 가산점을 아무리 받아도 못 들어옴).
    _RELEVANCE_POOL_SIZE = max(top_k * 3, 15)
    candidate_urls = sorted(all_candidate_urls, key=lambda u: doc_raw_score[u], reverse=True)[:_RELEVANCE_POOL_SIZE]

    # 의미유사도가 너무 낮은 결과는 쳐낸다. 컷은 가산점 재정렬 "전에" 풀에 먼저 적용한다 — 재정렬 후
    # "최종점수 1위는 항상 유지"로 하면, 타이틀 가산으로 1위에 오른 0.48 문서가 살아남고 더 관련 있는
    # 0.51~0.57 문서들 위에 놓이면서 "관련도 낮음" 경고까지 잘못 뜨는 버그가 있었음(2026-10-16 3차 피드백).
    # 전부 컷 미만이면 순수 의미유사도 1위 하나만 경고와 함께 남긴다(빈 응답보다 LLM에게 유용한 신호).
    _MIN_SIMILARITY = 0.50
    would_be = sorted(candidate_urls, key=lambda u: doc_best_score[u], reverse=True)[:top_k]
    passed = [u for u in candidate_urls if doc_raw_score[u] >= _MIN_SIMILARITY] or candidate_urls[:1]
    # 점수 순으로 정렬하여 상위 top_k개 선택 (Parent 문서 중복 제거 및 top_k 보장)
    sorted_urls = sorted(passed, key=lambda u: doc_best_score[u], reverse=True)[:top_k]
    n_filtered_out = len([u for u in would_be if u not in sorted_urls])

    if not sorted_urls:
        return "검색 결과 없음 — 일치하는 문서를 찾을 수 없습니다."

    top1_raw_score = doc_raw_score[sorted_urls[0]]

    # 2. 할루시네이션 방지 점검: 상위 문서 본문 전체에서 쿼리 핵심 키워드 출현 여부 점검
    all_body_norm = "".join([full_store[u]["text"] + " " + full_store[u]["title"] for u in sorted_urls]).lower().replace(" ", "")
    non_year_keywords = [kw for kw in q_keywords if not re.match(r"^\d+년?도?$", kw)]
    matched_non_year = [kw for kw in non_year_keywords if "".join(kw.lower().split()) in all_body_norm]
    match_ratio = len(matched_non_year) / len(non_year_keywords) if non_year_keywords else 1.0

    head_lines = []
    if _requested_top_k > 10:
        head_lines.append(f"(※ 요청 top_k={_requested_top_k} → 서버 상한 10으로 조정됨)")
    if auto_recent:
        head_lines.append(f"(※ '{query}'에서 최신성 질의로 판단 → 최근 {_RECENCY_WINDOW_DAYS}일 자료 우선 검색)")
    if n_filtered_out > 0:
        head_lines.append(f"(※ 의미유사도 {_MIN_SIMILARITY} 미만 {n_filtered_out}건 결과에서 제외됨)")
    head_lines.append(f"(1위 순수 의미유사도={top1_raw_score:.3f})")
    if top1_raw_score < 0.50:
        head_lines.append("⚠️ 관련도 낮음 — 검색된 문서가 질문과 직접 관련이 없을 수 있습니다. 본문에 없는 내용은 '수집된 자료에 없음'으로 답변하세요.")
    elif non_year_keywords and (len(matched_non_year) == 0 or match_ratio <= 0.50):
        head_lines.append("⚠️ 주의: 검색된 상위 문서 본문에서 질문의 핵심 키워드 대부분이 발견되지 않았습니다. 자료에 명시되지 않은 수치나 사실은 지어내지 말고 반드시 '수집된 자료에 없음'으로 답변해야 합니다.")

    # 정확매칭 보완검색 — 의미검색이 놓쳤을 수 있는 정확 일치 문서를 추가.
    rescue_hits = _exact_match_rescue(query, full_store, exclude_urls=set(sorted_urls), category=category)
    if rescue_hits:
        head_lines.append(
            f"(※ 정확매칭 보완검색으로 {len(rescue_hits)}건 추가 발견 — 아래 '[정확매칭]' 표시된 항목, "
            f"의미검색 순위와는 별개로 본문에 그 수치/약어가 실제로 있는 문서임)"
        )

    header_block = "\n".join(head_lines)

    blocks = []
    for i, url in enumerate(sorted_urls, start=1):
        art = full_store[url]
        body = art["text"].strip()
        if not full_text:
            body = _extract_snippet(body, q_keywords)
        header = (
            f"[{i}위 | 유사도(raw)={doc_raw_score[url]:.4f} "
            f"최종점수(타이틀매칭+최신성 가중치 포함, 정렬기준)={doc_best_score[url]:.4f} "
            f"| {art.get('category','?')} | {art.get('date') or '날짜없음'}]"
        )
        display_url = urllib.parse.unquote(url)  # 퍼센트인코딩된 URL을 읽을 수 있게 디코딩(토큰도 줄어듦)
        blocks.append(f"{header}\n{art['title']}\n{display_url}\n\n{body}")

    for url, tok in rescue_hits[:3]:
        art = full_store[url]
        body = art["text"].strip()
        if not full_text:
            body = _extract_snippet(body, [tok])
        display_url = urllib.parse.unquote(url)
        header = f"[정확매칭 | 일치 토큰='{tok}' | {art.get('category','?')} | {art.get('date') or '날짜없음'}]"
        blocks.append(f"{header}\n{art['title']}\n{display_url}\n\n{body}")

    content = "\n\n---\n\n".join(blocks)
    return f"{header_block}\n\n{content}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
