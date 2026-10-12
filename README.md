# SK AX(skax.co.kr) 자소서/면접 준비용 RAG — Claude Code용 패키지

SK AX(SK그룹 IT서비스/AX 계열사) 관련 자료를 모아 BGE-M3 임베딩 + Qdrant로 만든 RAG.
**Claude Code 사용자용**이다 (Codex 등 다른 도구용이 아님) — `claude mcp add`로 등록해서 쓴다.

## 이게 뭘 담고 있나 (176개 문서, 1045개 청크, 2026-10-12 수집)
- `www.skax.co.kr` 사이트 전체 142개 페이지: 회사소개, 서비스(AI/Cloud/Digital Manufacturing/Digital ESG 등),
  산업별 솔루션, 인사이트(뉴스레터/트렌드), 채용, 조직문화, 컴플라이언스, 윤리경영, 공정거래, 인권경영,
  동반성장, 사회공헌, 지속가능경영, 환경경영, 안전보건 등
- `www.skcareers.com` 실제 진행중인 채용공고 28건 — 담당업무/자격요건/우대사항/근무조건/전형절차, 마감일 포함
- `skax.skcareers.com` (채용 전용 서브사이트) — 직무소개, **인재상**(Passionate/Professional/Proactive/People),
  조직문화(SUPEX/VWBE/Winning Culture), 복리후생, 전형절차

## 알아두어야 할 한계
- **채용공고는 시점 스냅샷이다.** 2026-10-12 기준 진행 중이던 28건이고, 마감되거나 새 공고가 올라오면
  내용이 달라진다 — 최신 채용 여부를 확인하고 싶으면 아래 "RAG 넓히기" 절차로 다시 긁어서 추가해야 한다.
- 일부 섹션(사업현황/인사이트 뉴스레터 등)은 원본이 그래픽/카드 UI 위주라 텍스트가 적을 수 있다.
- `www.skcareers.com/QnA`(1:1문의 FAQ)는 로그인 필요 페이지라서 제외했다 — 실제 FAQ 내용은 없다.

## 구성
- `skax_articles_full.json`: 수집 원본(176개 문서)
- `build_embeddings.py`: 청킹(800자/100오버랩) + BGE-M3 임베딩 + Qdrant 적재. **incremental** —
  이미 임베딩된(content_hash 동일) 문서는 재처리 안 하고, 컬렉션을 지우고 새로 만들지도 않는다.
  (재실행해도 기존 데이터가 안전하게 보존됨 — 새 채용공고 추가 등 RAG를 계속 넓혀갈 때 중요.)
- `skax_embedded_log.json`: 어떤 문서가 이미 임베딩됐는지 추적하는 로그 (url → content_hash)
- `mcp_server.py`: 독립 MCP 서버 (`search_skax(query, top_k)` Tool 1개 제공)
- `requirements.txt`

## 셋업

### 1. Qdrant 띄우기 (Docker)
```bash
docker run -d --name qdrant -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```
이미 다른 RAG(rag_docs 등)로 Qdrant를 쓰고 있다면 그 컨테이너를 그대로 쓰면 된다 —
`rag_skax`는 별도 컬렉션이라 안전하게 같은 Qdrant 인스턴스에 같이 둘 수 있다.

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 임베딩 빌드 (최초 1회, 1~2분 소요)
```bash
python3 build_embeddings.py
```
→ Qdrant에 `rag_skax` 컬렉션 생성됨 (1045개 청크 예상)

### 4. Claude Code에 MCP로 등록
```bash
claude mcp add skax-rag --scope user -- python3 /절대경로/mcp_server.py
```
(가상환경을 쓴다면 `python3`를 그 venv의 python 경로로 바꿀 것. `--scope user`로 등록하면
모든 프로젝트에서 공용으로 쓸 수 있다 — 이 프로젝트에서만 쓰려면 `--scope project`.)

등록 후 헬스체크:
```bash
claude mcp list
```
`skax-rag` 항목이 `✔ Connected`로 뜨면 성공. Claude Code 세션에서 `search_skax` Tool이 보이면 된다.

## 검증된 테스트 쿼리
- "SK AX는 어떤 회사야? 설립 배경과 사업 영역 알려줘" → 회사소개 정확 반환 (score 0.67)
- "k8s Engineer PL 채용 자격요건이 뭐야?" → 해당 채용공고 전문 반환 (score 0.72)
- "SK AX의 인재상이나 조직문화는 어떤 느낌이야?" → 인재상/조직문화 페이지 정확 반환 (score 0.70)
- "SK AX의 윤리경영/공정거래 관련 정책은?" → 인권경영 페이지 반환 (score 0.72)

## 활용 아이디어 (자소서 말고도)
- **면접 대비**: 조직문화/비전 알고 있다는 티 내는 답변 초안, 역질문 생성
- **채용공고 스크리닝**: "28개 공고 중 AI/ML 경험자가 쓸만한 거 추려줘" 식으로 전체 비교
- **경험 매칭**: 자기 프로젝트 경험 텍스트 주고 "어떤 공고 자격요건이랑 가장 잘 맞아?" 추천받기
- **정합성 체크**: 자소서/면접 답변이 회사가 실제로 말하는 인재상과 결이 맞는지 검증

## RAG를 더 넓혀서 채우고 싶으면
새 공고가 올라오거나, 더 많은 페이지를 넣고 싶으면:

1. **새 페이지/문서를 텍스트로 수집한다** (curl로 되면 curl, JS 렌더링 필요하면 브라우저 자동화 도구로).
   결과를 아래 형태로 만든다:
   ```json
   {
     "https://실제-URL": {
       "title": "페이지 제목",
       "category": "적당한 분류명(예: 채용공고, company, 채용_인재상 등)",
       "text": "본문 텍스트 (네비게이션 메뉴나 푸터 반복 노이즈는 미리 제거해둘 것)",
       "content_hash": "<text의 MD5 해시, hashlib.md5(text.encode()).hexdigest()>",
       "scraped_at": "YYYY-MM-DD HH:MM:SS",
       "source": "아무 문자열(출처 구분용)"
     }
   }
   ```
2. 위 형태로 `skax_articles_full.json`에 **항목을 추가**한다(기존 항목은 안 건드림, 새 키만 추가/갱신).
3. `python3 build_embeddings.py` 그냥 재실행한다.
   → `skax_embedded_log.json`과 비교해서 **새로 추가되거나 내용이 바뀐 문서만** 자동으로 청킹+임베딩됨.
   → 기존 데이터는 삭제되지 않음(컬렉션을 지우고 새로 만들지 않도록 이미 짜여진 스크립트임).
4. `mcp_server.py`는 수정할 필요 없음 — 컬렉션만 채우면 바로 검색에 반영됨.

수집 시 주의할 점:
- 페이지마다 반복되는 네비게이션 메뉴/푸터/개인정보처리방침 블록은 노이즈이므로 걸러낼 것
- 마감된 채용공고를 계속 남겨두면 "지금 뽑고 있나?" 질문에 틀린 답이 나올 수 있으니,
  마감공고는 category를 "채용공고_마감"처럼 바꿔서 구분해두는 걸 권장
- 같은 내용을 다른 URL로 중복 추가하지 않도록 주의
