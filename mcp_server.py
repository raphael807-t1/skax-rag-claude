"""
SK AX(skax.co.kr) RAG — Claude Code용 독립 MCP 서버.

다른 RAG(rag_docs, rag_skhynix_news 등)와 분리된 단일 목적 패키지다.
Qdrant(localhost:6333)의 'rag_skax' 컬렉션을 BGE-M3로 검색한다.

실행: python3 mcp_server.py (stdio) — 보통은 `claude mcp add`로 등록해서 Claude Code가 띄움.
사전조건:
  1. Docker로 Qdrant 떠 있어야 함.
  2. build_embeddings.py 로 최초 1회 임베딩 빌드해뒀어야 함 (README 참고).
  3. pip install -r requirements.txt
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from qdrant_client import QdrantClient

from mcp.server.mcpserver import MCPServer

COLLECTION = "rag_skax"
mcp = MCPServer("skax-rag")

_model = None
_client = None


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


@mcp.tool()
def search_skax(query: str, top_k: int = 5) -> str:
    """SK AX(SK그룹 IT서비스/AX 계열사) 자소서/면접 준비용 컬렉션에서 검색한다.
    www.skax.co.kr 사이트 전체(142개 페이지 — 회사소개/서비스/산업별 솔루션/인사이트/채용/조직문화/
    컴플라이언스/윤리경영/공정거래/인권경영/동반성장/사회공헌/지속가능경영/환경경영/안전보건 등)와
    www.skcareers.com 실제 진행중인 채용공고(담당업무/자격요건/우대사항/근무조건/전형절차, 마감일 포함),
    skax.skcareers.com 채용 전용 서브사이트(직무소개/인재상/조직문화/복리후생/전형절차)까지 다룬다.
    (수집 시점 2026-10-12 기준, 축적형 컬렉션 — 채용공고는 시점에 따라 마감/신규가 바뀔 수 있음.)

    Args:
        query: 검색할 질문/키워드
        top_k: 반환할 결과 개수 (기본 5)
    """
    try:
        model = _get_model()
        client = _get_client()
    except Exception as e:
        return f"[에러] 모델/Qdrant 연결 실패: {e}\n- Docker에서 Qdrant가 떠 있는지 확인하세요: docker ps --filter name=qdrant"

    q_vec = np.array(model.encode([query])["dense_vecs"][0]).astype("float32").tolist()

    try:
        hits = client.query_points(collection_name=COLLECTION, query=q_vec, limit=top_k).points
    except Exception as e:
        return f"[에러] 검색 실패: {e}\n- 컬렉션 '{COLLECTION}'이 존재하는지 확인하세요 (build_embeddings.py를 먼저 돌렸는지 확인)."

    if not hits:
        return "검색 결과 없음 — 수집된 자료 범위 밖일 수 있습니다."

    blocks = []
    for i, h in enumerate(hits, start=1):
        title = h.payload.get("title", "?")
        cat = h.payload.get("category", "?")
        url = h.payload.get("url", "?")
        text = h.payload.get("text", "")
        blocks.append(f"[{i}위 score={h.score:.4f} | {cat}]\n{title}\n{url}\n{text.strip()}")

    return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    mcp.run(transport="stdio")
