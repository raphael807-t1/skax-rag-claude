# SK AX(skax.co.kr) 자소서/면접 준비용 RAG — Claude Code용 패키지

SK AX 관련 자료(회사소개/뉴스룸/고객사례/채용공고/인재상/인사이트 트렌드 등 1,172개 문서)를
BGE-M3 임베딩 + Qdrant로 만든 RAG. 표준 MCP(stdio) 서버라 Claude Code뿐 아니라 Codex CLI,
Gemini CLI / Antigravity에서도 쓸 수 있다 → [`다른_AI환경_등록가이드.md`](다른_AI환경_등록가이드.md).

검색 도구가 담고 있는 카테고리/한계/쓰는 법은 `mcp_server.py`의 `search_skax` 함수 docstring에
다 적혀 있다 — 등록만 하면 AI가 그걸 읽고 알아서 씀.

## 셋업

**1. Qdrant 띄우기 (Docker)**
```bash
docker run -d --name qdrant -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

**2. 의존성 설치**
```bash
pip install -r requirements.txt
```

**3. 임베딩 채우기 (최초 1회, 권장: 스냅샷 복구)**
```bash
python3 restore_snapshot.py
```
이미 임베딩 완료된 Qdrant 스냅샷(5,397개 포인트)을 그대로 복구한다. 수십 초면 끝나고, 디스크에
별도 임시 파일을 남기지 않는다.
(새 문서를 추가해서 직접 다시 임베딩하고 싶으면 `python3 build_embeddings.py` — incremental이라
기존 데이터는 안전하게 유지되고 신규/변경분만 처리된다. 느림: 수십 분.)

**4. Claude Code에 MCP로 등록**
```bash
claude mcp add skax-rag --scope user -- python3 /절대경로/mcp_server.py
```
등록 후 `claude mcp list`로 `skax-rag`가 `✔ Connected`인지 확인.

## 구성
- `skax_articles_full.json` / `skax_embedded_log.json`: 수집 원본 + 임베딩 추적 로그
- `qdrant_snapshot/rag_skax.snapshot.gz` + `restore_snapshot.py`: 스냅샷 복구 경로
- `build_embeddings.py`: 직접 임베딩 빌드 (incremental)
- `mcp_server.py`: MCP 서버 (`search_skax` Tool 1개)
