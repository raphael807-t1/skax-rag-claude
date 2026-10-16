# SK AX RAG를 다른 AI 환경에 등록하는 법

이 RAG(`mcp_server.py` + Qdrant `rag_skax` 컬렉션)는 Claude에 종속된 게 전혀 없다.
임베딩도 로컬 BGE-M3, 검색도 로컬 Qdrant, 서버도 표준 MCP(stdio) 프로토콜이라서
**MCP를 지원하는 어떤 AI 코딩 에이전트에도 그대로 꽂을 수 있다.** 아래는 환경별 등록 방법.

공통 전제 (모든 환경에서 동일):
1. Docker로 Qdrant가 떠 있어야 함: `docker ps --filter name=qdrant`
2. `pip install -r requirements.txt` (또는 각자 쓰는 패키지 매니저로)
3. 서버 파일 경로를 안다: `/절대/경로/mcp_server.py`

등록 방법만 클라이언트마다 다르고, `mcp_server.py` 자체는 **수정할 필요 없음**.

---

## 1. Claude Code (이미 쓰고 있는 환경)

```bash
claude mcp add skax-rag --scope user -- python3 /절대경로/mcp_server.py
claude mcp list   # skax-rag 가 ✔ Connected로 뜨면 성공
```

---

## 2. Codex CLI (OpenAI, GPT 환경)

Codex CLI는 Claude Code와 거의 같은 포지션(터미널 코딩 에이전트)이고, MCP도 지원한다.
등록은 설정 파일(`~/.codex/config.toml`)에 직접 써주는 방식이다.

`~/.codex/config.toml`에 추가:
```toml
[mcp_servers.skax-rag]
command = "python3"
args = ["/절대경로/mcp_server.py"]
```

확인:
```bash
codex mcp list
```
(버전에 따라 `codex` 자체에 `mcp add` 서브커맨드가 있을 수도 있음 — `codex mcp add --help`로 먼저 확인해볼 것.)


---

## 3. Antigravity / Gemini CLI (Google, Gemini 환경)

Gemini CLI(및 이를 기반으로 하는 Antigravity)는 `settings.json`에 MCP 서버를 등록한다.

`~/.gemini/settings.json` (없으면 생성):
```json
{
  "mcpServers": {
    "skax-rag": {
      "command": "python3",
      "args": ["/절대경로/mcp_server.py"]
    }
  }
}
```

Antigravity는 보통 IDE 설정 UI에서 "MCP 서버 추가"로도 등록 가능 — 커맨드/args는 위와 동일하게 넣으면 됨.

확인: Gemini CLI/Antigravity를 재시작한 뒤 세션에서 `search_skax` Tool이 보이는지 확인.

---

## 환경별 체크리스트 (공통)

| 체크 | 내용 |
|---|---|
| Qdrant 떠 있나 | `docker ps --filter name=qdrant` |
| 패키지 설치됐나 | `pip install -r requirements.txt` |
| 경로가 절대경로인가 | 상대경로 쓰면 각 클라이언트가 어느 디렉터리에서 띄우는지에 따라 깨짐 — 항상 절대경로 |
| 등록 후 Tool이 실제로 보이나 | 클라이언트마다 "Connected"만 뜨고 Tool 로드가 늦게/안 되는 경우가 있음 — 세션에서 `search_skax` 호출해서 직접 확인 |

## 주의할 점
- 세 환경 다 **같은 Qdrant, 같은 컬렉션(`rag_skax`)을 공유**해도 안전함 — 읽기 전용 검색만 하니까 충돌 안 남.
  (단, `build_embeddings.py`로 데이터를 추가/수정하는 건 한 번에 한 프로세스만 돌릴 것 — 동시에 upsert하면 꼬일 수 있음.)
- `mcp_server.py`를 환경별로 복붙해서 따로 둘 필요 없음 — 하나의 파일을 세 클라이언트가 각자 다른 설정으로 가리키게만 하면 됨.
- `search_skax`의 Parent-Child 로직은 서버 코드에 있는 거라, 어느 클라이언트로 붙든 동일하게 적용됨.
