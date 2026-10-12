"""SK AX incremental 빌드. 매번 재생성 안 하고 upsert만 — 신규/변경분만 임베딩."""
import hashlib, json, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION = "rag_skax"
FULL_STORE = "skax_articles_full.json"
EMBEDDED_LOG = "skax_embedded_log.json"

def stable_id(url, idx):
    h = hashlib.md5(f"{url}::{idx}".encode()).hexdigest()
    return int(h[:16], 16) % (2**63 - 1)

with open(FULL_STORE, encoding="utf-8") as f:
    full_store = json.load(f)

embedded_log = {}
if os.path.exists(EMBEDDED_LOG):
    with open(EMBEDDED_LOG, encoding="utf-8") as f:
        embedded_log = json.load(f)

to_embed = {url: art for url, art in full_store.items() if embedded_log.get(url) != art.get("content_hash")}
print(f"전체 {len(full_store)}개 문서 중 신규/변경 {len(to_embed)}개만 임베딩 대상")

if not to_embed:
    print("임베딩할 신규 없음. 종료.")
else:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = []
    for url, art in to_embed.items():
        pieces = splitter.split_text(art["text"])
        for idx, piece in enumerate(pieces):
            chunks.append({"id": stable_id(url, idx), "text": piece, "title": art["title"], "url": url, "category": art.get("category","?"), "chunk_idx": idx, "total_chunks": len(pieces)})
    print(f"신규 청크 {len(chunks)}개")

    import torch
    from FlagEmbedding import BGEM3FlagModel
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, device=device)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=12)["dense_vecs"]

    client = QdrantClient("localhost", port=6333)
    if not client.collection_exists(COLLECTION):
        client.create_collection(collection_name=COLLECTION, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
        print(f"'{COLLECTION}' 신규 생성")
    else:
        print(f"'{COLLECTION}' 기존 컬렉션에 upsert")

    points = [PointStruct(id=c["id"], vector=vec.tolist(), payload={k:v for k,v in c.items() if k!="id"}) for c, vec in zip(chunks, embeddings)]
    for i in range(0, len(points), 256):
        client.upsert(collection_name=COLLECTION, points=points[i:i+256])

    for url, art in to_embed.items():
        embedded_log[url] = art["content_hash"]
    with open(EMBEDDED_LOG, "w", encoding="utf-8") as f:
        json.dump(embedded_log, f, ensure_ascii=False, indent=2)

    print(f"\nQdrant '{COLLECTION}' 최종 포인트 수: {client.count(COLLECTION).count}")
    print(f"누적 임베딩된 고유 문서 수: {len(embedded_log)}")
