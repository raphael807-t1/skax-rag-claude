"""rag_skax 컬렉션을 qdrant_snapshot/rag_skax.snapshot.gz 에서 바로 복구한다.
BGE-M3 임베딩을 처음부터 다시 계산할 필요가 없어서(수십 분 걸리는 작업), 이미 만들어둔
임베딩 결과(5,397개 포인트, 1,172개 문서)를 그대로 받아서 복구하는 빠른 경로다.

gzip 스트림을 그대로 업로드해서 압축 해제된 파일(원본 447MB)을 디스크에 남기지 않는다
(qdrant-client의 snapshot 업로드 wrapper는 파일 크기를 미리 알아야 해서 압축 해제 파일이
필요했음 — requests로 직접 멀티파트 업로드하면 스트리밍이 되어 이 문제가 없다).

실행: python3 restore_snapshot.py
전제: Qdrant가 localhost:6333에 떠 있어야 함 (README의 "1. Qdrant 띄우기" 참고).
이미 rag_skax 컬렉션이 있으면 덮어쓰기 전 확인을 한 번 받는다.
"""
import gzip
import os
import sys

import requests
from qdrant_client import QdrantClient

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_GZ = os.path.join(WORKSPACE, "qdrant_snapshot", "rag_skax.snapshot.gz")
COLLECTION = "rag_skax"
QDRANT_URL = "http://localhost:6333"


def main():
    if not os.path.exists(SNAPSHOT_GZ):
        print(f"[에러] 스냅샷 파일 없음: {SNAPSHOT_GZ}")
        sys.exit(1)

    client = QdrantClient("localhost", port=6333)

    if client.collection_exists(COLLECTION):
        info = client.count(COLLECTION)
        print(f"'{COLLECTION}' 컬렉션이 이미 있음 (현재 {info.count}개 포인트).")
        ans = input("덮어쓰고 스냅샷으로 복구할까요? (y/N): ").strip().lower()
        if ans != "y":
            print("취소함.")
            return

    print("Qdrant에 업로드 및 복구 중... (gzip 스트리밍, 압축 해제 파일을 디스크에 남기지 않음)")
    with gzip.open(SNAPSHOT_GZ, "rb") as f:
        resp = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/snapshots/upload?priority=snapshot",
            files={"snapshot": ("rag_skax.snapshot", f, "application/octet-stream")},
            timeout=600,
        )
    if resp.status_code != 200:
        print(f"[에러] 복구 실패: {resp.status_code} {resp.text[:500]}")
        sys.exit(1)

    info = client.count(COLLECTION)
    print(f"복구 완료. '{COLLECTION}' 포인트 수: {info.count}")


if __name__ == "__main__":
    main()
