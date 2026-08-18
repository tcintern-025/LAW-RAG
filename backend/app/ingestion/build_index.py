"""
Ingestion entrypoint.

Run this whenever you add/change documents for a client:

    python -m app.ingestion.build_index

It loads everything in settings.DOCUMENTS_DIR, splits it, embeds it, and
(re)writes it into that client's Chroma collection (settings.CLIENT_ID).
Re-running this is safe — it drops and rebuilds the collection rather than
silently duplicating chunks on every run.
"""

import sys

from app.config import settings
from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents
from app.retrieval.vectorstore import get_vectorstore
from langchain_community.vectorstores.utils import filter_complex_metadata


def build_index() -> None:
    print(f"[ingest] client_id       = {settings.CLIENT_ID}")
    print(f"[ingest] documents dir   = {settings.DOCUMENTS_DIR}")
    print(f"[ingest] chroma dir      = {settings.CHROMA_PERSIST_DIR}")

    print("[ingest] loading documents...")
    documents = load_documents(settings.DOCUMENTS_DIR)
    print(f"[ingest] loaded {len(documents)} document(s)")

    print("[ingest] splitting into chunks...")
    chunks = split_documents(documents)
    print(f"[ingest] produced {len(chunks)} chunk(s) "
          f"(chunk_size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})")

    print("[ingest] embedding + writing to Chroma "
          f"(model={settings.EMBEDDING_MODEL})... this can take a minute the "
          f"first time the embedding model downloads.")

    store = get_vectorstore()

    # Reset the collection so re-running ingestion doesn't duplicate chunks.
    existing_ids = store.get()["ids"]
    if existing_ids:
        store.delete(ids=existing_ids)
        print(f"[ingest] cleared {len(existing_ids)} existing chunk(s) "
              f"from collection '{settings.CLIENT_ID}'")

    # Chroma rejects metadata values of None (e.g. the 'page' field on
    # non-PDF documents), so strip anything it can't store before writing.
    chunks = filter_complex_metadata(chunks)

    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    store.add_documents(documents=chunks, ids=ids)

    print(f"[ingest] done. Collection '{settings.CLIENT_ID}' now has "
          f"{store._collection.count()} chunk(s).")


if __name__ == "__main__":
    try:
        build_index()
    except Exception as exc:  # noqa: BLE001
        print(f"[ingest] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)