"""
Text splitting.

Legal documents are section-structured (Articles, Sections, Clauses), so we use
a recursive character splitter with separators ordered to prefer breaking on
section boundaries and paragraph breaks before falling back to sentences or
raw characters. That keeps a single Article/Section together in one chunk
whenever it reasonably fits, which matters for retrieval precision — you want
a chunk to be a coherent legal provision, not an arbitrary slice of one.

Chunk size (800 chars) / overlap (150 chars) is a deliberate trade-off:
  - Small enough that a retrieved chunk is a specific, citable provision
    rather than a vague page-sized dump.
  - Large enough to hold a full Section/Article with its conditions intact.
  - The 150-char overlap means a clause split right at a chunk boundary is
    still fully readable from at least one of the two adjacent chunks.
Tune both in app/config.py if a client's documents run to longer or shorter
provisions (e.g. insurance policies often need a bigger chunk size).
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import settings

LEGAL_SEPARATORS = [
    "\nARTICLE ",
    "\nSECTION ",
    "\nSection ",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=LEGAL_SEPARATORS,
    )
    chunks = splitter.split_documents(documents)

    # Give each chunk a stable, human-readable id (source + index) so the API
    # can reference "which chunk" without leaking a raw vector-store id.
    per_source_counter: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        idx = per_source_counter.get(src, 0)
        chunk.metadata["chunk_id"] = f"{src}::chunk-{idx}"
        per_source_counter[src] = idx + 1

    return chunks
