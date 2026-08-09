"""
Document loading.

Loads every supported file from a configurable folder (app.config.DOCUMENTS_DIR
by default, but callable with any path). This is deliberately format-flexible —
a real client's document set is rarely all-PDF — so we route by extension to
the right LangChain loader instead of assuming one file type.
"""

from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


def load_documents(folder: Path) -> list[Document]:
    """Load every supported document in `folder` into LangChain Document objects.

    Each Document's metadata includes `source` (the filename), which is what
    lets the API tell the user which document an answer came from.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Document folder not found: {folder}")

    documents: list[Document] = []
    skipped: list[str] = []

    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        loader_cls = LOADER_MAP.get(path.suffix.lower())
        if loader_cls is None:
            skipped.append(path.name)
            continue

        if loader_cls is TextLoader:
            loader = loader_cls(str(path), encoding="utf-8")
        else:
            loader = loader_cls(str(path))

        loaded = loader.load()
        for doc in loaded:
            # Normalize metadata: always use the filename (not the full path)
            # as the source, so it's readable in an API response.
            doc.metadata["source"] = path.name
            doc.metadata.setdefault("page", None)
        documents.extend(loaded)

    if skipped:
        print(f"[loader] Skipped unsupported file types: {skipped}")

    if not documents:
        raise ValueError(
            f"No supported documents found in {folder}. "
            f"Supported extensions: {list(LOADER_MAP.keys())}"
        )

    return documents
