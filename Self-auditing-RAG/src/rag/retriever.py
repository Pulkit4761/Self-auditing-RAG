import json
from pathlib import Path
import faiss
import fitz
import numpy as np
from . import config
from .embedder import Embedder


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def _read_pdf(filepath: Path) -> str:
    doc = fitz.open(filepath)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


class Retriever:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[str] = []

    def ingest(self, docs_dir: Path | None = None) -> int:
        docs_dir = docs_dir or config.DOCUMENTS_DIR
        docs_dir = Path(docs_dir)

        texts: list[str] = []
        for filepath in sorted(docs_dir.glob("*.txt")):
            texts.append(filepath.read_text(encoding="utf-8"))
        for filepath in sorted(docs_dir.glob("*.pdf")):
            texts.append(_read_pdf(filepath))

        if not texts:
            raise FileNotFoundError(
                f"No .txt or .pdf files found in {docs_dir}. "
                "Add documents and try again."
            )

        self._chunks = []
        for text in texts:
            self._chunks.extend(
                _chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            )

        embeddings = self._embedder.embed(self._chunks)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        self._save()
        return len(self._chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        top_k = top_k or config.TOP_K
        if self._index is None:
            self._load()
        if self._index is None:
            raise RuntimeError("No index found. Run 'ingest' first.")

        query_vec = self._embedder.embed([query])
        faiss.normalize_L2(query_vec)

        _, indices = self._index.search(query_vec, top_k)
        return [self._chunks[i] for i in indices[0] if i < len(self._chunks)]

    def _save(self):
        config.INDEX_DIR.mkdir(exist_ok=True)
        faiss.write_index(self._index, str(config.INDEX_DIR / "faiss.index"))
        meta = config.INDEX_DIR / "chunks.json"
        meta.write_text(json.dumps(self._chunks), encoding="utf-8")

    def _load(self):
        index_path = config.INDEX_DIR / "faiss.index"
        meta_path = config.INDEX_DIR / "chunks.json"
        if not index_path.exists():
            return
        self._index = faiss.read_index(str(index_path))
        self._chunks = json.loads(meta_path.read_text(encoding="utf-8"))
