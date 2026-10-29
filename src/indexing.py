"""Indexação dos chunks em um ChromaDB persistente."""
import json
from functools import lru_cache
from pathlib import Path

import chromadb

from src.embeddings import MODELOS, Embedder

PERSIST_DIR = "chroma_db"
CAMPOS_META = ("artigo", "parte", "capitulo", "capitulo_titulo", "secao")


def carregar_chunks(caminho: Path) -> list[dict]:
    with caminho.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


@lru_cache(maxsize=None)
def cliente(persist_dir: str = PERSIST_DIR):
    return chromadb.PersistentClient(path=persist_dir)


def indexar(chunks: list[dict], colecao: str, embedder: Embedder,
            persist_dir: str = PERSIST_DIR):
    """Recria a coleção do zero, para a indexação ser idempotente."""
    db = cliente(persist_dir)
    if colecao in [c.name for c in db.list_collections()]:
        db.delete_collection(colecao)
    col = db.create_collection(colecao, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["texto"] for c in chunks],
        embeddings=embedder.passagens([c["texto"] for c in chunks]),
        metadatas=[{k: c[k] for k in CAMPOS_META} for c in chunks],
    )
    return col


def nome_colecao(chunking: str, modelo: str) -> str:
    return f"lgpd_{chunking}_{modelo}"


def main() -> None:
    for modelo, hf_nome in MODELOS.items():
        embedder = Embedder(hf_nome)
        for chunking in ("artigo", "paragrafo"):
            chunks = carregar_chunks(Path(f"data/processed/chunks_{chunking}.jsonl"))
            col = indexar(chunks, nome_colecao(chunking, modelo), embedder)
            print(f"{col.name}: {col.count()} chunks indexados")


if __name__ == "__main__":
    main()
