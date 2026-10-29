"""Busca top-k com filtro opcional por metadados."""
import sys

from src.embeddings import MODELOS, Embedder
from src.indexing import cliente, nome_colecao


def buscar(pergunta: str, colecao: str, embedder: Embedder, k: int = 5,
           where: dict | None = None, persist_dir: str = "chroma_db") -> list[dict]:
    """`where` usa a sintaxe do Chroma, ex.: {"capitulo": "III"}."""
    col = cliente(persist_dir).get_collection(colecao)
    r = col.query(query_embeddings=[embedder.pergunta(pergunta)], n_results=k, where=where)
    return [
        {"chunk_id": i, "artigo": m["artigo"], "capitulo": m["capitulo"],
         "distancia": d, "texto": t}
        for i, m, d, t in zip(r["ids"][0], r["metadatas"][0], r["distances"][0], r["documents"][0])
    ]


def artigos_unicos(resultados: list[dict]) -> list[str]:
    """Artigos na ordem do ranking, sem repetição (vários chunks podem ser do mesmo artigo)."""
    return list(dict.fromkeys(r["artigo"] for r in resultados))


if __name__ == "__main__":
    pergunta = " ".join(sys.argv[1:]) or "Quais são os direitos do titular dos dados?"
    for modelo, hf_nome in MODELOS.items():
        emb = Embedder(hf_nome)
        for chunking in ("artigo", "paragrafo"):
            colecao = nome_colecao(chunking, modelo)
            print(f"\n== {colecao} ==")
            for r in buscar(pergunta, colecao, emb):
                print(f"Art. {r['artigo']:6} dist={r['distancia']:.3f}  {r['texto'][:80]!r}")
