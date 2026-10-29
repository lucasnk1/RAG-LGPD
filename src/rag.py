"""Pipeline completo: pergunta -> recuperação -> geração ancorada."""
import sys

from src.embeddings import MODELOS, Embedder
from src.generation import PROVEDORES, artigos_citados, montar_entrada
from src.indexing import nome_colecao
from src.query import buscar

CONFIG_PADRAO = {"chunking": "artigo", "modelo": "small", "k": 5}


def responder(pergunta: str, embedder: Embedder, provedor: str = "gemini",
              chunking: str = "artigo", modelo: str = "small", k: int = 5,
              where: dict | None = None) -> dict:
    trechos = buscar(pergunta, nome_colecao(chunking, modelo), embedder, k=k, where=where)
    resposta, modelo_llm = PROVEDORES[provedor](montar_entrada(pergunta, trechos))
    return {
        "pergunta": pergunta,
        "resposta": resposta,
        "modelo_llm": modelo_llm,
        "artigos_citados": artigos_citados(resposta),
        "artigos_recuperados": [t["artigo"] for t in trechos],
        "trechos": trechos,
    }


if __name__ == "__main__":
    pergunta = " ".join(sys.argv[1:]) or "Quais são os direitos do titular dos dados?"
    r = responder(pergunta, Embedder(MODELOS[CONFIG_PADRAO["modelo"]]))
    print(r["resposta"])
    print(f"\n[modelo: {r['modelo_llm']}]")
    print("\nRecuperados:", r["artigos_recuperados"], "| Citados:", r["artigos_citados"])
