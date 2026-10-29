"""Modelo de embeddings multilíngue (família E5)."""
from sentence_transformers import SentenceTransformer

MODELOS = {
    "small": "intfloat/multilingual-e5-small",  # 118M parâmetros
    "base": "intfloat/multilingual-e5-base",  # 278M parâmetros
}
MODELO = MODELOS["small"]


class Embedder:
    """Os modelos E5 foram treinados com prefixos: "query: " na pergunta e
    "passage: " no documento. Omitir os prefixos degrada a recuperação."""

    def __init__(self, nome: str = MODELO):
        self.modelo = SentenceTransformer(nome)

    def passagens(self, textos: list[str]) -> list[list[float]]:
        return self._codificar([f"passage: {t}" for t in textos])

    def pergunta(self, texto: str) -> list[float]:
        return self._codificar([f"query: {texto}"])[0]

    def _codificar(self, textos: list[str]) -> list[list[float]]:
        # normalizar os vetores faz a similaridade de cosseno virar produto interno
        return self.modelo.encode(
            textos, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        ).tolist()
