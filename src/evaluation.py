"""Avaliação da recuperação: hit rate@k e recall@k contra o gabarito."""
import json
from pathlib import Path

from src.embeddings import MODELOS, Embedder
from src.indexing import nome_colecao
from src.query import artigos_unicos, buscar

KS = (1, 3, 5)
CHUNKINGS = {"A (artigo)": "artigo", "B (parágrafos)": "paragrafo"}


def carregar_perguntas(caminho: Path = Path("data/eval/perguntas.jsonl")) -> list[dict]:
    return [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines()]


def avaliar(colecao: str, perguntas: list[dict], embedder: Embedder) -> dict:
    """k é o número de CHUNKS recuperados (mesmo orçamento de contexto nas duas
    configurações); os chunks são reduzidos a artigos únicos antes de comparar."""
    hits = {k: [] for k in KS}
    recalls = {k: [] for k in KS}
    detalhes = []
    for p in perguntas:
        gold = set(p["artigos_corretos"])
        res = buscar(p["pergunta"], colecao, embedder, k=max(KS))
        for k in KS:
            achados = set(artigos_unicos(res[:k])) & gold
            hits[k].append(1.0 if achados else 0.0)
            recalls[k].append(len(achados) / len(gold))
        ranking = artigos_unicos(res)
        detalhes.append({"id": p["id"], "pergunta": p["pergunta"], "gabarito": sorted(gold),
                         "recuperados@5": ranking})
    media = lambda xs: sum(xs) / len(xs)
    return {
        "hit": {k: media(v) for k, v in hits.items()},
        "recall": {k: media(v) for k, v in recalls.items()},
        "detalhes": detalhes,
    }


def tabela_markdown(resultados: dict[str, dict], n: int) -> str:
    cab = "| Configuração | " + " | ".join(f"Hit@{k}" for k in KS) + " | " + \
          " | ".join(f"Recall@{k}" for k in KS) + " |"
    linhas = [cab, "|" + "---|" * (1 + 2 * len(KS))]
    for nome, r in resultados.items():
        linhas.append(f"| {nome} | " + " | ".join(f"{r['hit'][k]:.2f}" for k in KS) + " | "
                      + " | ".join(f"{r['recall'][k]:.2f}" for k in KS) + " |")
    return "\n".join(linhas) + f"\n\n_{n} perguntas._"


def main() -> None:
    perguntas = carregar_perguntas()
    resultados = {}
    for modelo, hf_nome in MODELOS.items():
        embedder = Embedder(hf_nome)
        for rotulo, chunking in CHUNKINGS.items():
            resultados[f"e5-{modelo} + {rotulo}"] = avaliar(
                nome_colecao(chunking, modelo), perguntas, embedder)
    tabela = tabela_markdown(resultados, len(perguntas))
    print(tabela)
    saida = Path("results")
    saida.mkdir(exist_ok=True)
    (saida / "avaliacao.md").write_text(tabela + "\n", encoding="utf-8")
    (saida / "avaliacao.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=1), encoding="utf-8")
    for nome, r in resultados.items():
        erros = [d for d in r["detalhes"] if not set(d["gabarito"]) & set(d["recuperados@5"])]
        print(f"\n{nome} - sem nenhum artigo correto no top-5 ({len(erros)}):")
        for d in erros:
            print(f"  #{d['id']} {d['pergunta']} | gabarito={d['gabarito']} | veio={d['recuperados@5']}")


if __name__ == "__main__":
    main()
