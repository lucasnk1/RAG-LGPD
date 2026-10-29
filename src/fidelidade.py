"""Checagem de fidelidade: as respostas geradas estão sustentadas pelos trechos?

Uso:
  python -m src.fidelidade gerar    # gera respostas (retoma de onde parou)
  python -m src.fidelidade resumo   # calcula os números a partir da sua marcação
"""
import csv
import json
import random
import sys
import time
from pathlib import Path

from src.embeddings import MODELOS, Embedder
from src.evaluation import carregar_perguntas
from src.generation import NAO_ENCONTREI
from src.rag import CONFIG_PADRAO, responder

N_AMOSTRA = 15
SEMENTE = 42
PAUSA_S = 4  # folga para os limites do plano gratuito
# Perguntas que a LGPD não responde: o certo é recusar.
FORA_DO_ESCOPO = [
    "Qual a capital da França?",
    "Como funciona a tributação do imposto de renda sobre aluguéis?",
    "Qual o prazo de garantia legal de um produto durável pelo Código de Defesa do Consumidor?",
]
SAIDA = Path("results")
RESPOSTAS = SAIDA / "respostas_fidelidade.jsonl"
MARCACAO = SAIDA / "fidelidade_marcacao.csv"
REVISAO = SAIDA / "fidelidade_revisao.md"


def _amostra() -> list[dict]:
    perguntas = carregar_perguntas()
    escolhidas = random.Random(SEMENTE).sample(perguntas, N_AMOSTRA)
    itens = [{"id": f"G{p['id']}", "pergunta": p["pergunta"], "gabarito": p["artigos_corretos"],
              "tipo": "gabarito"} for p in sorted(escolhidas, key=lambda p: p["id"])]
    itens += [{"id": f"F{i}", "pergunta": q, "gabarito": [], "tipo": "fora_do_escopo"}
              for i, q in enumerate(FORA_DO_ESCOPO, 1)]
    return itens


def _ler_respostas() -> dict[str, dict]:
    if not RESPOSTAS.exists():
        return {}
    linhas = RESPOSTAS.read_text(encoding="utf-8").splitlines()
    return {r["id"]: r for r in map(json.loads, linhas)}


def gerar() -> None:
    SAIDA.mkdir(exist_ok=True)
    feitas = _ler_respostas()
    embedder = Embedder(MODELOS[CONFIG_PADRAO["modelo"]])
    for item in _amostra():
        if item["id"] in feitas:
            continue
        r = responder(item["pergunta"], embedder, k=CONFIG_PADRAO["k"])
        reg = {**item, "resposta": r["resposta"], "modelo_llm": r["modelo_llm"],
               "artigos_recuperados": r["artigos_recuperados"],
               "artigos_citados": r["artigos_citados"],
               "trechos": [{"artigo": t["artigo"], "texto": t["texto"]} for t in r["trechos"]]}
        with RESPOSTAS.open("a", encoding="utf-8") as f:  # grava a cada resposta
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
        print(f"{item['id']} ok ({r['modelo_llm']})")
        time.sleep(PAUSA_S)
    _escrever_arquivos_de_revisao(list(_ler_respostas().values()))


def _escrever_arquivos_de_revisao(regs: list[dict]) -> None:
    # CSV de marcação: só cria se não existir, para nunca apagar o seu trabalho.
    if not MARCACAO.exists():
        with MARCACAO.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["id", "tipo", "pergunta", "fiel (sim/parcial/nao)", "observacao"])
            for r in regs:
                w.writerow([r["id"], r["tipo"], r["pergunta"], "", ""])
    with REVISAO.open("w", encoding="utf-8") as f:
        f.write("# Checagem de fidelidade\n\nPara cada item, leia a resposta e os trechos e marque "
                f"em `{MARCACAO.name}`: **sim** (tudo sustentado), **parcial** (parte sustentada "
                "ou detalhe extra) ou **nao** (afirma algo que os trechos não dizem).\n\n"
                "Para itens `fora_do_escopo`, o correto é a recusa.\n\n")
        for r in regs:
            f.write(f"---\n\n## {r['id']} ({r['tipo']}): {r['pergunta']}\n\n")
            f.write(f"**Resposta** (`{r['modelo_llm']}`):\n\n> {r['resposta']}\n\n")
            f.write(f"Recuperados: {r['artigos_recuperados']} | Citados: {r['artigos_citados']}"
                    f" | Gabarito: {r['gabarito']}\n\n<details><summary>Trechos</summary>\n\n")
            for t in r["trechos"]:
                f.write(f"**Art. {t['artigo']}**\n\n{t['texto']}\n\n")
            f.write("</details>\n\n")
    print(f"\nLeia {REVISAO} e preencha {MARCACAO}")


def resumo() -> None:
    regs = _ler_respostas()
    with MARCACAO.open(encoding="utf-8-sig", newline="") as f:
        marc = {l["id"]: l["fiel (sim/parcial/nao)"].strip().lower() for l in csv.DictReader(f, delimiter=";")}
    faltam = [i for i, v in marc.items() if v not in ("sim", "parcial", "nao", "recusa")]
    if faltam:
        sys.exit(f"Faltam marcações válidas (sim/parcial/nao/recusa) em: {faltam}")
    gab = [r for r in regs.values() if r["tipo"] == "gabarito"]
    fora = [r for r in regs.values() if r["tipo"] == "fora_do_escopo"]
    # "recusa" só vale para respostas que são literalmente a recusa do modelo
    errados = [r["id"] for r in gab
               if (marc[r["id"]] == "recusa") != (r["resposta"].strip() == NAO_ENCONTREI)]
    if errados:
        sys.exit(f"'recusa' deve marcar exatamente as respostas '{NAO_ENCONTREI}'. Revise: {errados}")
    respondidas = [r for r in gab if marc[r["id"]] != "recusa"]
    cont = {v: sum(marc[r["id"]] == v for r in respondidas) for v in ("sim", "parcial", "nao")}
    print(f"Fidelidade (marcação manual, {len(respondidas)} respostas com conteúdo de "
          f"{len(gab)}): sim={cont['sim']} parcial={cont['parcial']} nao={cont['nao']}")
    print(f"Recusas (o modelo disse '{NAO_ENCONTREI}'): {len(gab) - len(respondidas)}/{len(gab)}")
    # métricas calculadas pelo código, sem julgamento
    ok_cit = sum(set(r["artigos_citados"]) <= set(r["artigos_recuperados"]) for r in gab)
    recusas = [r for r in gab if r["resposta"].strip() == NAO_ENCONTREI]
    print(f"Citações só de artigos recuperados: {ok_cit}/{len(gab)}")
    print(f"Recusas indevidas em perguntas com gabarito: {len(recusas)}/{len(gab)}")
    recusou_fora = sum(r["resposta"].strip() == NAO_ENCONTREI for r in fora)
    print(f"Recusa correta fora do escopo: {recusou_fora}/{len(fora)}")


if __name__ == "__main__":
    {"gerar": gerar, "resumo": resumo}.get(sys.argv[1] if len(sys.argv) > 1 else "", gerar)()
