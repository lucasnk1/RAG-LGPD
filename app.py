"""Demo Gradio do assistente RAG sobre a LGPD (publicável no Hugging Face Spaces).

A chave do Gemini vem da variável de ambiente GEMINI_API_KEY (no Spaces: Settings > Secrets).
"""
import time
from pathlib import Path

import gradio as gr

from src.embeddings import MODELOS, Embedder
from src.indexing import carregar_chunks, cliente, indexar, nome_colecao
from src.rag import responder

CHUNKS = Path("data/processed/chunks_artigo.jsonl")
COLECAO = nome_colecao("artigo", "small")
MAX_PERGUNTA = 300

embedder = Embedder(MODELOS["small"])
chunks = carregar_chunks(CHUNKS)
# O índice não vai para o Git; no primeiro start ele é reconstruído (73 chunks, segundos em CPU).
if COLECAO not in [c.name for c in cliente().list_collections()]:
    indexar(chunks, COLECAO, embedder)
CAPITULOS = ["Todos"] + list(dict.fromkeys(c["capitulo"] for c in chunks))


def perguntar(pergunta: str, capitulo: str):
    pergunta = (pergunta or "").strip()
    if not pergunta:
        return "Digite uma pergunta sobre a LGPD.", ""
    if len(pergunta) > MAX_PERGUNTA:
        return f"Pergunta longa demais (máximo de {MAX_PERGUNTA} caracteres).", ""
    where = None if capitulo == "Todos" else {"capitulo": capitulo}
    inicio = time.perf_counter()
    print(f"[app] pergunta recebida (capítulo={capitulo}): {pergunta[:60]!r}", flush=True)
    try:
        r = responder(pergunta, embedder, where=where)
    except Exception as e:  # não vaza detalhes internos para o usuário
        print(f"[app] ERRO após {time.perf_counter() - inicio:.1f}s: {type(e).__name__}: "
              f"{str(e)[:200]}", flush=True)
        return "Não consegui gerar a resposta agora. Tente novamente em instantes.", ""
    print(f"[app] respondeu em {time.perf_counter() - inicio:.1f}s via {r['modelo_llm']} "
          f"(recuperados={r['artigos_recuperados']})", flush=True)
    trechos = "\n\n---\n\n".join(
        f"**Art. {t['artigo']}** (cap. {t['capitulo']}, distância {t['distancia']:.3f})\n\n{t['texto']}"
        for t in r["trechos"])
    return f"{r['resposta']}\n\n<sub>Modelo: {r['modelo_llm']}</sub>", trechos


with gr.Blocks(title="Assistente LGPD (RAG)") as demo:
    gr.Markdown(
        "# Assistente LGPD (RAG)\n"
        "Perguntas e respostas sobre a Lei 13.709/2018, com base **somente** nos trechos "
        "recuperados da lei. Projeto de portfólio; **não é aconselhamento jurídico** e pode errar "
        "ou recusar-se a responder (\"Não encontrei na lei.\").")
    pergunta = gr.Textbox(label="Pergunta", lines=2, max_lines=4,
                          placeholder="Ex.: O titular pode retirar o consentimento que deu?")
    capitulo = gr.Dropdown(CAPITULOS, value="Todos", label="Filtrar por capítulo (metadado)")
    botao = gr.Button("Perguntar", variant="primary")
    resposta = gr.Markdown(label="Resposta")
    with gr.Accordion("Trechos recuperados (o que o modelo recebeu)", open=False):
        trechos = gr.Markdown()
    gr.Examples([["O titular pode voltar atrás e retirar o consentimento que deu?", "Todos"],
                 ["Em quais casos a LGPD não se aplica?", "Todos"],
                 ["Qual o limite da multa simples pelo descumprimento da lei?", "Todos"]],
                inputs=[pergunta, capitulo])
    botao.click(perguntar, [pergunta, capitulo], [resposta, trechos], concurrency_limit=2)
    pergunta.submit(perguntar, [pergunta, capitulo], [resposta, trechos], concurrency_limit=2)

if __name__ == "__main__":
    demo.queue(max_size=10).launch()
