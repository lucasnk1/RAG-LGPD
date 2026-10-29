"""Isola por que a chamada ao Gemini não responde. Uso: python scripts/diagnostico_gemini.py [modelo ...]"""
import os
import sys
import time

from google import genai
from google.genai import types

if not os.environ.get("GEMINI_API_KEY"):
    sys.exit("GEMINI_API_KEY não está definida neste terminal.")

cliente = genai.Client(http_options=types.HttpOptions(timeout=30_000))  # 30 s


def passo(nome, fn):
    t = time.time()
    try:
        saida = fn()
        print(f"[OK  {time.time() - t:5.1f}s] {nome}: {saida}")
        return saida
    except Exception as e:
        print(f"[FALHOU {time.time() - t:5.1f}s] {nome}: {type(e).__name__} "
              f"(status={getattr(e, 'status_code', getattr(e, 'code', None))}) {str(e)[:150]}")


modelos_pedidos = sys.argv[1:]

# 1) A chave e a rede funcionam? (a lista completa só aparece quando nenhum modelo é pedido)
if not modelos_pedidos:
    nomes = passo("listar modelos", lambda: [m.name.removeprefix("models/") for m in cliente.models.list()])
    modelos_pedidos = [os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")]

for modelo in modelos_pedidos:
    print("\n== modelo:", modelo)
    # 2) API clássica (generate_content)
    passo("generate_content", lambda: cliente.models.generate_content(
        model=modelo, contents="Responda só: ok").text)
    # 3) API nova (interactions), a que o src/generation.py usa
    passo("interactions", lambda: cliente.interactions.create(
        model=modelo, input="Responda só: ok", timeout=30).output_text)
