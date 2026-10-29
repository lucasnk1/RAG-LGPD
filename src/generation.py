"""Geração da resposta ancorada nos trechos recuperados (Gemini ou Ollama)."""
import json
import os
import re
import sys
import threading
import time
import urllib.request

NAO_ENCONTREI = "Não encontrei na lei."

PROMPT_SISTEMA = f"""Você é um assistente que responde perguntas sobre a LGPD (Lei 13.709/2018).

Regras obrigatórias:
1. Responda SOMENTE com base nos trechos da lei fornecidos em <trechos>. Não use conhecimento externo.
2. Cite o artigo que sustenta cada afirmação, no formato (Art. 18) ou (Art. 55-A).
3. Se os trechos não contiverem a informação necessária para responder, responda exatamente: "{NAO_ENCONTREI}" e nada além disso.
4. Não invente artigos, prazos, valores ou penalidades que não estejam nos trechos.
5. Seja objetivo, em português, e não dê aconselhamento jurídico."""


def montar_entrada(pergunta: str, trechos: list[dict]) -> str:
    blocos = "\n\n".join(f"[Art. {t['artigo']}]\n{t['texto']}" for t in trechos)
    return f"<trechos>\n{blocos}\n</trechos>\n\nPergunta: {pergunta}"


def artigos_citados(resposta: str) -> list[str]:
    achados = re.findall(r"Art\.?\s*(\d+(?:-[A-Z])?)", resposta)
    return list(dict.fromkeys(achados))


LIMITE_CHAMADA_S = 30  # respostas normais levam 3 a 6 s; o gemini-3.5-flash às vezes fica mudo


def _com_limite(fn, segundos: float):
    """Roda fn com um limite rígido de tempo. O timeout do SDK não cobre toda conexão
    pendurada (visto na prática: processo parado por minutos sem erro). A thread
    abandonada é daemon, então não impede o encerramento do programa."""
    resultado: dict = {}

    def alvo():
        try:
            resultado["ok"] = fn()
        except BaseException as e:  # repassa o erro original para quem chamou
            resultado["erro"] = e

    t = threading.Thread(target=alvo, daemon=True)
    t.start()
    t.join(segundos)
    if t.is_alive():
        raise TimeoutError(f"sem resposta em {segundos:.0f} s")
    if "erro" in resultado:
        raise resultado["erro"]
    return resultado["ok"]


# Ordem de preferência; o segundo entra se o primeiro ficar indisponível (503/timeout).
MODELOS_GEMINI = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]


def gerar_gemini(entrada: str, modelos: list[str] | None = None,
                 tentativas: int = 2) -> tuple[str, str]:
    """Retorna (resposta, modelo que respondeu). A chave vem de GEMINI_API_KEY
    (nunca do código). Repete em 429/5xx/timeout; se o modelo seguir falhando,
    passa para o próximo da lista. GEMINI_MODEL força um modelo único."""
    from google import genai

    forcado = os.environ.get("GEMINI_MODEL")
    modelos = [forcado] if forcado else (modelos or MODELOS_GEMINI)
    client = genai.Client()
    ultimo_erro: Exception | None = None
    for modelo in modelos:
        for i in range(tentativas):
            try:
                r = _com_limite(lambda: client.interactions.create(
                    model=modelo, system_instruction=PROMPT_SISTEMA, input=entrada,
                    timeout=25), LIMITE_CHAMADA_S)
                return r.output_text.strip(), modelo
            except Exception as e:
                status = getattr(e, "status_code", None)
                # sem status_code = falha de rede/timeout, que também vale repetir
                transitorio = status is None or status == 429 or status >= 500
                print(f"[gemini:{modelo}] tentativa {i + 1}/{tentativas} falhou "
                      f"({type(e).__name__}, status={status}): {str(e)[:150]}", file=sys.stderr)
                if not transitorio:
                    raise  # chave inválida, modelo inexistente etc.: repetir não ajuda
                ultimo_erro = e
                if i < tentativas - 1:  # sem espera inútil antes de trocar de modelo
                    time.sleep(5)
    raise RuntimeError("Todos os modelos Gemini falharam") from ultimo_erro


def gerar_ollama(entrada: str, modelo: str | None = None) -> tuple[str, str]:
    """Opção local e gratuita. Requer `ollama serve` e `ollama pull <modelo>`."""
    modelo = modelo or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    corpo = json.dumps({
        "model": modelo, "stream": False, "options": {"temperature": 0},
        "messages": [{"role": "system", "content": PROMPT_SISTEMA},
                     {"role": "user", "content": entrada}],
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=corpo,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["message"]["content"].strip(), modelo


PROVEDORES = {"gemini": gerar_gemini, "ollama": gerar_ollama}
