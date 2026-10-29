import json
import re
from pathlib import Path

from src.ingestion import carregar_texto, extrair_artigos, limpar_texto


def _norm(s):
    return re.sub(r"\s+", " ", s.lower())


def test_evidencias_existem_no_texto_da_lei():
    artigos = {a.artigo: _norm(a.texto)
               for a in extrair_artigos(limpar_texto(carregar_texto()))}
    for linha in Path("data/eval/perguntas.jsonl").read_text(encoding="utf-8").splitlines():
        p = json.loads(linha)
        assert set(p["evidencia"]) == set(p["artigos_corretos"])
        for art, trecho in p["evidencia"].items():
            assert _norm(trecho) in artigos[art], (p["id"], art)
