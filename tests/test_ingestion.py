import re

import pytest

from src.ingestion import (
    _vetado, carregar_texto, chunk_por_artigo, chunk_por_paragrafo,
    extrair_artigos, limpar_texto,
)


@pytest.fixture(scope="module")
def artigos():
    return [a for a in extrair_artigos(limpar_texto(carregar_texto())) if not _vetado(a)]


def _chave(num):
    n, _, letra = num.partition("-")
    return int(n), letra


def test_ordem_e_sem_duplicatas(artigos):
    chaves = [_chave(a.artigo) for a in artigos]
    assert chaves == sorted(chaves)
    assert len(set(chaves)) == len(chaves)


def test_artigos_esperados(artigos):
    nums = {a.artigo for a in artigos}
    assert {"1", "5", "7", "18", "42", "46", "52", "55-A", "55-J", "65"} <= nums
    # vetados e glitch "Art. 5 7." não podem aparecer
    assert not nums & {"55", "56", "57", "58"}


def test_sem_ruido(artigos):
    for a in artigos:
        assert not re.search(r"Redação dada|Incluíd|planalto\.gov|L13709compilado", a.texto)


def test_metadados(artigos):
    por_num = {a.artigo: a for a in artigos}
    assert por_num["18"].capitulo == "III"
    assert por_num["7"].secao.startswith("Seção I")
    assert por_num["55-A"].capitulo == "IX"
    assert all(a.capitulo for a in artigos)


def test_chunks(artigos):
    a = chunk_por_artigo(artigos)
    b = chunk_por_paragrafo(artigos)
    assert len(a) == len(artigos)
    assert len(b) >= len(a)
    assert len({c.chunk_id for c in b}) == len(b)
