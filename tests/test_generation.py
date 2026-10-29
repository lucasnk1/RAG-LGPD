from src.generation import NAO_ENCONTREI, PROMPT_SISTEMA, artigos_citados, montar_entrada


def test_entrada_contem_trechos_e_pergunta():
    e = montar_entrada("Qual o prazo?", [{"artigo": "48", "texto": "Art. 48. O controlador..."}])
    assert "[Art. 48]" in e and "Pergunta: Qual o prazo?" in e


def test_prompt_exige_recusa_literal():
    assert NAO_ENCONTREI in PROMPT_SISTEMA


def test_artigos_citados():
    assert artigos_citados("Conforme (Art. 18) e Art. 55-A, também (Art. 18).") == ["18", "55-A"]
    assert artigos_citados(NAO_ENCONTREI) == []


def test_limite_rigido_de_tempo():
    import time

    import pytest

    from src.generation import _com_limite

    assert _com_limite(lambda: 42, 2) == 42
    with pytest.raises(TimeoutError):
        _com_limite(lambda: time.sleep(5), 0.3)
    with pytest.raises(ValueError):
        _com_limite(lambda: (_ for _ in ()).throw(ValueError("x")), 2)
