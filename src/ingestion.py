"""Ingestão da LGPD: PDF -> texto limpo -> artigos -> chunks com metadados."""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

PDF_PATH = Path("data/lgpd_L13709_compilado.pdf")

# Ruído do PDF gerado a partir do site do Planalto (cabeçalho/rodapé de página).
_LINHA_RUIDO = re.compile(
    r"^(\d{2}/\d{2}/\d{4}, \d{2}:\d{2} .*|https?://\S+(?: \d+/\d+)?|\d+/\d+)$"
)
# Anotações de alteração legislativa; podem quebrar linha, por isso [^)]*.
_ANOTACAO = re.compile(
    r"\s*\((?:Redação dada|Incluíd[oa]|Revogad[oa]|Vide|Vigência)[^)]*\)"
)
_VIGENCIA = re.compile(r"[ \t]*\bVigência\b")
_ARTIGO = re.compile(r"^Art\.\s*(\d+)(?:º|°)?(?:-([A-Z]))?\.?(?:\s+(.*))?$")
_CAPITULO = re.compile(r"^CAP[ÍI]TULO\s+([IVX]+)\s*$")
_SECAO = re.compile(r"^Se[çc][ãa]o\s+([IVX]+)\s*$")
_DISPOSITIVO_VAZIO = re.compile(
    r"^((§\s*\d+º?|Parágrafo único|[IVXLC]+(-[A-Z])?\s*-)\s*\.?\s*(\((VETADO|revogad[oa])\)\.?;?)?"
    r"|\((VETADO|revogad[oa])\)\.?)$", re.I)
_PARAGRAFO = re.compile(r"^(§\s*\d+|Parágrafo único)")


@dataclass
class Artigo:
    artigo: str  # "18", "55-A"
    capitulo: str  # "IV"
    capitulo_titulo: str
    secao: str  # "" quando não há seção
    texto: str  # caput + incisos + parágrafos


@dataclass
class Chunk:
    chunk_id: str
    texto: str
    artigo: str
    parte: str  # "completo", "caput", "§1", "paragrafo-unico"
    capitulo: str
    capitulo_titulo: str
    secao: str


def carregar_texto(pdf_path: Path = PDF_PATH) -> str:
    leitor = PdfReader(str(pdf_path))
    return "\n".join(p.extract_text() or "" for p in leitor.pages)


def limpar_texto(bruto: str) -> str:
    texto = _ANOTACAO.sub("", bruto)
    texto = _VIGENCIA.sub("", texto)
    # "Art. 5 7." (defeito do HTML original do Planalto) -> "Art. 57."
    texto = re.sub(r"Art\.\s*(\d)\s+(\d)\.", r"Art. \1\2.", texto)
    linhas = [l.strip() for l in texto.splitlines()]
    linhas = [l for l in linhas if l and not _LINHA_RUIDO.match(l)]
    # dispositivos revogados/vetados viram só o rótulo ("§ 1º") ou "(revogado)"
    linhas = [l for l in linhas if not _DISPOSITIVO_VAZIO.match(l)]
    # descarta o preâmbulo (cabeçalho do site) até o primeiro capítulo
    inicio = next(i for i, l in enumerate(linhas) if _CAPITULO.match(l))
    return "\n".join(linhas[inicio:])


def _eh_titulo_caixa_alta(linha: str) -> bool:
    letras = re.sub(r"[^A-Za-zÀ-ÿ]", "", linha)
    return bool(letras) and letras.isupper() and not _ARTIGO.match(linha)


def extrair_artigos(texto_limpo: str) -> list[Artigo]:
    artigos: list[Artigo] = []
    capitulo = titulo_cap = secao = ""
    atual: list[str] | None = None
    meta: tuple[str, str, str, str] | None = None

    def fechar():
        if atual is not None and meta is not None:
            artigos.append(Artigo(meta[0], meta[1], meta[2], meta[3], "\n".join(atual)))

    linhas = texto_limpo.split("\n")
    i = 0
    while i < len(linhas):
        linha = linhas[i]
        if m := _CAPITULO.match(linha):
            fechar(); atual = None
            capitulo, titulo_cap, secao = m.group(1), "", ""
            i += 1
            partes = []
            while i < len(linhas) and _eh_titulo_caixa_alta(linhas[i]):
                partes.append(linhas[i].strip("�� "))
                i += 1
            titulo_cap = " ".join(partes)
            continue
        if m := _SECAO.match(linha):
            fechar(); atual = None
            secao = f"Seção {m.group(1)} - {linhas[i + 1]}" if i + 1 < len(linhas) else ""
            i += 2
            continue
        if m := _ARTIGO.match(linha):
            fechar()
            num = m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
            meta = (num, capitulo, titulo_cap, secao)
            atual = [linha]
        elif atual is not None:
            atual.append(linha)
        i += 1
    fechar()
    return artigos


def _vetado(a: Artigo) -> bool:
    """Artigo sem conteúdo recuperável: vetado, revogado ou só o rótulo."""
    corpo = re.sub(r"^Art\.\s*\S+", "", a.texto.strip()).strip(" .\n")
    return corpo == "" or bool(re.fullmatch(r"\((VETADO|revogado)\)", corpo, re.I))


def chunk_por_artigo(artigos: list[Artigo]) -> list[Chunk]:
    """Configuração A: 1 chunk = 1 artigo inteiro."""
    return [
        Chunk(f"art-{a.artigo}", a.texto, a.artigo, "completo",
              a.capitulo, a.capitulo_titulo, a.secao)
        for a in artigos
    ]


def chunk_por_paragrafo(artigos: list[Artigo]) -> list[Chunk]:
    """Configuração B: caput (com incisos) e cada parágrafo viram chunks próprios.

    Cada chunk repete o prefixo "Art. N" para não perder o contexto.
    """
    chunks: list[Chunk] = []
    for a in artigos:
        blocos: list[list[str]] = [[]]
        for linha in a.texto.split("\n"):
            if _PARAGRAFO.match(linha) and blocos[-1]:
                blocos.append([])
            blocos[-1].append(linha)
        if len(blocos) == 1:
            chunks.extend(chunk_por_artigo([a]))
            continue
        for k, bloco in enumerate(blocos):
            primeira = bloco[0]
            if k == 0:
                parte = "caput"
                texto = "\n".join(bloco)
            else:
                m = _PARAGRAFO.match(primeira)
                rotulo = m.group(1)
                parte = "paragrafo-unico" if rotulo.startswith("Par") else \
                    "§" + re.sub(r"\D", "", rotulo)
                texto = f"Art. {a.artigo} - " + "\n".join(bloco)
            chunks.append(Chunk(f"art-{a.artigo}#{parte}", texto, a.artigo, parte,
                                a.capitulo, a.capitulo_titulo, a.secao))
    return chunks


def salvar_jsonl(chunks: list[Chunk], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def main() -> None:
    artigos = [a for a in extrair_artigos(limpar_texto(carregar_texto())) if not _vetado(a)]
    print(f"{len(artigos)} artigos (vetados excluídos)")
    for nome, fn in [("artigo", chunk_por_artigo), ("paragrafo", chunk_por_paragrafo)]:
        chunks = fn(artigos)
        salvar_jsonl(chunks, Path(f"data/processed/chunks_{nome}.jsonl"))
        tam = sorted(len(c.texto) for c in chunks)
        print(f"{nome}: {len(chunks)} chunks | min={tam[0]} mediana={tam[len(tam)//2]} max={tam[-1]} chars")


if __name__ == "__main__":
    main()
