"""
WhatsApp Formatter (post-hook)
------------------------------

Rede de seguranca de formatacao. Mesmo que o modelo escape e emita markdown
padrao (**bold**, # heading, [text](url), tabela), este hook normaliza para
WhatsApp-safe e garante chunking em ate 4 blocos separados por ---CHUNK---.

Regras de saida:
- *bold* (asterisco simples), _italic_, ~strike~
- Headings viram linhas em *bold*
- Links [texto](url) viram "texto: url"
- Tabelas markdown viram listas simples
- Markdown orfao (**, __, ~~) e removido
- Resposta longa e quebrada em ate 4 chunks; o transporte separa por ---CHUNK---
"""

from __future__ import annotations

import re

from agno.run.agent import RunOutput

CHUNK_SEPARATOR = "\n\n---CHUNK---\n\n"
_MAX_CHUNKS = 4
_TARGET_CHUNK_CHARS = 600
_HARD_CHUNK_CHARS = 900


def _normalize_markdown(text: str) -> str:
    # [texto](url) -> "texto: url"
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1: \2", text)

    # tabelas markdown: tira pipes e linhas so de dashes
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?", stripped):
            continue  # separador de tabela
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            lines.append(" — ".join(c for c in cells if c))
        else:
            lines.append(line)
    text = "\n".join(lines)

    # headings (# ## ###) -> *bold* line
    text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)

    # bold duplo/italico duplo -> simples
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"_\1_", text, flags=re.DOTALL)

    # blockquotes "> " no comeco da linha
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)

    # remove markers orfaos que sobraram
    text = re.sub(r"\*\*+", "", text)
    text = re.sub(r"__+", "", text)
    text = re.sub(r"~~+", "", text)

    return text


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def _pack_into_chunks(text: str) -> list[str]:
    """Agrupa paragrafos em ate _MAX_CHUNKS chunks de ~_TARGET_CHUNK_CHARS."""
    paras = _paragraphs(text)
    if not paras:
        return [text.strip()]

    chunks: list[str] = []
    current = ""
    for para in paras:
        candidate = f"{current}\n\n{para}" if current else para
        if len(current) == 0:
            current = para
        elif len(candidate) <= _HARD_CHUNK_CHARS:
            current = candidate
        else:
            chunks.append(current)
            current = para
            if len(chunks) == _MAX_CHUNKS - 1:
                # ultimo chunk absorve o resto
                break
    if current:
        chunks.append(current)

    # se ainda sobrou paragrafo, concatena no ultimo
    consumed = sum(len(_paragraphs(c)) for c in chunks)
    if consumed < len(paras):
        remainder = "\n\n".join(paras[consumed:])
        chunks[-1] = f"{chunks[-1]}\n\n{remainder}" if chunks else remainder

    return chunks[:_MAX_CHUNKS]


def _needs_chunking(text: str) -> bool:
    return len(text) > _TARGET_CHUNK_CHARS and CHUNK_SEPARATOR not in text


def whatsapp_format(run_output: RunOutput) -> None:
    """Post-hook: normalize markdown to WhatsApp-safe and split into <=4 chunks."""
    content = run_output.content
    if not isinstance(content, str) or not content.strip():
        return

    # Se modelo ja marcou chunks, respeita (mas valida limite)
    if CHUNK_SEPARATOR in content:
        parts = [p.strip() for p in content.split(CHUNK_SEPARATOR) if p.strip()]
        parts = [_normalize_markdown(p) for p in parts]
        if len(parts) > _MAX_CHUNKS:
            head = parts[: _MAX_CHUNKS - 1]
            tail = "\n\n".join(parts[_MAX_CHUNKS - 1 :])
            parts = head + [tail]
        run_output.content = CHUNK_SEPARATOR.join(parts)
        return

    normalized = _normalize_markdown(content).strip()
    if not _needs_chunking(normalized):
        run_output.content = normalized
        return

    chunks = _pack_into_chunks(normalized)
    run_output.content = CHUNK_SEPARATOR.join(chunks)
