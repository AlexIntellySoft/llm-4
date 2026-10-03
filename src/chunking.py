"""Нарезка документов на чанки.

Два решения, которые заметно влияют на качество поиска:

1. Документ сначала режется по разделам («1. Область применения», «2. ...»),
   и только потом внутри раздела — RecursiveCharacterTextSplitter. Так граница
   чанка почти никогда не проходит посреди смыслового блока.
2. В начало каждого чанка подставляется «хлебная крошка» — название документа
   и раздела. Это дешёвая страховка: после извлечения текста из PDF таблицы
   превращаются в поток ячеек, и чанк «2 500 ₽ 3 500 ₽ 4 500 ₽» сам по себе
   не находится по запросу «суточные», а с крошкой «Регламент служебных
   командировок / Нормы суточных» — находится.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ChunkConfig
from src.loaders import HEADING_RE, Document

SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    chunk_id: str
    text: str          # то, что уходит в эмбеддинг (возможно, с крошкой)
    raw_text: str      # исходный фрагмент без крошки — его показываем пользователю
    source: str
    doc_id: str
    doc_title: str
    section: str
    doc_class: str
    position: int

    def metadata(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d.pop("raw_text")
        return d


def split_sections(doc: Document) -> list[tuple[str, str]]:
    """Делит текст документа на (название раздела, текст раздела)."""
    sections: list[tuple[str, list[str]]] = []
    current = "Общие сведения"
    buf: list[str] = []
    for line in doc.text.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            if buf:
                sections.append((current, buf))
            current = f"{m.group(1)}. {m.group(2)}".strip()
            buf = []
        else:
            buf.append(line)
    if buf:
        sections.append((current, buf))
    out = []
    for name, lines in sections:
        body = "\n".join(lines).strip()
        if body:
            out.append((name, body))
    return out


def chunk_document(doc: Document, cfg: ChunkConfig) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        keep_separator=True,
    )
    chunks: list[Chunk] = []
    position = 0
    for section, body in split_sections(doc):
        for piece in splitter.split_text(body):
            piece = piece.strip()
            if len(piece) < 30:        # обрывки вроде «3 года.» смысла не несут
                continue
            crumb = f"{doc.title} / {section}"
            text = f"{crumb}\n{piece}" if cfg.contextual_header else piece
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}#{position}",
                text=text,
                raw_text=piece,
                source=doc.source,
                doc_id=doc.doc_id,
                doc_title=doc.title,
                section=section,
                doc_class=doc.doc_class,
                position=position,
            ))
            position += 1
    return chunks


def chunk_corpus(docs: list[Document], cfg: ChunkConfig) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, cfg))
    return out


def stats(chunks: list[Chunk]) -> dict:
    lens = [len(c.raw_text) for c in chunks]
    by_source: dict[str, int] = {}
    for c in chunks:
        by_source[c.source] = by_source.get(c.source, 0) + 1
    return {
        "chunks": len(chunks),
        "avg_len": round(sum(lens) / len(lens), 1) if lens else 0,
        "min_len": min(lens) if lens else 0,
        "max_len": max(lens) if lens else 0,
        "by_source": by_source,
    }


if __name__ == "__main__":
    from src import config
    from src.loaders import load_corpus

    docs = load_corpus(config.DATA_RAW)
    chunks = chunk_corpus(docs, config.CHUNK)
    s = stats(chunks)
    print(f"чанков: {s['chunks']}, средняя длина {s['avg_len']} симв. "
          f"(от {s['min_len']} до {s['max_len']})")
    for src, n in s["by_source"].items():
        print(f"  {src:34s} {n:3d}")
    print("\n--- пример чанка ---")
    ex = next(c for c in chunks if "суточны" in c.raw_text.lower())
    print(f"[{ex.chunk_id}] раздел: {ex.section}")
    print(ex.text[:420])
