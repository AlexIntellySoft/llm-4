"""Загрузка корпоративных документов: PDF, markdown, txt.

Здесь же — чистка служебных строк. Колонтитул, который reportlab рисует на
канве, pypdf возвращает в начале текста каждой страницы; если его не убрать,
он попадает в каждый чанк и засоряет поиск.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

# «АО «Полярис Технологии» · 02_polozhenie_otpusk» и «Класс: Внутренний · стр. 1»
FOOTER_PATTERNS = (
    re.compile(r"^АО «Полярис Технологии»\s*·\s*\S+$"),
    re.compile(r"^(Класс:\s*.+·\s*)?стр\.\s*\d+$"),
)
DOC_META_KEYS = ("Документ", "Редакция", "Утверждено", "Владелец документа",
                 "Класс", "Ограничение доступа", "Источник", "Актуальность")
HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(.{3,90})$")


@dataclass
class Document:
    """Один документ корпуса до нарезки на чанки."""

    doc_id: str
    title: str
    source: str
    text: str
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def doc_class(self) -> str:
        return self.meta.get("Класс", "не указан")


def _strip_footers(raw: str) -> str:
    kept = [ln for ln in raw.splitlines()
            if not any(p.match(ln.strip()) for p in FOOTER_PATTERNS)]
    return "\n".join(kept)


def _is_structural(line: str) -> bool:
    """Строка, которую нельзя приклеивать к предыдущей: заголовок, метаданные, пункт списка."""
    s = line.strip()
    if not s:
        return True
    if HEADING_RE.match(s):
        return True
    if s.startswith(("•", "- ", "|")):
        return True
    return any(s.startswith(k + ":") for k in DOC_META_KEYS)


def _reflow(text: str) -> str:
    """Склеивает строки, разорванные вёрсткой PDF.

    Правило простое и потому предсказуемое: строка приклеивается к предыдущей,
    только если ни одна из них не структурная и предыдущая не закончена знаком
    препинания. Строки таблиц при этом сливаются в одну — pypdf всё равно отдаёт
    ячейки потоком, без границ строк (см. раздел про слабые места в отчёте).
    """
    out: list[str] = []
    for raw in text.split("\n"):
        s = raw.strip()
        if not s:
            out.append("")
            continue
        if (not out or not out[-1] or _is_structural(s) or _is_structural(out[-1])
                or out[-1].endswith((".", ":", ";", "!", "?"))):
            out.append(s)
        else:
            out[-1] = f"{out[-1]} {s}"
    return "\n".join(ln for ln in out if ln != "" or True)


def _extract_meta(text: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for ln in text.splitlines()[:14]:
        s = ln.strip()
        for key in DOC_META_KEYS:
            if s.startswith(key + ":"):
                meta[key] = s.split(":", 1)[1].strip()
    return meta


def load_pdf(path: Path) -> Document:
    reader = PdfReader(str(path))
    pages = [_strip_footers(p.extract_text() or "") for p in reader.pages]
    text = _reflow("\n".join(pages).strip())
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title = (reader.metadata.title if reader.metadata and reader.metadata.title
             else (lines[0] if lines else path.stem))
    return Document(doc_id=path.stem, title=str(title).strip(), source=path.name,
                    text=text, meta=_extract_meta(text))


def load_text(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    text = raw.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title = lines[0].lstrip("# ").strip() if lines else path.stem
    # В markdown-файлах заголовки разделов идут как «## ...» — приводим к общему виду.
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    return Document(doc_id=path.stem, title=title, source=path.name,
                    text=text, meta=_extract_meta(text))


def load_document(path: Path) -> Document:
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    if path.suffix.lower() in {".md", ".txt"}:
        return load_text(path)
    raise ValueError(f"неподдерживаемый формат: {path.name}")


def load_corpus(*dirs: Path) -> list[Document]:
    docs: list[Document] = []
    for d in dirs:
        if not d.exists():
            continue
        for path in sorted(d.iterdir()):
            if path.name.startswith(".") or path.is_dir():
                continue
            docs.append(load_document(path))
    return docs


if __name__ == "__main__":
    from src import config

    for doc in load_corpus(config.DATA_RAW):
        print(f"{doc.source:34s} {len(doc.text):6d} симв.  класс={doc.doc_class!r}")
        print(f"   {doc.title}")
