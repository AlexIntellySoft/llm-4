"""Рендер исходных markdown-документов корпуса в PDF.

Исходники лежат в data/source_md (их удобно читать и править в git),
а пайплайн RAG работает уже с data/raw — тем, что похоже на реальную
корпоративную базу: PDF-регламенты плюс пара текстовых файлов.

Запуск:  python -m src.build_corpus
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    ListFlowable,
    ListItem,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "source_md"
OUT_DIR = ROOT / "data" / "raw"
OUT_OPTIONAL_DIR = ROOT / "data" / "raw_optional"

# Документы, которые остаются текстовыми файлами базы знаний (их не рендерим в PDF).
AS_TEXT = {"08_faq.md", "09_glossarij.txt"}
# Документ-зонд для эксперимента с prompt injection: лежит отдельно,
# в основной индекс не попадает.
OPTIONAL = {"10_injection_probe.md"}

COMPANY = "АО «Полярис Технологии»"


def register_fonts() -> None:
    """DejaVu поставляется вместе с matplotlib и умеет кириллицу."""
    import matplotlib

    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(ttf / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(ttf / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")


def styles() -> dict[str, ParagraphStyle]:
    base = dict(fontName="DejaVu", leading=14, fontSize=9.5)
    return {
        "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=16,
                                leading=20, spaceAfter=4 * mm),
        "meta": ParagraphStyle("meta", fontName="DejaVu", fontSize=8,
                               leading=11, textColor=colors.HexColor("#555555")),
        "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=11.5, leading=15,
                             spaceBefore=5 * mm, spaceAfter=2 * mm),
        "body": ParagraphStyle("body", alignment=TA_JUSTIFY, spaceAfter=2.5 * mm, **base),
        "li": ParagraphStyle("li", spaceAfter=1.2 * mm, **base),
        "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=8.5, leading=11),
        "cellh": ParagraphStyle("cellh", fontName="DejaVu-Bold", fontSize=8.5, leading=11),
    }


def inline(text: str) -> str:
    """Минимальный markdown-инлайн: **жирный** и экранирование XML."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def parse(md: str) -> tuple[str, list[str], list[tuple]]:
    """Разбирает markdown в (заголовок, строки метаданных, блоки контента)."""
    lines = md.splitlines()
    title, meta, blocks = "", [], []
    i = 0

    while i < len(lines) and not lines[i].startswith("# "):
        i += 1
    if i < len(lines):
        title = lines[i][2:].strip()
        i += 1

    # Метаданные — непустые строки «Ключ: значение» до первого раздела.
    while i < len(lines) and not lines[i].startswith("## "):
        s = lines[i].strip()
        if s:
            meta.append(s)
        i += 1

    para: list[str] = []
    bullets: list[str] = []
    table: list[list[str]] = []

    def flush() -> None:
        nonlocal para, bullets, table
        if para:
            blocks.append(("p", " ".join(para)))
            para = []
        if bullets:
            blocks.append(("ul", bullets))
            bullets = []
        if table:
            blocks.append(("table", table))
            table = []

    for line in lines[i:]:
        s = line.strip()
        if s.startswith("## "):
            flush()
            blocks.append(("h2", s[3:].strip()))
        elif s.startswith("| "):
            if para or bullets:
                flush()
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not all(set(c) <= {"-", ":", " "} and c for c in cells):
                table.append(cells)
        elif s.startswith("- "):
            if para or table:
                flush()
            bullets.append(s[2:].strip())
        elif not s:
            flush()
        else:
            if bullets or table:
                flush()
            para.append(s)
    flush()
    return title, meta, blocks


def build_pdf(md_path: Path, out_path: Path) -> None:
    st = styles()
    title, meta, blocks = parse(md_path.read_text(encoding="utf-8"))
    doc_class = next((m.split(":", 1)[1].strip() for m in meta if m.startswith("Класс:")), "")

    story: list = [Paragraph(inline(title), st["title"])]
    for m in meta:
        story.append(Paragraph(inline(m), st["meta"]))
    story.append(Spacer(1, 5 * mm))

    for kind, payload in blocks:
        if kind == "h2":
            story.append(Paragraph(inline(payload), st["h2"]))
        elif kind == "p":
            story.append(Paragraph(inline(payload), st["body"]))
        elif kind == "ul":
            story.append(ListFlowable(
                [ListItem(Paragraph(inline(x), st["li"]), leftIndent=6 * mm)
                 for x in payload],
                bulletType="bullet", bulletFontName="DejaVu", bulletFontSize=7,
                start="•", leftIndent=5 * mm, spaceAfter=2.5 * mm,
            ))
        elif kind == "table":
            head, *rows = payload
            data = [[Paragraph(inline(c), st["cellh"]) for c in head]]
            data += [[Paragraph(inline(c), st["cell"]) for c in r] for r in rows]
            ncols = len(head)
            avail = 170 * mm
            widths = ([avail * 0.28] + [(avail * 0.72) / (ncols - 1)] * (ncols - 1)
                      if ncols > 1 else [avail])
            tbl = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
            tbl.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b0b0")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eceff3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story += [tbl, Spacer(1, 4 * mm)]

    def decorate(canvas, doc_) -> None:
        canvas.saveState()
        canvas.setFont("DejaVu", 7)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(20 * mm, 12 * mm, f"{COMPANY} · {md_path.stem}")
        right = f"Класс: {doc_class} · стр. {canvas.getPageNumber()}" if doc_class \
            else f"стр. {canvas.getPageNumber()}"
        canvas.drawRightString(190 * mm, 12 * mm, right)
        canvas.setStrokeColor(colors.HexColor("#cccccc"))
        canvas.line(20 * mm, 15 * mm, 190 * mm, 15 * mm)
        canvas.restoreState()

    # invariant=1 — фиксированная дата создания внутри PDF: иначе каждая пересборка
    # меняет все файлы корпуса и засоряет историю git.
    pdf = BaseDocTemplate(str(out_path), pagesize=A4, title=title, author=COMPANY,
                          leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=18 * mm, bottomMargin=20 * mm, invariant=1)
    frame = Frame(20 * mm, 20 * mm, 170 * mm, A4[1] - 38 * mm, id="main")
    pdf.addPageTemplates([PageTemplate(id="std", frames=[frame], onPage=decorate)])
    pdf.build(story)


def main() -> None:
    register_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_OPTIONAL_DIR.mkdir(parents=True, exist_ok=True)

    for path in sorted(SRC_DIR.iterdir()):
        if path.name.startswith("."):
            continue
        target_dir = OUT_OPTIONAL_DIR if path.name in OPTIONAL else OUT_DIR
        if path.name in AS_TEXT:
            shutil.copy2(path, target_dir / path.name)
            print(f"  copy  {path.name}")
        else:
            out = target_dir / (path.stem + ".pdf")
            build_pdf(path, out)
            print(f"  pdf   {out.name:38s} {out.stat().st_size / 1024:6.1f} KB")

    print(f"\nОсновной корпус: {OUT_DIR}")
    print(f"Зонд для эксперимента с инъекцией: {OUT_OPTIONAL_DIR}")


if __name__ == "__main__":
    main()
