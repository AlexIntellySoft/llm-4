"""Тестовый набор вопросов и автоматическая проверка ответов.

Категории соответствуют заданию и расширяют его:

* in_base      — ответ точно есть в документах;
* out_of_base  — ответа в базе нет, ожидается отказ;
* conflict     — корпоративная норма расходится с тем, что «знает» модель
                 из общих знаний; ожидается корпоративное значение;
* control      — корпоративная норма совпадает с законом: страховка от вывода
                 «модель просто всегда повторяет документ»;
* confidential — ответ есть, но в документе класса SECRET; показывает, что
                 у RAG из коробки нет разграничения доступа.

Проверка намеренно простая — по подстрокам, а не LLM-судьёй: её легко
воспроизвести и глазами проверить. Ограничение честное: формулировку «тридцать
один день» вместо «31» такая проверка не зачтёт, поэтому в отчёте рядом с
метрикой приводятся сами ответы.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TestQuestion:
    qid: str
    category: str
    question: str
    expect_source: str = ""          # ожидаемый файл-источник
    expect_any: list[str] = field(default_factory=list)   # должно встретиться всё из списка
    forbid_any: list[str] = field(default_factory=list)   # не должно встретиться ничего
    expect_refusal: bool = False
    note: str = ""                   # что именно проверяет вопрос


QUESTIONS: list[TestQuestion] = [
    # ---------- 1. Ответ есть в базе ----------
    TestQuestion("B1", "in_base",
                 "Куда и в какой срок работник обязан сообщить об инциденте информационной безопасности?",
                 "05_poryadok_incidenty.pdf", ["7777", "30 минут"],
                 note="факт из двух разных абзацев одного раздела"),
    TestQuestion("B2", "in_base",
                 "Какой лимит на проживание в сутки при командировке в Санкт-Петербург?",
                 "03_reglament_komandirovki.pdf", ["8 000"],
                 note="значение внутри таблицы, которая при извлечении из PDF схлопывается"),
    TestQuestion("B3", "in_base",
                 "Кто должен согласовать закупку на сумму 1 200 000 рублей?",
                 "06_reglament_zakupki.pdf", ["финансовый директор"],
                 note="нужно выбрать нужный диапазон в таблице лимитов"),
    TestQuestion("B4", "in_base",
                 "С какого месяца работы действует ДМС и какая страховая компания его обслуживает?",
                 "08_faq.md", ["МедАльянс"],
                 note="источник — markdown-файл базы знаний, а не PDF"),
    TestQuestion("B5", "in_base",
                 "Что такое «зелёный контур» в терминологии компании?",
                 "09_glossarij.txt", ["изолирован"],
                 note="источник — txt-глоссарий; проверяет работу с внутренним термином"),
    TestQuestion("B6", "in_base",
                 "За сколько дней до начала отпуска нужно подать заявление и через какую систему?",
                 "02_polozhenie_otpusk.pdf", ["14", "HR-портал"],
                 note="два факта в одном ответе"),
    TestQuestion("B7", "in_base",
                 "Какое время реакции SOC на инцидент уровня P1?",
                 "05_poryadok_incidenty.pdf", ["15 минут"],
                 note="значение из таблицы классификации"),
    TestQuestion("B8", "in_base",
                 "Сколько коммерческих предложений нужно при закупке на 400 000 рублей?",
                 "06_reglament_zakupki.pdf", ["3", "три"],
                 note="требует сравнить сумму с порогом 300 000 ₽"),

    # ---------- 2. Ответа в базе нет ----------
    TestQuestion("O1", "out_of_base",
                 "Как испечь шарлотку с яблоками?",
                 expect_refusal=True,
                 note="классический вопрос не по теме"),
    TestQuestion("O2", "out_of_base",
                 "Сколько дней больничного оплачивает компания и в каком размере?",
                 expect_refusal=True,
                 note="правдоподобный HR-вопрос, которого в корпусе нет — самый опасный случай"),
    TestQuestion("O3", "out_of_base",
                 "Какой порядок командировок установлен в дочерней компании в Казахстане?",
                 expect_refusal=True,
                 note="тема близка к корпусу, но объект другой"),
    TestQuestion("O4", "out_of_base",
                 "Кто выиграл Лигу чемпионов УЕФА в 2024 году?",
                 expect_refusal=True,
                 note="общие знания модели, к компании отношения не имеют"),
    TestQuestion("O5", "out_of_base",
                 "Какая сейчас средняя рыночная зарплата Python-разработчика в Москве?",
                 expect_refusal=True,
                 note="рядом есть документ про оклады — проверяем, не подменит ли модель вопрос"),

    # ---------- 3. Конфликт с общими знаниями модели ----------
    TestQuestion("C1", "conflict",
                 "Сколько календарных дней основного ежегодного отпуска положено работнику?",
                 "02_polozhenie_otpusk.pdf", ["31"], forbid_any=["28 календарных", "28 дней"],
                 note="ТК РФ даёт 28 дней, коллективный договор компании — 31"),
    TestQuestion("C2", "conflict",
                 "Какие суточные выплачиваются при командировке по России?",
                 "03_reglament_komandirovki.pdf", ["2 500"], forbid_any=["700"],
                 note="необлагаемый лимит НК РФ — 700 ₽, норма компании — 2 500 ₽"),
    TestQuestion("C3", "conflict",
                 "Какой испытательный срок устанавливается рядовому работнику при приёме на работу?",
                 "04_onboarding.pdf", ["2 месяца", "два месяца"],
                 forbid_any=["3 месяца", "три месяца"],
                 note="ТК РФ допускает до 3 месяцев, в компании — 2"),
    TestQuestion("C4", "conflict",
                 "Какие требования к длине пароля и как часто его менять?",
                 "01_politika_ib.pdf", ["14", "180"],
                 forbid_any=["8 символов", "90 дней", "каждые 90"],
                 note="распространённая практика 8 символов / 90 дней против 14 / 180"),

    # ---------- 4. Контроль: норма компании совпадает с законом ----------
    TestQuestion("K1", "control",
                 "В какой срок компания уведомляет Роскомнадзор об утечке персональных данных?",
                 "05_poryadok_incidenty.pdf", ["24"],
                 note="здесь документ и закон совпадают: проверяем, что модель не ломает верный факт"),

    # ---------- 5. Конфиденциальный документ ----------
    TestQuestion("S1", "confidential",
                 "Какая вилка оклада у руководителя отдела разработки?",
                 "07_grejdy_konfidencialno.pdf", ["420 000"],
                 note="документ класса SECRET: ответ будет выдан любому — у RAG нет разграничения доступа"),
    TestQuestion("S2", "confidential",
                 "Какая максимальная квартальная премия в компании?",
                 "07_grejdy_konfidencialno.pdf", ["20"],
                 note="тот же SECRET-документ, другой раздел"),
]


def _norm(text: str) -> str:
    """Нижний регистр + склейка разрядов: «2 500», «2 500» и «2500» становятся одним."""
    t = text.lower().replace(" ", " ").replace(" ", " ")
    return re.sub(r"(?<=\d)[\s](?=\d)", "", t)


def _contains(haystack: str, needle: str) -> bool:
    """Проверка вхождения факта с учётом русского словоизменения.

    Прямое сравнение подстрок проваливается на падежах: ожидаем «финансовый
    директор», а модель отвечает «финансовым директором» — ответ верный, проверка
    его не зачитывает. Так и случилось в первом прогоне эксперимента с Top-K
    (см. отчёт), поэтому сравнение идёт по основам слов.

    Фраза превращается в регулярное выражение: числа должны совпасть точно,
    слова — по первым пяти буквам, порядок и соседство слов обязательны.
    Соседство важно: без него «3 месяца» находилось бы в ответе, где «3» —
    это номер раздела, а «месяца» из другого предложения.
    """
    h, n = _norm(haystack), _norm(needle)
    if n in h:
        return True
    tokens = re.findall(r"\d+|[^\W\d_]+", n)
    if not tokens:
        return False
    parts = []
    for tok in tokens:
        if tok.isdigit():
            parts.append(re.escape(tok))
        elif len(tok) <= 4:
            parts.append(re.escape(tok) + r"[^\W\d_]*")
        else:
            parts.append(re.escape(tok[:5]) + r"[^\W\d_]*")
    return re.search(r"\W+".join(parts), h) is not None


@dataclass
class Verdict:
    qid: str
    category: str
    passed: bool
    reason: str
    refused: bool
    source_ok: bool | None


def grade(q: TestQuestion, answer_text: str, refused: bool,
          sources: list[str]) -> Verdict:
    source_ok: bool | None = None
    if q.expect_source:
        source_ok = any(q.expect_source in s for s in sources)

    if q.expect_refusal:
        passed = refused
        reason = "отказался отвечать" if passed else "ответил, хотя в базе ответа нет"
        return Verdict(q.qid, q.category, passed, reason, refused, source_ok)

    if refused:
        return Verdict(q.qid, q.category, False,
                       "отказался, хотя ответ в базе есть", refused, source_ok)

    # expect_any: варианты формулировки одного и того же факта разделяем по смыслу —
    # для C3 подходит «2 месяца» ИЛИ «два месяца», для B1 нужны оба факта.
    if q.qid in {"C3", "B8"}:
        ok_expect = any(_contains(answer_text, e) for e in q.expect_any)
        missing = [] if ok_expect else q.expect_any
    else:
        missing = [e for e in q.expect_any if not _contains(answer_text, e)]
        ok_expect = not missing

    bad = [f for f in q.forbid_any if _contains(answer_text, f)]

    if bad:
        return Verdict(q.qid, q.category, False,
                       f"подставил значение из общих знаний: {', '.join(bad)}",
                       refused, source_ok)
    if not ok_expect:
        return Verdict(q.qid, q.category, False,
                       f"нет ожидаемого значения: {', '.join(missing)}", refused, source_ok)
    return Verdict(q.qid, q.category, True, "верно", refused, source_ok)


def by_category() -> dict[str, list[TestQuestion]]:
    out: dict[str, list[TestQuestion]] = {}
    for q in QUESTIONS:
        out.setdefault(q.category, []).append(q)
    return out


if __name__ == "__main__":
    for cat, qs in by_category().items():
        print(f"\n{cat} ({len(qs)}):")
        for q in qs:
            print(f"  {q.qid}  {q.question}")
    print(f"\nвсего вопросов: {len(QUESTIONS)}")
