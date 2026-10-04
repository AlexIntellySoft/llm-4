"""Многоходовые диалоги для проверки памяти.

Отличие от `eval_questions.py`: там каждый вопрос самостоятельный, здесь
большинство ходов осмысленны только вместе с предыдущими. Проверяются три
разных свойства, и они специально разведены по разным ходам:

* **короткая отсылка** — «А за рубежом сколько?» сразу после вопроса о суточных.
  Достаточно помнить один предыдущий ход;
* **дальняя отсылка** — «Повтори самую первую цифру, которую ты назвал» в конце
  диалога. Формулировка намеренно лишена слов темы: поиск по такому вопросу
  не находит ничего полезного, ответить можно только из истории. Формулировки
  дорабатывались дважды, и обе правки стоит знать при чтении отчёта: сначала ходы
  были написаны как «Напомни, какими были суточные по России?» — такой вопрос
  самодостаточен, и на него правильно отвечала даже конфигурация без памяти;
  затем «самую первую цифру» модель прочла буквально и ответила «2» (первая цифра
  числа 2 500), что измеряло формулировку, а не память. Именно на исправленных
  ходах стратегии расходятся;
* **устойчивость отказа** — вопросы вне базы посреди диалога. Память не должна
  провоцировать модель отвечать на них по инерции предыдущих ходов.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.eval_questions import _contains


@dataclass
class Turn:
    question: str
    expect_any: list[str] = field(default_factory=list)
    forbid_any: list[str] = field(default_factory=list)
    expect_refusal: bool = False
    expect_source: str = ""
    kind: str = "simple"      # simple | followup | longrange | offbase
    note: str = ""


@dataclass
class Dialog:
    did: str
    topic: str
    turns: list[Turn]


DIALOGS: list[Dialog] = [
    Dialog("D1", "командировки", [
        Turn("Какие суточные выплачиваются при командировке по России?",
             ["2 500"], ["700"], expect_source="03_reglament_komandirovki.pdf",
             kind="simple"),
        Turn("А за рубежом сколько?",
             ["4 500"], ["12 000"], expect_source="03_reglament_komandirovki.pdf",
             kind="followup",
             note="без памяти уходит в нормы на проживание и отвечает 12 000 ₽"),
        Turn("Какой лимит на проживание в Москве?",
             ["8 000"], expect_source="03_reglament_komandirovki.pdf", kind="simple"),
        Turn("А в прочих городах России?",
             ["5 000"], expect_source="03_reglament_komandirovki.pdf", kind="followup"),
        Turn("В какой срок нужно подать отчёт о расходах?",
             ["3 рабочих"], expect_source="03_reglament_komandirovki.pdf", kind="simple"),
        Turn("Повтори, пожалуйста, самую первую сумму, которую ты назвал.",
             ["2 500"], kind="longrange",
             note="чистая анафора на пять ходов назад: в вопросе нет ни одного "
                  "слова темы, поиском такое не вытянуть — только памятью"),
    ]),
    Dialog("D2", "информационная безопасность", [
        Turn("Куда сообщать об инциденте информационной безопасности?",
             ["7777"], expect_source="05_poryadok_incidenty.pdf", kind="simple"),
        Turn("А в какой срок?",
             ["30 минут"], expect_source="05_poryadok_incidenty.pdf", kind="followup",
             note="без памяти уходит в документ про отпуска и просит уточнить вопрос"),
        Turn("Какое время реакции на инцидент уровня P1?",
             ["15 минут"], expect_source="05_poryadok_incidenty.pdf", kind="simple"),
        Turn("А на P3?",
             ["4 часа"], expect_source="05_poryadok_incidenty.pdf", kind="followup"),
        Turn("Какие требования к длине пароля?",
             ["14"], ["8 символов"], expect_source="01_politika_ib.pdf", kind="simple"),
        Turn("А как часто его менять?",
             ["180"], ["90 дней"], expect_source="01_politika_ib.pdf", kind="followup"),
        Turn("Повтори дословно свой самый первый ответ в этом разговоре.",
             ["7777"], kind="longrange",
             note="анафора на шесть ходов назад; формулировка без слова «номер», "
                  "иначе ответ случайно находится поиском в руководстве по адаптации"),
    ]),
    Dialog("D3", "отпуск и приём на работу", [
        Turn("Сколько календарных дней основного отпуска положено работнику?",
             ["31"], ["28 календарных"], expect_source="02_polozhenie_otpusk.pdf",
             kind="simple"),
        Turn("А если стаж в компании больше трёх лет?",
             ["5"], expect_source="02_polozhenie_otpusk.pdf", kind="followup"),
        Turn("За сколько дней подавать заявление?",
             ["14"], expect_source="02_polozhenie_otpusk.pdf", kind="followup"),
        Turn("Какой испытательный срок при приёме на работу?",
             ["2 месяца", "два месяца"], ["3 месяца"],
             expect_source="04_onboarding.pdf", kind="simple"),
        Turn("А у руководителей подразделений?",
             ["4 месяца", "четыре месяца"], expect_source="04_onboarding.pdf",
             kind="followup"),
        Turn("Сколько длится наставничество?",
             ["8 недель", "восемь недель"], expect_source="04_onboarding.pdf",
             kind="simple"),
        Turn("Назови ещё раз число из своего самого первого ответа.",
             ["31"], kind="longrange",
             note="анафора на первый ход после полной смены темы"),
    ]),
    Dialog("D4", "устойчивость отказа внутри диалога", [
        Turn("Какие суточные при командировке по России?",
             ["2 500"], expect_source="03_reglament_komandirovki.pdf", kind="simple"),
        Turn("А как испечь шарлотку с яблоками?",
             expect_refusal=True, kind="offbase",
             note="память не должна провоцировать ответ по инерции"),
        Turn("Сколько дней больничного оплачивает компания?",
             expect_refusal=True, kind="offbase"),
        Turn("Хорошо. А суточные за рубежом?",
             ["4 500"], ["12 000"], expect_source="03_reglament_komandirovki.pdf",
             kind="followup",
             note="возврат к теме после двух отказов: отсылка «за рубежом» "
                  "указывает на первый ход, а не на шарлотку"),
    ]),
]


@dataclass
class TurnVerdict:
    passed: bool
    reason: str


def grade_turn(turn: Turn, answer: str, refused: bool) -> TurnVerdict:
    if turn.expect_refusal:
        return TurnVerdict(refused, "отказался" if refused
                           else "ответил, хотя в базе ответа нет")
    if refused:
        return TurnVerdict(False, "отказался, хотя ответ в базе есть")

    bad = [f for f in turn.forbid_any if _contains(answer, f)]
    if bad:
        return TurnVerdict(False, f"чужое значение: {', '.join(bad)}")

    # Несколько вариантов формулировки одного факта считаем за «или».
    ok = any(_contains(answer, e) for e in turn.expect_any) if turn.expect_any else True
    if not ok:
        return TurnVerdict(False, f"нет ожидаемого: {', '.join(turn.expect_any)}")
    return TurnVerdict(True, "верно")


def stats() -> dict:
    kinds: dict[str, int] = {}
    for d in DIALOGS:
        for t in d.turns:
            kinds[t.kind] = kinds.get(t.kind, 0) + 1
    return {"dialogs": len(DIALOGS),
            "turns": sum(len(d.turns) for d in DIALOGS), "by_kind": kinds}


if __name__ == "__main__":
    s = stats()
    print(f"диалогов: {s['dialogs']}, ходов: {s['turns']}")
    for k, n in s["by_kind"].items():
        print(f"   {k:10s} {n}")
    for d in DIALOGS:
        print(f"\n{d.did} — {d.topic}")
        for i, t in enumerate(d.turns, 1):
            print(f"  {i}. [{t.kind:9s}] {t.question}")
