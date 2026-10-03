"""Перепроверка сохранённых ответов новым грейдером — без повторных запросов к LLM.

Нужна после правки `_contains` в src/eval_questions.py: первая версия сравнивала
подстроки буквально и не зачитывала верные ответы с другим падежом
(«финансовым директором» вместо «финансовый директор»). Ответы моделей при этом
никуда не делись — они лежат в outputs/*.json, поэтому достаточно пересчитать
вердикты, а не прогонять модели заново.

Оригиналы сохраняются в outputs/pre_regrade/, чтобы в отчёте можно было показать
и исходные цифры, и исправленные.

    python tools/regrade.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.eval_questions import QUESTIONS, grade

OUT = config.OUTPUTS
BACKUP = OUT / "pre_regrade"
INDEX = {q.qid: q for q in QUESTIONS}
FILES = ["exp_main", "exp_prompts", "exp_topk", "exp_models"]


def regrade_rows(rows: list[dict]) -> int:
    changed = 0
    for r in rows:
        q = INDEX.get(r.get("qid", ""))
        if q is None:
            continue
        sources = [s.strip() for s in str(r.get("sources", "")).split(";") if s.strip()]
        v = grade(q, r.get("answer", ""), bool(r.get("refused")), sources)
        if r.get("passed") != v.passed:
            changed += 1
            r["passed_before_regrade"] = r.get("passed")
            r["verdict_before_regrade"] = r.get("verdict")
        r["passed"], r["verdict"] = v.passed, v.reason
    return changed


def rebuild_summary(rows: list[dict], old: dict) -> dict:
    cats: dict[str, dict] = {}
    for r in rows:
        c = cats.setdefault(r["category"], {"total": 0, "passed": 0, "latency": []})
        c["total"] += 1
        c["passed"] += int(r["passed"])
        c["latency"].append(r["total_s"])
    for c in cats.values():
        c["pass_rate"] = round(c["passed"] / c["total"], 3)
        c["avg_latency_s"] = round(sum(c["latency"]) / len(c["latency"]), 2)
        c.pop("latency")
    new = dict(old)
    new.update(passed=sum(r["passed"] for r in rows), total=len(rows),
               pass_rate=round(sum(r["passed"] for r in rows) / len(rows), 3),
               by_category=cats)
    return new


def walk(node) -> int:
    """Рекурсивно находит все блоки {"summary": ..., "rows": ...} и пересчитывает их."""
    changed = 0
    if isinstance(node, dict):
        if isinstance(node.get("rows"), list) and node["rows"] and \
                isinstance(node["rows"][0], dict) and "qid" in node["rows"][0]:
            changed += regrade_rows(node["rows"])
            if isinstance(node.get("summary"), dict):
                node["summary"] = rebuild_summary(node["rows"], node["summary"])
        for v in node.values():
            changed += walk(v)
    elif isinstance(node, list):
        for v in node:
            changed += walk(v)
    return changed


def main() -> None:
    BACKUP.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in FILES:
        path = OUT / f"{name}.json"
        if not path.exists():
            print(f"  пропуск {name}: файла нет")
            continue
        if not (BACKUP / path.name).exists():
            shutil.copy2(path, BACKUP / path.name)
        data = json.loads(path.read_text(encoding="utf-8"))
        changed = walk(data)
        data["regraded"] = True
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        total += changed
        print(f"  {name}: изменено вердиктов — {changed}")
    print(f"\nвсего изменено: {total}; оригиналы в {BACKUP.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
