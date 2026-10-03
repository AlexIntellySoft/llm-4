"""Сборка RAG-цепочки: вопрос → вектор → поиск top-K → промпт → локальная LLM.

Всё, что нужно для измерений, возвращается в одном объекте RagAnswer:
отдельно время поиска и время генерации, список источников, расход токенов.
Без этого нельзя честно обсуждать latency — большая часть задержки
приходится на генерацию, а не на поиск, и это видно только в разбивке.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src import config
from src.llm import Completion, complete
from src.prompts import REFUSAL, build_messages
from src.vectorstore import Hit, search


@dataclass
class RagAnswer:
    question: str
    answer: str
    hits: list[Hit] = field(default_factory=list)
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    completion: Completion | None = None
    variant: str = "strict"
    model: str = ""
    top_k: int = 0

    @property
    def total_s(self) -> float:
        return round(self.retrieval_s + self.generation_s, 2)

    @property
    def refused(self) -> bool:
        """Отказ считаем по сути, а не по точному совпадению строки."""
        a = self.answer.lower()
        if REFUSAL.lower().rstrip(".") in a:
            return True
        markers = ("нет ответа на этот вопрос", "нет информации", "не содержит",
                   "не знаю", "отсутствует в", "не нашёл", "не найдено",
                   "нет данных", "не располагаю")
        return any(m in a for m in markers)

    @property
    def sources(self) -> list[str]:
        seen, out = set(), []
        for h in self.hits:
            c = h.citation()
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def to_record(self) -> dict:
        c = self.completion
        return {
            "question": self.question,
            "answer": self.answer,
            "model": self.model,
            "variant": self.variant,
            "top_k": self.top_k,
            "refused": self.refused,
            "sources": "; ".join(self.sources),
            "top1_similarity": round(self.hits[0].similarity, 4) if self.hits else None,
            "retrieval_s": round(self.retrieval_s, 3),
            "generation_s": round(self.generation_s, 2),
            "total_s": self.total_s,
            "prompt_tokens": c.prompt_tokens if c else 0,
            "completion_tokens": c.completion_tokens if c else 0,
            "reasoning_tokens": c.reasoning_tokens if c else 0,
            "tokens_per_second": c.tokens_per_second if c else 0.0,
            "error": c.error if c else "",
        }


def answer(question: str, top_k: int | None = None, model: str | None = None,
           variant: str = "strict", embedder: str | None = None,
           max_distance: float | None = None, chunk_cfg=None,
           temperature: float | None = None,
           enable_thinking: bool | None = None) -> RagAnswer:
    top_k = config.RAG.top_k if top_k is None else top_k
    model = model or config.DEFAULT_LLM

    t0 = time.perf_counter()
    hits: list[Hit] = []
    if variant != "no_rag":
        hits = search(question, top_k=top_k, embedder_name=embedder,
                      chunk_cfg=chunk_cfg,
                      max_distance=(config.RAG.max_distance if max_distance is None
                                    else max_distance))
    retrieval_s = time.perf_counter() - t0

    messages = build_messages(question, hits, variant=variant)
    t1 = time.perf_counter()
    comp = complete(messages, model=model, temperature=temperature,
                    enable_thinking=enable_thinking)
    generation_s = time.perf_counter() - t1

    text = comp.text or (f"[ошибка генерации] {comp.error}" if comp.error else "")
    return RagAnswer(question=question, answer=text, hits=hits,
                     retrieval_s=retrieval_s, generation_s=generation_s,
                     completion=comp, variant=variant, model=model, top_k=top_k)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Сколько календарных дней основного отпуска?"
    r = answer(q)
    print(f"ВОПРОС: {r.question}\n")
    print(r.answer)
    print(f"\n— найдено {len(r.hits)} фрагментов, "
          f"поиск {r.retrieval_s * 1000:.0f} мс, генерация {r.generation_s:.1f} с, "
          f"модель {r.completion.model}, отказ={r.refused}")
