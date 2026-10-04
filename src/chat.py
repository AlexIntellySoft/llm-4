"""Диалоговый RAG: та же цепочка, но с памятью и переформулировкой запроса.

Ключевое отличие от одноходового `src/rag.py` — два разных текста вопроса:

* **запрос в поиск** — переформулированный, самостоятельный. «А за рубежом
  сколько?» превращается в «Какие суточные при командировке за рубежом?»,
  иначе вектор такого вопроса не содержит ни слова «суточные», ни «командировка»,
  и поиск уходит в случайный документ.
* **вопрос в промпт** — исходный, дословный. Модель видит историю целиком
  и разрешает отсылку сама; подменять формулировку работника незачем, а при
  неудачной переформулировке это ещё и исказило бы вопрос.

Переформулировка стоит отдельного запроса к модели, и это видно в замерах:
на короткие ответы она удваивает задержку. Поэтому она включается только
при непустой истории.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage

from src import config
from src.llm import Completion, complete
from src.memory import Memory, NoMemory, count_tokens, make_memory
from src.prompts import REFUSAL, build_contextualize_messages, build_messages
from src.vectorstore import Hit, search


def to_dicts(messages: list[BaseMessage]) -> list[dict]:
    return [{"role": "assistant" if isinstance(m, AIMessage) else "user",
             "content": str(m.content)} for m in messages]


@dataclass
class ChatAnswer:
    question: str
    search_query: str
    answer: str
    hits: list[Hit] = field(default_factory=list)
    contextualize_s: float = 0.0
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    completion: Completion | None = None
    memory_stats: dict = field(default_factory=dict)
    turn: int = 0

    @property
    def rewritten(self) -> bool:
        return self.search_query.strip().lower() != self.question.strip().lower()

    @property
    def total_s(self) -> float:
        return round(self.contextualize_s + self.retrieval_s + self.generation_s, 2)

    @property
    def refused(self) -> bool:
        a = self.answer.lower()
        if REFUSAL.lower().rstrip(".") in a:
            return True
        markers = ("нет ответа на этот вопрос", "нет информации", "не содержит",
                   "не знаю", "отсутствует в", "не нашёл", "не найдено",
                   "нет данных", "не располагаю", "уточните")
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
            "turn": self.turn,
            "question": self.question,
            "search_query": self.search_query,
            "rewritten": self.rewritten,
            "answer": self.answer,
            "refused": self.refused,
            "sources": "; ".join(self.sources),
            "top1_similarity": round(self.hits[0].similarity, 4) if self.hits else None,
            "contextualize_s": round(self.contextualize_s, 2),
            "retrieval_s": round(self.retrieval_s, 3),
            "generation_s": round(self.generation_s, 2),
            "total_s": self.total_s,
            "prompt_tokens": c.prompt_tokens if c else 0,
            "completion_tokens": c.completion_tokens if c else 0,
            **{f"mem_{k}": v for k, v in self.memory_stats.items()},
        }


class ChatSession:
    """Один диалог: держит память и прогоняет ходы через RAG."""

    def __init__(self, memory: Memory | str = "buffer", model: str | None = None,
                 top_k: int | None = None, variant: str = "strict",
                 contextualize: bool = True, embedder: str | None = None,
                 temperature: float | None = None,
                 max_distance: float | None = None) -> None:
        self.memory = make_memory(memory) if isinstance(memory, str) else memory
        self.model = model or config.DEFAULT_LLM
        self.top_k = config.RAG.top_k if top_k is None else top_k
        self.variant = variant
        self.contextualize = contextualize
        self.embedder = embedder
        self.temperature = temperature
        self.max_distance = max_distance
        self.turn = 0
        if isinstance(self.memory, type(None)):
            self.memory = NoMemory()

    # ------------------------------------------------------------------ #

    def _rewrite(self, question: str, history: list[dict]) -> tuple[str, float]:
        """Уточняющий вопрос → самостоятельный поисковый запрос."""
        if not self.contextualize or not history:
            return question, 0.0
        t0 = time.perf_counter()
        comp = complete(build_contextualize_messages(question, history),
                        model=self.model, temperature=0.0, max_tokens=120)
        elapsed = time.perf_counter() - t0
        text = (comp.text or "").strip().strip('"').splitlines()
        rewritten = text[0].strip() if text else ""
        # Страховка от мусора: пустая строка, слишком длинный «ответ» вместо
        # запроса или потеря смысла — возвращаемся к исходному вопросу.
        if not rewritten or len(rewritten) > 300:
            return question, elapsed
        return rewritten, elapsed

    def ask(self, question: str) -> ChatAnswer:
        self.turn += 1
        history = to_dicts(self.memory.messages())

        search_query, ctx_s = self._rewrite(question, history)

        t0 = time.perf_counter()
        hits = search(search_query, top_k=self.top_k, embedder_name=self.embedder,
                      max_distance=(config.RAG.max_distance
                                    if self.max_distance is None else self.max_distance))
        retrieval_s = time.perf_counter() - t0

        messages = build_messages(question, hits, variant=self.variant,
                                  history=history,
                                  history_prefix=self.memory.prefix())
        t1 = time.perf_counter()
        comp = complete(messages, model=self.model, temperature=self.temperature)
        generation_s = time.perf_counter() - t1

        text = comp.text or (f"[ошибка генерации] {comp.error}" if comp.error else "")
        self.memory.add_turn(question, text)

        return ChatAnswer(
            question=question, search_query=search_query, answer=text, hits=hits,
            contextualize_s=ctx_s, retrieval_s=retrieval_s,
            generation_s=generation_s, completion=comp,
            memory_stats=self.memory.stats(), turn=self.turn,
        )

    def reset(self) -> None:
        self.memory.reset()
        self.turn = 0


if __name__ == "__main__":
    dialog = [
        "Какие суточные при командировке по России?",
        "А за рубежом сколько?",
        "Куда сообщать об инциденте информационной безопасности?",
        "А в какой срок?",
    ]
    for strategy in ("no-memory", "buffer"):
        print("=" * 96)
        print(f"стратегия памяти: {strategy}, переформулировка: "
              f"{'нет' if strategy == 'no-memory' else 'да'}")
        s = ChatSession(memory=strategy, contextualize=(strategy != "no-memory"))
        for q in dialog:
            a = s.ask(q)
            mark = "→ переписан: " + a.search_query if a.rewritten else ""
            print(f"\n[{a.turn}] {q}")
            if mark:
                print(f"    {mark}")
            print(f"    {a.answer.splitlines()[0][:150]}")
            print(f"    источник: {a.sources[0] if a.sources else '—'} | {a.total_s} c")
