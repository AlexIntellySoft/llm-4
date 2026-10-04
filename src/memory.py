"""Память диалога: четыре стратегии с общим интерфейсом.

Соответствие тому, что разбиралось на занятии, и современному API LangChain:

| На занятии                      | Здесь                | Чем реализовано                    |
| ------------------------------- | -------------------- | ---------------------------------- |
| (без памяти)                    | `NoMemory`           | базовая линия для сравнения        |
| `ConversationBufferMemory`      | `BufferMemory`       | список `BaseMessage` целиком       |
| обрезка токенов                 | `TrimmedMemory`      | `langchain_core.trim_messages`     |
| `ConversationSummaryMemory`     | `SummaryMemory`      | сжатие старых ходов локальной LLM  |

Классы `ConversationBufferMemory` и `ConversationSummaryMemory` в LangChain 1.x
объявлены устаревшими и из пакета удалены, поэтому взяты их современные
эквиваленты: `trim_messages` для обрезки и собственная суммаризация поверх той
же локальной модели. Поведение то же, API — актуальный.

Счётчик токенов — эмпирический. Точного токенизатора для модели под рукой нет
(LM Studio не отдаёт его через API), поэтому коэффициент измерен на реальных
ответах: при изменении Top-K от 1 до 12 отношение «символов на токен» держалось
в диапазоне 2.71-2.83 (см. раздел про память в отчёте). Взято 2.6 - чуть ниже
измеренного, чтобы счётчик скорее завышал число токенов, чем занижал: ошибка
в большую сторону приводит к лишней обрезке, в меньшую - к выходу за окно.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages

# Символов на токен для русского текста, измерено (см. docstring выше).
CHARS_PER_TOKEN = 2.6
# Накладные расходы шаблона чата на одно сообщение.
PER_MESSAGE_OVERHEAD = 4


def count_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def count_message_tokens(messages: list[BaseMessage]) -> int:
    """Счётчик в том виде, в каком его ожидает trim_messages."""
    return sum(count_tokens(str(m.content)) + PER_MESSAGE_OVERHEAD for m in messages)


@dataclass
class Memory(ABC):
    """Общий интерфейс памяти диалога."""

    name: str = "base"
    history: list[BaseMessage] = field(default_factory=list)

    def add_turn(self, question: str, answer: str) -> None:
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))
        self._after_turn()

    def _after_turn(self) -> None:
        """Крючок для стратегий, которым нужно что-то сделать после хода."""

    @abstractmethod
    def messages(self) -> list[BaseMessage]:
        """История в том виде, в каком она уйдёт в промпт."""

    def prefix(self) -> str:
        """Текстовая врезка перед историей (используется суммаризацией)."""
        return ""

    def reset(self) -> None:
        self.history = []

    def stats(self) -> dict:
        used = self.messages()
        return {
            "strategy": self.name,
            "turns_total": len(self.history) // 2,
            "messages_in_prompt": len(used),
            "history_tokens": count_message_tokens(used) + count_tokens(self.prefix()),
        }


@dataclass
class NoMemory(Memory):
    """Базовая линия: каждый вопрос обрабатывается как самостоятельный."""

    name: str = "no-memory"

    def messages(self) -> list[BaseMessage]:
        return []


@dataclass
class BufferMemory(Memory):
    """Аналог ConversationBufferMemory: хранится весь диалог целиком.

    Простая и точная стратегия ровно до того момента, когда история перестаёт
    помещаться в контекстное окно вместе с найденными фрагментами.
    """

    name: str = "buffer"

    def messages(self) -> list[BaseMessage]:
        return list(self.history)


@dataclass
class TrimmedMemory(Memory):
    """Обрезка по бюджету токенов через langchain_core.trim_messages.

    Держим последние ходы, пока они укладываются в max_tokens. Старые
    выбрасываются без следа — это дёшево, но информация теряется безвозвратно.
    """

    name: str = "trim"
    max_tokens: int = 600

    def messages(self) -> list[BaseMessage]:
        if not self.history:
            return []
        return trim_messages(
            self.history,
            max_tokens=self.max_tokens,
            token_counter=count_message_tokens,
            strategy="last",
            start_on="human",
            include_system=False,
            allow_partial=False,
        )


@dataclass
class SummaryMemory(Memory):
    """Аналог ConversationSummaryBufferMemory.

    Последние `keep_last` ходов хранятся дословно, всё, что старше, сжимается
    локальной LLM в накопительное резюме. Компромисс: история не растёт
    бесконечно и при этом не исчезает целиком, но за каждое сжатие платим
    отдельным запросом к модели — это видно в замерах задержки.
    """

    name: str = "summary"
    keep_last: int = 2          # сколько последних ходов оставлять дословно
    summary: str = ""
    summary_calls: int = 0
    summary_seconds: float = 0.0
    model: str | None = None

    def _after_turn(self) -> None:
        keep = self.keep_last * 2
        if len(self.history) <= keep:
            return
        older, self.history = self.history[:-keep], self.history[-keep:]
        self._summarize(older)

    def _summarize(self, older: list[BaseMessage]) -> None:
        import time

        from src.llm import complete
        from src.prompts import build_summary_messages

        lines = []
        for m in older:
            role = "Работник" if isinstance(m, HumanMessage) else "Ассистент"
            lines.append(f"{role}: {m.content}")
        t0 = time.perf_counter()
        comp = complete(
            build_summary_messages(self.summary, "\n".join(lines)),
            model=self.model, temperature=0.1, max_tokens=400,
        )
        self.summary_seconds += time.perf_counter() - t0
        self.summary_calls += 1
        if comp.text:
            self.summary = comp.text.strip()

    def messages(self) -> list[BaseMessage]:
        return list(self.history)

    def prefix(self) -> str:
        return (f"Краткое содержание более ранней части диалога:\n{self.summary}"
                if self.summary else "")

    def reset(self) -> None:
        super().reset()
        self.summary = ""
        self.summary_calls = 0
        self.summary_seconds = 0.0

    def stats(self) -> dict:
        s = super().stats()
        s.update(summary_calls=self.summary_calls,
                 summary_seconds=round(self.summary_seconds, 2),
                 summary_chars=len(self.summary))
        return s


STRATEGIES: dict[str, type[Memory]] = {
    "no-memory": NoMemory,
    "buffer": BufferMemory,
    "trim": TrimmedMemory,
    "summary": SummaryMemory,
}


def make_memory(name: str, **kwargs) -> Memory:
    if name not in STRATEGIES:
        raise ValueError(f"неизвестная стратегия памяти: {name}; "
                         f"доступны: {', '.join(STRATEGIES)}")
    return STRATEGIES[name](**kwargs)
