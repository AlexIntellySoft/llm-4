"""Клиент локальной LLM через LM Studio (OpenAI-совместимый API на localhost).

Ни один запрос не уходит за пределы машины: base_url всегда 127.0.0.1.

Отдельная тонкость — «думающие» модели. gemma-4-e4b и qwen3.5-9b возвращают
рассуждение в поле reasoning_content, а ответ — в content. Рассуждение
оплачивается токенами и временем, поэтому считаем его отдельно: в отчёте
видно, во что обходится reasoning внутри RAG-цикла.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from openai import OpenAI

from src import config


@dataclass
class Completion:
    text: str
    reasoning: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str = ""
    error: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_second(self) -> float:
        return round(self.completion_tokens / self.latency_s, 1) if self.latency_s else 0.0


_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.LMS_BASE_URL, api_key=config.LMS_API_KEY,
                         timeout=600.0, max_retries=1)
    return _client


def available_models() -> list[str]:
    try:
        return [m.id for m in client().models.list().data]
    except Exception as exc:  # сервер не запущен
        return [f"<LM Studio недоступен: {exc}>"]


def resolve(model: str) -> str:
    """Короткое имя из конфига → id модели в LM Studio."""
    return config.LLM_MODELS.get(model, model)


def complete(messages: list[dict], model: str | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             enable_thinking: bool | None = None) -> Completion:
    model_id = resolve(model or config.DEFAULT_LLM)
    kwargs: dict = {
        "model": model_id,
        "messages": messages,
        "temperature": config.RAG.temperature if temperature is None else temperature,
        "max_tokens": config.RAG.max_tokens if max_tokens is None else max_tokens,
    }
    # Qwen3 умеет отключать рассуждение через шаблон чата; LM Studio
    # пробрасывает chat_template_kwargs в движок.
    if enable_thinking is not None:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

    t0 = time.perf_counter()
    try:
        resp = client().chat.completions.create(**kwargs)
    except Exception as exc:
        return Completion(text="", model=model_id, latency_s=time.perf_counter() - t0,
                          error=str(exc))
    latency = time.perf_counter() - t0

    choice = resp.choices[0]
    msg = choice.message
    reasoning = getattr(msg, "reasoning_content", None) or ""
    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
    if not reasoning_tokens and reasoning:
        # LM Studio не всегда заполняет completion_tokens_details — оцениваем
        # по длине текста рассуждения (≈3 символа на токен для русского).
        reasoning_tokens = max(1, len(reasoning) // 3)

    return Completion(
        text=(msg.content or "").strip(),
        reasoning=reasoning.strip(),
        model=model_id,
        prompt_tokens=int(usage.prompt_tokens or 0),
        completion_tokens=int(usage.completion_tokens or 0),
        reasoning_tokens=reasoning_tokens,
        latency_s=round(latency, 2),
        finish_reason=choice.finish_reason or "",
    )


if __name__ == "__main__":
    print("модели на локальном сервере:")
    for m in available_models():
        print("  ", m)
    c = complete([{"role": "user", "content": "Скажи ровно одно слово: готово."}],
                 model="qwen3-4b-nothink", max_tokens=50)
    print(f"\n{c.model}: {c.text!r}  ({c.latency_s} c, "
          f"{c.completion_tokens} ток., {c.tokens_per_second} ток/с)")
