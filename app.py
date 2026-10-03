"""Веб-интерфейс корпоративного ассистента (Gradio).

Запуск:  python app.py        → http://127.0.0.1:7860

Интерфейс сделан так, чтобы в нём было видно не только ответ, но и то, на чём
он построен: найденные фрагменты с их близостью, разбивка времени на поиск и
генерацию, расход токенов. Для отладки RAG это важнее, чем красивый чат:
без списка источников невозможно понять, ошибся поиск или модель.
"""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from src import config
from src.llm import available_models
from src.rag import answer
from src.vectorstore import collection_exists, list_collections

CSS = """
.sources-box {font-size: 13px;}
footer {display: none !important;}
"""

INTRO = """## Внутренний ассистент АО «Полярис Технологии»

Отвечает по внутренним регламентам компании: информационная безопасность, отпуска,
командировки, адаптация, инциденты, закупки. Работает полностью локально —
LLM и векторная база развёрнуты на этой машине, запросы никуда не уходят.
"""

EXAMPLES = [
    "Сколько календарных дней основного отпуска положено работнику?",
    "Какие суточные выплачиваются при командировке по России?",
    "Куда и в какой срок сообщать об инциденте информационной безопасности?",
    "Какие требования к длине пароля и как часто его менять?",
    "Кто должен согласовать закупку на 1 200 000 рублей?",
    "Как испечь шарлотку с яблоками?",
    "Сколько дней больничного оплачивает компания?",
]


def format_sources(hits) -> str:
    if not hits:
        return "_Релевантных фрагментов не найдено._"
    parts = ["### Найденные фрагменты\n"]
    for i, h in enumerate(hits, 1):
        secret = " 🔒" if "SECRET" in h.doc_class.upper() else ""
        parts.append(
            f"**{i}. {h.source}** · _{h.section}_{secret}\n\n"
            f"близость **{h.similarity:.3f}** · класс: {h.doc_class}\n\n"
            f"> {h.text[:420]}{'…' if len(h.text) > 420 else ''}\n"
        )
    return "\n".join(parts)


def format_metrics(r) -> str:
    c = r.completion
    rows = [
        f"| поиск | {r.retrieval_s * 1000:.0f} мс |",
        f"| генерация | {r.generation_s:.1f} с |",
        f"| **всего** | **{r.total_s:.1f} с** |",
        f"| модель | `{c.model if c else '—'}` |",
        f"| контекст | {c.prompt_tokens if c else 0} ток. |",
        f"| ответ | {c.completion_tokens if c else 0} ток. |",
    ]
    if c and c.reasoning_tokens:
        rows.append(f"| из них рассуждение | {c.reasoning_tokens} ток. |")
    if c and c.tokens_per_second:
        rows.append(f"| скорость | {c.tokens_per_second} ток/с |")
    rows.append(f"| отказ от ответа | {'да' if r.refused else 'нет'} |")
    return "| метрика | значение |\n| --- | --- |\n" + "\n".join(rows)


def respond(message: str, history: list, model: str, top_k: int, variant: str,
            temperature: float, use_threshold: bool, show_reasoning: bool):
    message = (message or "").strip()
    if not message:
        return history, "", "", ""

    history = list(history) + [{"role": "user", "content": message}]
    r = answer(message, top_k=int(top_k), model=model, variant=variant,
               temperature=float(temperature),
               max_distance=(0.22 if use_threshold else None))

    text = r.answer or "_(модель вернула пустой ответ — вероятно, весь бюджет токенов ушёл на рассуждение)_"
    history = history + [{"role": "assistant", "content": text}]

    reasoning = ""
    if show_reasoning and r.completion and r.completion.reasoning:
        reasoning = ("### Рассуждение модели\n\n```\n"
                     + r.completion.reasoning[:3000] + "\n```")
    return history, format_sources(r.hits), format_metrics(r), reasoning


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Полярис · внутренний ассистент", css=CSS,
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown(INTRO)

        with gr.Row():
            with gr.Column(scale=3):
                # Gradio 6: формат сообщений {"role", "content"} единственный,
                # отдельный параметр type больше не нужен.
                chat = gr.Chatbot(height=440, show_label=False,
                                  placeholder="История диалога появится здесь")
                msg = gr.Textbox(placeholder="Задайте вопрос по внутренним регламентам…",
                                 show_label=False, lines=2, autofocus=True)
                with gr.Row():
                    send = gr.Button("Спросить", variant="primary")
                    clear = gr.Button("Очистить")
                gr.Examples(examples=EXAMPLES, inputs=msg, label="Примеры, в том числе вне базы")
                metrics = gr.Markdown(label="Метрики")

            with gr.Column(scale=2):
                with gr.Accordion("Параметры", open=True):
                    model = gr.Dropdown(choices=list(config.LLM_MODELS),
                                        value=config.DEFAULT_LLM, label="Локальная LLM")
                    top_k = gr.Slider(1, 12, value=config.RAG.top_k, step=1,
                                      label="Top-K фрагментов")
                    variant = gr.Radio(["strict", "naive"], value="strict",
                                       label="Системный промпт",
                                       info="naive — промпт без защиты от инъекций "
                                            "и без правила о приоритете контекста")
                    temperature = gr.Slider(0.0, 1.0, value=config.RAG.temperature,
                                            step=0.05, label="Temperature")
                    use_threshold = gr.Checkbox(
                        value=False, label="Отсекать фрагменты по порогу близости (0.78)",
                        info="показывает, почему порог не спасает от вопросов вне базы")
                    show_reasoning = gr.Checkbox(value=False,
                                                 label="Показывать рассуждение модели")
                sources = gr.Markdown("_Задайте вопрос, здесь появятся источники._",
                                      elem_classes="sources-box")
                reasoning = gr.Markdown("")

        inputs = [msg, chat, model, top_k, variant, temperature, use_threshold, show_reasoning]
        outputs = [chat, sources, metrics, reasoning]
        send.click(respond, inputs, outputs).then(lambda: "", None, msg)
        msg.submit(respond, inputs, outputs).then(lambda: "", None, msg)
        clear.click(lambda: ([], "_Задайте вопрос, здесь появятся источники._", "", ""),
                    None, outputs)
    return demo


def preflight() -> None:
    if not collection_exists():
        raise SystemExit(
            f"Коллекция для эмбеддера {config.DEFAULT_EMBEDDER} не найдена.\n"
            f"Сначала проиндексируйте корпус:  python -m src.ingest"
        )
    models = available_models()
    if models and models[0].startswith("<"):
        raise SystemExit(f"{models[0]}\nЗапустите сервер:  lms server start")
    print("коллекции:", ", ".join(f"{n} ({c})" for n, c in list_collections()))
    print("модели LM Studio:", ", ".join(models))


if __name__ == "__main__":
    preflight()
    # server_name жёстко 127.0.0.1: интерфейс не должен слушать внешний интерфейс.
    build_ui().launch(server_name="127.0.0.1", server_port=7860,
                      inbrowser=False, quiet=False)
