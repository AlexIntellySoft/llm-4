"""Веб-интерфейс корпоративного ассистента (Gradio).

Запуск:  python app.py        → http://127.0.0.1:7860

Интерфейс показывает не только ответ, но и то, на чём он построен: найденные
фрагменты с их близостью, переформулированный поисковый запрос, разбивку времени
и размер истории в токенах. Для отладки RAG это важнее красивого чата: без этих
чисел непонятно, ошибся поиск, память или модель.

Диалог ведётся через `ChatSession`, состояние которой живёт в `gr.State` —
у каждого пользователя своя история.
"""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from src import config
from src.chat import ChatSession
from src.llm import available_models
from src.memory import STRATEGIES, SummaryMemory, TrimmedMemory, make_memory
from src.vectorstore import collection_exists, list_collections

CSS = """
.sources-box {font-size: 13px;}
footer {display: none !important;}
"""

INTRO = """## Внутренний ассистент АО «Полярис Технологии»

Отвечает по внутренним регламентам компании: информационная безопасность, отпуска,
командировки, адаптация, инциденты, закупки. Работает полностью локально —
LLM и векторная база развёрнуты на этой машине, запросы никуда не уходят.

Диалог ведётся с памятью: уточняющие вопросы вроде «а за рубежом сколько?»
понимаются в контексте предыдущих ходов.
"""

EXAMPLES = [
    "Какие суточные выплачиваются при командировке по России?",
    "А за рубежом сколько?",
    "Куда сообщать об инциденте информационной безопасности?",
    "А в какой срок?",
    "Сколько календарных дней основного отпуска положено работнику?",
    "Как испечь шарлотку с яблоками?",
    "Сколько дней больничного оплачивает компания?",
]

MEMORY_LABELS = {
    "no-memory": "no-memory — без памяти",
    "buffer": "buffer — весь диалог",
    "trim": "trim — обрезка по токенам",
    "summary": "summary — сжатие старых ходов",
}


def build_memory(name: str):
    if name == "trim":
        return TrimmedMemory(max_tokens=600)
    if name == "summary":
        return SummaryMemory(keep_last=2)
    return make_memory(name)


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


def format_metrics(a) -> str:
    c = a.completion
    m = a.memory_stats
    rows = []
    if a.contextualize_s:
        rows.append(f"| переформулировка запроса | {a.contextualize_s:.1f} с |")
    rows += [
        f"| поиск | {a.retrieval_s * 1000:.0f} мс |",
        f"| генерация | {a.generation_s:.1f} с |",
        f"| **всего** | **{a.total_s:.1f} с** |",
        f"| модель | `{c.model if c else '—'}` |",
        f"| контекст промпта | {c.prompt_tokens if c else 0} ток. |",
        f"| из них история | {m.get('history_tokens', 0)} ток. |",
        f"| ходов в памяти | {m.get('messages_in_prompt', 0) // 2} из {m.get('turns_total', 0)} |",
        f"| ответ | {c.completion_tokens if c else 0} ток. |",
    ]
    if m.get("summary_calls"):
        rows.append(f"| сжатий истории | {m['summary_calls']} "
                    f"({m.get('summary_seconds', 0)} с) |")
    if c and c.reasoning_tokens:
        rows.append(f"| из них рассуждение | {c.reasoning_tokens} ток. |")
    rows.append(f"| отказ от ответа | {'да' if a.refused else 'нет'} |")
    return "| метрика | значение |\n| --- | --- |\n" + "\n".join(rows)


def respond(message: str, history: list, session, model: str, top_k: int,
            variant: str, temperature: float, memory_name: str,
            contextualize: bool, use_threshold: bool, show_reasoning: bool):
    message = (message or "").strip()
    if not message:
        return history, session, "", "", ""

    # Смена стратегии памяти начинает диалог заново: сравнивать стратегии
    # на наполовину общей истории бессмысленно.
    if session is None or session.memory.name != memory_name:
        session = ChatSession(memory=build_memory(memory_name))

    session.model = model
    session.top_k = int(top_k)
    session.variant = variant
    session.temperature = float(temperature)
    session.contextualize = bool(contextualize)
    session.max_distance = 0.22 if use_threshold else None

    a = session.ask(message)

    text = a.answer or ("_(модель вернула пустой ответ — вероятно, весь бюджет "
                        "токенов ушёл на рассуждение)_")
    if a.rewritten:
        text = f"{text}\n\n*Запрос в поиск переписан: «{a.search_query}»*"

    history = list(history) + [{"role": "user", "content": message},
                               {"role": "assistant", "content": text}]

    reasoning = ""
    if show_reasoning and a.completion and a.completion.reasoning:
        reasoning = ("### Рассуждение модели\n\n```\n"
                     + a.completion.reasoning[:3000] + "\n```")
    return history, session, format_sources(a.hits), format_metrics(a), reasoning


def clear_all(session):
    if session is not None:
        session.reset()
    return [], session, "_Задайте вопрос, здесь появятся источники._", "", ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Полярис · внутренний ассистент", css=CSS,
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown(INTRO)
        session = gr.State(None)

        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(height=440, show_label=False,
                                  placeholder="История диалога появится здесь")
                msg = gr.Textbox(placeholder="Задайте вопрос по внутренним регламентам…",
                                 show_label=False, lines=2, autofocus=True)
                with gr.Row():
                    send = gr.Button("Спросить", variant="primary")
                    clear = gr.Button("Очистить диалог")
                gr.Examples(examples=EXAMPLES, inputs=msg,
                            label="Примеры: подряд 1–2 и 3–4 показывают работу памяти")
                metrics = gr.Markdown(label="Метрики")

            with gr.Column(scale=2):
                with gr.Accordion("Память диалога", open=True):
                    memory_name = gr.Radio(
                        choices=[(MEMORY_LABELS[k], k) for k in STRATEGIES],
                        value="buffer", label="Стратегия",
                        info="смена стратегии начинает диалог заново")
                    contextualize = gr.Checkbox(
                        value=True, label="Переписывать уточняющие вопросы перед поиском",
                        info="«А за рубежом?» → «Какие суточные за рубежом?»; "
                             "без этого поиск уходит не в тот документ")
                with gr.Accordion("Параметры", open=False):
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

        inputs = [msg, chat, session, model, top_k, variant, temperature,
                  memory_name, contextualize, use_threshold, show_reasoning]
        outputs = [chat, session, sources, metrics, reasoning]
        send.click(respond, inputs, outputs).then(lambda: "", None, msg)
        msg.submit(respond, inputs, outputs).then(lambda: "", None, msg)
        clear.click(clear_all, session, outputs)
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
