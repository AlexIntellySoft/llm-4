"""Единая конфигурация пайплайна. Всё переопределяется через .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_RAW_OPTIONAL = ROOT / "data" / "raw_optional"
CHROMA_DIR = ROOT / "data" / "chroma"
OUTPUTS = ROOT / "outputs"

# --- Локальный сервер LM Studio (OpenAI-совместимый, без выхода в интернет) ---
LMS_BASE_URL = os.getenv("LMS_BASE_URL", "http://127.0.0.1:1234/v1")
LMS_API_KEY = os.getenv("LMS_API_KEY", "lm-studio")

# Модели генерации. Ключ — короткое имя, значение — id в LM Studio.
LLM_MODELS: dict[str, str] = {
    "gemma-4-e4b": "google/gemma-4-e4b",
    "qwen3.5-9b": "qwen/qwen3.5-9b",
    "qwen3-4b-nothink": "qwen/qwen3-4b-2507",
}
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "qwen3-4b-nothink")

# --- Эмбеддинги ---
# e5 требует префиксы "query: " / "passage: " — без них качество заметно падает.
EMBEDDERS: dict[str, dict] = {
    "e5-base": {
        "kind": "sentence-transformers",
        "model": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "dim": 768,
    },
    "nomic-lms": {
        "kind": "lmstudio",
        "model": "text-embedding-nomic-embed-text-v1.5",
        "query_prefix": "search_query: ",
        "passage_prefix": "search_document: ",
        "dim": 768,
    },
}
DEFAULT_EMBEDDER = os.getenv("DEFAULT_EMBEDDER", "e5-base")


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    # Префикс «документ → раздел» в каждом чанке: страхует от потери контекста
    # при нарезке таблиц и списков.
    contextual_header: bool = os.getenv("CONTEXTUAL_HEADER", "true").lower() == "true"


@dataclass(frozen=True)
class RagConfig:
    top_k: int = int(os.getenv("TOP_K", "4"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "3000"))
    # Порог косинусного расстояния: чанки дальше порога не идут в контекст.
    # None — фильтр выключен (поведение «наивного» RAG).
    max_distance: float | None = (
        float(os.environ["MAX_DISTANCE"]) if os.getenv("MAX_DISTANCE") else None
    )
    chunk: ChunkConfig = field(default_factory=ChunkConfig)


CHUNK = ChunkConfig()
RAG = RagConfig()


def collection_name(embedder: str, chunk: ChunkConfig | None = None) -> str:
    """Отдельная коллекция на каждую комбинацию эмбеддера и параметров нарезки."""
    c = chunk or CHUNK
    return f"polaris__{embedder.replace('.', '_')}__{c.chunk_size}_{c.chunk_overlap}"
