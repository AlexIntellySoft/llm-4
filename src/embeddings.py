"""Эмбеддинги: два взаимозаменяемых бэкенда.

* sentence-transformers — multilingual-e5-base, работает локально на MPS (Apple
  Silicon). Модель скачивается с HuggingFace один раз и далее берётся из кэша:
  после первого запуска пайплайн работает без интернета
  (проверяется переменной HF_HUB_OFFLINE=1).
* LM Studio — эмбеддинги через тот же локальный OpenAI-совместимый сервер,
  что и генерация. Нужен для сравнения: модель nomic-embed-text в основном
  англоязычная, и на русском корпусе это видно.

У e5 и nomic обязательные префиксы для запроса и документа — без них
качество поиска падает, поэтому они заданы в конфиге, а не зашиты здесь.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from src import config


class Embedder(ABC):
    name: str
    dim: int
    query_prefix: str
    passage_prefix: str

    @abstractmethod
    def _encode(self, texts: list[str]) -> list[list[float]]: ...

    def encode_passages(self, texts: Iterable[str], batch_size: int = 32) -> list[list[float]]:
        items = [self.passage_prefix + t for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(items), batch_size):
            out.extend(self._encode(items[i:i + batch_size]))
        return out

    def encode_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, spec: dict, name: str) -> None:
        from sentence_transformers import SentenceTransformer
        import torch

        self.name = name
        self.dim = spec["dim"]
        self.query_prefix = spec["query_prefix"]
        self.passage_prefix = spec["passage_prefix"]
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(spec["model"], device=device)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)
        return vecs.tolist()


class LmStudioEmbedder(Embedder):
    def __init__(self, spec: dict, name: str) -> None:
        from openai import OpenAI

        self.name = name
        self.dim = spec["dim"]
        self.query_prefix = spec["query_prefix"]
        self.passage_prefix = spec["passage_prefix"]
        self.model = spec["model"]
        self.client = OpenAI(base_url=config.LMS_BASE_URL, api_key=config.LMS_API_KEY)
        self.device = "lm-studio"

    def _encode(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        return [_l2(v) for v in vecs]


def _l2(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


_CACHE: dict[str, Embedder] = {}


def get_embedder(name: str | None = None) -> Embedder:
    name = name or config.DEFAULT_EMBEDDER
    if name in _CACHE:
        return _CACHE[name]
    spec = config.EMBEDDERS[name]
    emb: Embedder = (SentenceTransformerEmbedder(spec, name)
                     if spec["kind"] == "sentence-transformers"
                     else LmStudioEmbedder(spec, name))
    _CACHE[name] = emb
    return emb


if __name__ == "__main__":
    import time

    for name in config.EMBEDDERS:
        t0 = time.perf_counter()
        emb = get_embedder(name)
        load = time.perf_counter() - t0
        t0 = time.perf_counter()
        vs = emb.encode_passages(["Основной отпуск составляет 31 календарный день."] * 8)
        enc = time.perf_counter() - t0
        q = emb.encode_query("сколько дней отпуска")
        sim = sum(a * b for a, b in zip(vs[0], q))
        print(f"{name:12s} dim={len(vs[0]):4d} device={emb.device:10s} "
              f"загрузка {load:5.1f} c, 8 чанков {enc * 1000:6.0f} мс, "
              f"cos(отпуск)={sim:.3f}")
