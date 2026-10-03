"""Локальная векторная база на ChromaDB (персистентная, в data/chroma).

Коллекция создаётся отдельно под каждую пару «эмбеддер + параметры нарезки»,
чтобы эксперименты не перетирали друг друга и ноутбук можно было
перезапускать без переиндексации.

В эмбеддинг уходит текст с «хлебной крошкой» (chunk.text), а в базе как
документ лежит исходный фрагмент (chunk.raw_text) — его и показываем
пользователю, и отдаём модели в контекст вместе с явной ссылкой на источник.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from src import config
from src.chunking import Chunk
from src.embeddings import Embedder, get_embedder


@dataclass
class Hit:
    text: str
    source: str
    doc_title: str
    section: str
    doc_class: str
    chunk_id: str
    distance: float

    @property
    def similarity(self) -> float:
        """Chroma отдаёт косинусное расстояние; удобнее смотреть на близость."""
        return 1.0 - self.distance

    def citation(self) -> str:
        return f"{self.source} · {self.section}"


def _client() -> chromadb.ClientAPI:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def build_index(chunks: list[Chunk], embedder: Embedder,
                chunk_cfg=None, recreate: bool = True) -> dict:
    """Векторизует чанки и складывает их в коллекцию. Возвращает статистику."""
    name = config.collection_name(embedder.name, chunk_cfg)
    client = _client()
    if recreate:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "embedder": embedder.name},
    )

    t0 = time.perf_counter()
    vectors = embedder.encode_passages([c.text for c in chunks])
    embed_s = time.perf_counter() - t0

    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=[c.raw_text for c in chunks],
        metadatas=[c.metadata() for c in chunks],
    )
    return {
        "collection": name,
        "chunks": len(chunks),
        "embedder": embedder.name,
        "dim": len(vectors[0]) if vectors else 0,
        "embed_seconds": round(embed_s, 2),
        "chunks_per_second": round(len(chunks) / embed_s, 1) if embed_s else 0.0,
    }


def get_collection(embedder_name: str | None = None, chunk_cfg=None):
    embedder_name = embedder_name or config.DEFAULT_EMBEDDER
    return _client().get_collection(config.collection_name(embedder_name, chunk_cfg))


def search(query: str, top_k: int = 4, embedder_name: str | None = None,
           chunk_cfg=None, max_distance: float | None = None) -> list[Hit]:
    """Семантический поиск: вектор запроса → top-K ближайших чанков."""
    embedder = get_embedder(embedder_name)
    collection = get_collection(embedder.name, chunk_cfg)
    res = collection.query(
        query_embeddings=[embedder.encode_query(query)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[Hit] = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                               res["distances"][0]):
        if max_distance is not None and dist > max_distance:
            continue
        hits.append(Hit(
            text=doc,
            source=str(meta.get("source", "")),
            doc_title=str(meta.get("doc_title", "")),
            section=str(meta.get("section", "")),
            doc_class=str(meta.get("doc_class", "")),
            chunk_id=str(meta.get("chunk_id", "")),
            distance=float(dist),
        ))
    return hits


def collection_exists(embedder_name: str | None = None, chunk_cfg=None) -> bool:
    try:
        get_collection(embedder_name, chunk_cfg)
        return True
    except Exception:
        return False


def list_collections() -> list[tuple[str, int]]:
    return [(c.name, c.count()) for c in _client().list_collections()]
