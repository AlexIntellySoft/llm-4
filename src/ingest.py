"""Индексация корпуса: документы → чанки → векторы → ChromaDB.

Примеры:
    python -m src.ingest                      # основной корпус, эмбеддер из .env
    python -m src.ingest --embedder nomic-lms
    python -m src.ingest --chunk-size 1200 --overlap 200
    python -m src.ingest --with-injection     # добавить документ-зонд (эксперимент 4)
"""
from __future__ import annotations

import argparse
import dataclasses

from src import config
from src.chunking import chunk_corpus, stats
from src.embeddings import get_embedder
from src.loaders import load_corpus
from src.vectorstore import build_index, list_collections


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default=config.DEFAULT_EMBEDDER,
                    choices=list(config.EMBEDDERS))
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK.chunk_size)
    ap.add_argument("--overlap", type=int, default=config.CHUNK.chunk_overlap)
    ap.add_argument("--no-contextual-header", action="store_true",
                    help="не подставлять крошку «документ / раздел» в чанк")
    ap.add_argument("--with-injection", action="store_true",
                    help="включить в индекс документ с prompt injection")
    args = ap.parse_args()

    chunk_cfg = dataclasses.replace(
        config.CHUNK, chunk_size=args.chunk_size, chunk_overlap=args.overlap,
        contextual_header=not args.no_contextual_header,
    )

    dirs = [config.DATA_RAW]
    if args.with_injection:
        dirs.append(config.DATA_RAW_OPTIONAL)

    docs = load_corpus(*dirs)
    print(f"документов: {len(docs)}")
    chunks = chunk_corpus(docs, chunk_cfg)
    s = stats(chunks)
    print(f"чанков: {s['chunks']} (размер {chunk_cfg.chunk_size}, "
          f"перекрытие {chunk_cfg.chunk_overlap}, "
          f"крошка={'да' if chunk_cfg.contextual_header else 'нет'}), "
          f"средняя длина {s['avg_len']} симв.")

    embedder = get_embedder(args.embedder)
    info = build_index(chunks, embedder, chunk_cfg)
    print(f"коллекция {info['collection']}: {info['chunks']} векторов "
          f"dim={info['dim']}, {info['embed_seconds']} c "
          f"({info['chunks_per_second']} чанк/с)")

    print("\nвсе коллекции в базе:")
    for name, count in list_collections():
        print(f"  {name:48s} {count:4d}")


if __name__ == "__main__":
    main()
