from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .document_loader import DocumentChunk, load_documents


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


class RagIndex:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_features=20_000,
        )
        self.matrix = self.vectorizer.fit_transform(chunk.text for chunk in chunks)

    @classmethod
    def from_paths(cls, paths: list[Path]) -> "RagIndex":
        return cls(load_documents(paths))

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        if not np.any(scores):
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(chunk=self.chunks[index], score=float(scores[index]))
            for index in top_indices
            if scores[index] > 0
        ]


def format_sources(results: list[RetrievedChunk], *, max_chars: int = 800) -> str:
    if not results:
        return "Nu am gasit fragmente relevante in corpus."

    sections: list[str] = []
    for idx, result in enumerate(results, start=1):
        text = result.chunk.text.replace("\n", " ")
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        sections.append(
            f"[{idx}] {result.chunk.source}, {result.chunk.location}, scor {result.score:.3f}\n{text}"
        )
    return "\n\n".join(sections)

