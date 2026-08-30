from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .document_loader import DocumentChunk, load_documents


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


RetrieverName = Literal["bm25", "tfidf"]
SUPPORTED_RETRIEVERS: tuple[RetrieverName, ...] = ("bm25", "tfidf")
DEFAULT_RETRIEVER: RetrieverName = "tfidf"
DEFAULT_BM25_K1 = 1.2
DEFAULT_BM25_B = 0.75


class RagIndex:
    """Sparse lexical index with selectable TF-IDF or Okapi BM25 ranking.

    Both backends intentionally share the same lowercase, accent-normalized
    unigram-and-bigram feature space. This keeps paired comparisons focused on
    weighting and length normalization rather than on tokenization changes.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        *,
        retriever: str = DEFAULT_RETRIEVER,
        bm25_k1: float = DEFAULT_BM25_K1,
        bm25_b: float = DEFAULT_BM25_B,
    ) -> None:
        if not chunks:
            raise ValueError("At least one document chunk is required.")

        normalized_retriever = retriever.strip().lower()
        if normalized_retriever not in SUPPORTED_RETRIEVERS:
            supported = ", ".join(SUPPORTED_RETRIEVERS)
            raise ValueError(
                f"Unsupported retriever {retriever!r}; expected one of: {supported}."
            )
        if not np.isfinite(bm25_k1) or bm25_k1 <= 0:
            raise ValueError("bm25_k1 must be a finite positive number.")
        if not np.isfinite(bm25_b) or not 0 <= bm25_b <= 1:
            raise ValueError("bm25_b must be finite and between 0 and 1.")

        self.chunks = chunks
        self.retriever: RetrieverName = normalized_retriever  # type: ignore[assignment]
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)

        vectorizer_options = {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_features": 20_000,
        }
        texts = [chunk.text for chunk in chunks]

        if self.retriever == "tfidf":
            self.vectorizer = TfidfVectorizer(**vectorizer_options)
            self.matrix = self.vectorizer.fit_transform(texts).tocsr()
            self.scoring_matrix = self.matrix
            analyzer = self.vectorizer.build_analyzer()
            self.document_lengths = np.asarray(
                [len(analyzer(text)) for text in texts], dtype=float
            )
            self.average_document_length = float(self.document_lengths.mean())
            self.inverse_document_frequency = np.asarray(
                self.vectorizer.idf_, dtype=float
            )
            return

        self.vectorizer = CountVectorizer(**vectorizer_options)
        self.matrix = self.vectorizer.fit_transform(texts).tocsr().astype(float)
        self.document_lengths = np.asarray(self.matrix.sum(axis=1)).ravel()
        self.average_document_length = float(self.document_lengths.mean())
        if self.average_document_length <= 0:
            raise ValueError("BM25 cannot index a corpus with no analyzer tokens.")

        document_frequency = np.asarray((self.matrix > 0).sum(axis=0)).ravel()
        document_count = self.matrix.shape[0]
        self.inverse_document_frequency = np.log1p(
            (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )

        length_normalizer = self.bm25_k1 * (
            1.0
            - self.bm25_b
            + self.bm25_b
            * self.document_lengths
            / self.average_document_length
        )
        weighted = self.matrix.copy()
        row_indices = np.repeat(
            np.arange(weighted.shape[0]), np.diff(weighted.indptr)
        )
        weighted.data = (
            weighted.data * (self.bm25_k1 + 1.0)
            / (weighted.data + length_normalizer[row_indices])
        )
        self.scoring_matrix = weighted.multiply(
            self.inverse_document_frequency
        ).tocsr()

    @classmethod
    def from_paths(
        cls,
        paths: list[Path],
        *,
        retriever: str = DEFAULT_RETRIEVER,
        bm25_k1: float = DEFAULT_BM25_K1,
        bm25_b: float = DEFAULT_BM25_B,
    ) -> "RagIndex":
        return cls(
            load_documents(paths),
            retriever=retriever,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k <= 0 or not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        scores = (self.scoring_matrix @ query_vector.T).toarray().ravel()
        if not np.any(scores):
            return []

        # Stable descending order makes exact-score ties reproducible and keeps
        # the original corpus order as the secondary key.
        top_indices = np.argsort(-scores, kind="stable")[:top_k]
        return [
            RetrievedChunk(chunk=self.chunks[index], score=float(scores[index]))
            for index in top_indices
            if scores[index] > 0
        ]


def format_sources(results: list[RetrievedChunk], *, max_chars: int = 800) -> str:
    if not results:
        return "No relevant excerpts were found in the corpus."

    sections: list[str] = []
    for idx, result in enumerate(results, start=1):
        text = result.chunk.text.replace("\n", " ")
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        sections.append(
            f"[{idx}] {result.chunk.source}, {result.chunk.location}, score {result.score:.3f}\n{text}"
        )
    return "\n\n".join(sections)
