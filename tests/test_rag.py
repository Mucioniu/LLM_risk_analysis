from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from credit_assistant.document_loader import DocumentChunk
from credit_assistant.rag import RagIndex
from credit_assistant.service import build_default_index


def chunk(source: str, text: str) -> DocumentChunk:
    return DocumentChunk(source=source, location="test", text=text)


class RagIndexTests(unittest.TestCase):
    def test_tfidf_remains_the_default_and_is_selectable(self) -> None:
        chunks = [chunk("a.txt", "alpha policy"), chunk("b.txt", "beta rule")]

        default_index = RagIndex(chunks)
        explicit_index = RagIndex(chunks, retriever="tfidf")

        self.assertEqual(default_index.retriever, "tfidf")
        self.assertEqual(explicit_index.search("alpha", top_k=1)[0].chunk.source, "a.txt")

    def test_bm25_uses_the_declared_okapi_formula(self) -> None:
        chunks = [
            chunk("short.txt", "alpha alpha"),
            chunk("long.txt", "alpha beta beta beta"),
        ]
        index = RagIndex(chunks, retriever="bm25", bm25_k1=1.2, bm25_b=0.75)

        results = index.search("alpha", top_k=2)
        scores = {result.chunk.source: result.score for result in results}

        inverse_document_frequency = math.log1p(0.5 / 2.5)
        average_length = 5.0  # unigram and bigram counts: 3 and 7
        short_normalizer = 1.2 * (1 - 0.75 + 0.75 * 3 / average_length)
        long_normalizer = 1.2 * (1 - 0.75 + 0.75 * 7 / average_length)
        expected_short = inverse_document_frequency * (2 * 2.2) / (2 + short_normalizer)
        expected_long = inverse_document_frequency * (1 * 2.2) / (1 + long_normalizer)

        self.assertEqual([result.chunk.source for result in results], ["short.txt", "long.txt"])
        self.assertAlmostEqual(scores["short.txt"], expected_short, places=12)
        self.assertAlmostEqual(scores["long.txt"], expected_long, places=12)

    def test_bm25_and_tfidf_share_the_same_analyzer_vocabulary(self) -> None:
        chunks = [
            chunk("a.txt", "Împrumut cu rată variabilă"),
            chunk("b.txt", "fixed interest credit"),
        ]
        bm25 = RagIndex(chunks, retriever="bm25")
        tfidf = RagIndex(chunks, retriever="tfidf")

        self.assertEqual(
            bm25.vectorizer.vocabulary_,
            tfidf.vectorizer.vocabulary_,
        )
        self.assertEqual(bm25.search("imprumut rata", 1)[0].chunk.source, "a.txt")

    def test_equal_scores_keep_original_chunk_order(self) -> None:
        chunks = [chunk("first.txt", "same term"), chunk("second.txt", "same term")]
        index = RagIndex(chunks, retriever="bm25")

        results = index.search("same term", top_k=2)

        self.assertEqual(
            [result.chunk.source for result in results],
            ["first.txt", "second.txt"],
        )

    def test_empty_oov_and_nonpositive_top_k_return_no_results(self) -> None:
        index = RagIndex([chunk("a.txt", "known vocabulary")], retriever="bm25")

        self.assertEqual(index.search("", top_k=5), [])
        self.assertEqual(index.search("unknown-token", top_k=5), [])
        self.assertEqual(index.search("known", top_k=0), [])
        self.assertEqual(index.search("known", top_k=-1), [])

    def test_invalid_configuration_is_rejected(self) -> None:
        chunks = [chunk("a.txt", "alpha")]

        with self.assertRaisesRegex(ValueError, "Unsupported retriever"):
            RagIndex(chunks, retriever="dense")
        with self.assertRaisesRegex(ValueError, "bm25_k1"):
            RagIndex(chunks, retriever="bm25", bm25_k1=0)
        with self.assertRaisesRegex(ValueError, "bm25_b"):
            RagIndex(chunks, retriever="bm25", bm25_b=1.1)
        with self.assertRaisesRegex(ValueError, "At least one"):
            RagIndex([], retriever="bm25")

    def test_from_paths_supports_both_backends(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "policy.txt"
            path.write_text("Rare policy term appears here.", encoding="utf-8")

            for retriever in ("tfidf", "bm25"):
                with self.subTest(retriever=retriever):
                    index = RagIndex.from_paths([path], retriever=retriever)
                    result = index.search("rare policy", top_k=1)
                    self.assertEqual(result[0].chunk.source, "policy.txt")

    def test_service_selects_bm25_and_parameters_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "policy.txt"
            path.write_text("Rare policy term appears here.", encoding="utf-8")
            settings = {
                "RAG_RETRIEVER": "bm25",
                "RAG_BM25_K1": "1.7",
                "RAG_BM25_B": "0.4",
            }
            with (
                patch.dict(os.environ, settings, clear=False),
                patch(
                    "credit_assistant.service.default_corpus_paths",
                    return_value=[path],
                ),
            ):
                index = build_default_index()

            self.assertEqual(index.retriever, "bm25")
            self.assertEqual(index.bm25_k1, 1.7)
            self.assertEqual(index.bm25_b, 0.4)

    def test_tfidf_ignores_irrelevant_malformed_bm25_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "policy.txt"
            path.write_text("Rare policy term appears here.", encoding="utf-8")
            settings = {
                "RAG_RETRIEVER": "tfidf",
                "RAG_BM25_K1": "not-a-number",
                "RAG_BM25_B": "also-invalid",
            }
            with (
                patch.dict(os.environ, settings, clear=False),
                patch(
                    "credit_assistant.service.default_corpus_paths",
                    return_value=[path],
                ),
            ):
                index = build_default_index()

            self.assertEqual(index.retriever, "tfidf")


if __name__ == "__main__":
    unittest.main()
