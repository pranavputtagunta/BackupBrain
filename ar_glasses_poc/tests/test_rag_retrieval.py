"""Offline unit tests for the semantic-retrieval RAG pipeline.

Run from ar_glasses_poc/:
    python -m unittest tests.test_rag_retrieval -v

No network or GEMINI_API_KEY needed: the embedding call is replaced with
a fixture dict keyed by exact candidate text, giving the fake vectors
real (hand-authored) semantic structure — a "sister" query is close to
Sarah's note and far from Ted's summary, so the tests prove relevance
filtering rather than just exercising code paths.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules import rag_pipeline

SISTER_QUERY = "How is your sister doing?"
FISHING_QUERY = "Have you been fishing lately?"
SARAH_NOTE = "Sarah is John's sister; they talk every week"
TED_SUMMARY = "Ted is the patient's friend; Retired firefighter"

# Hand-authored 3-dim "semantic space": axis 0 = family, axis 2 = fishing.
FAKE_VECTORS = {
    SISTER_QUERY: [1.0, 0.0, 0.0],
    FISHING_QUERY: [0.0, 0.0, 1.0],
    SARAH_NOTE: [0.95, 0.05, 0.0],
    TED_SUMMARY: [0.05, 0.0, 0.95],
}
UNRELATED = [0.0, 1.0, 0.0]  # orthogonal to both queries -> below threshold


def fake_embed_texts(texts, task_type):
    """Deterministic stand-in for rag_pipeline._embed_texts (L2-normalized)."""
    out = []
    for text in texts:
        vector = np.asarray(FAKE_VECTORS.get(text, UNRELATED), dtype=np.float64)
        out.append(vector / np.linalg.norm(vector))
    return out


JOHN = {
    "name": "John",
    "relationship": "son",
    "last_interaction": "Tuesday phone call",
    "facts": [
        {"text": "Lives in Austin", "important": True},
        {"text": "Works in tech", "important": False},
        "Has two kids",
    ],
    "recent_conversations": [
        {"text": "Talked about Thanksgiving plans", "timestamp": "2026-06-20T18:00:00"},
        {"text": "Mentioned his new job is going well", "timestamp": "2026-07-10T09:30:00"},
        {"text": "Complained about the Austin heat", "timestamp": "2026-05-01T10:00:00"},
        {"text": "Asked about old photo albums", "timestamp": "2026-04-01T10:00:00"},
    ],
    "related_people": [
        {"name": "sarah", "note": SARAH_NOTE},
        "ted",
    ],
}

TED = {
    "name": "Ted",
    "relationship": "friend",
    "last_interaction": "Coffee last month",
    "facts": [{"text": "Retired firefighter", "important": True}, "Enjoys fishing"],
    "recent_conversations": [],
    "related_people": [],
}


class RagRetrievalTest(unittest.TestCase):
    """Candidate building, ranking, offline fallback, and cache behavior."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.memories_dir = self._tmp.name
        for name, data in (("john", JOHN), ("ted", TED)):
            with open(os.path.join(self.memories_dir, f"{name}.json"), "w", encoding="utf-8") as f:
                json.dump(data, f)

    def tearDown(self):
        self._tmp.cleanup()

    def test_build_candidates_related_paths(self):
        always, candidates = rag_pipeline._build_candidates("john", JOHN, self.memories_dir)
        self.assertIn("Important fact: Lives in Austin", always)
        related = {c.source_name: c for c in candidates if c.kind == "related_person"}
        self.assertEqual(related["sarah"].text, SARAH_NOTE)  # note used verbatim
        self.assertEqual(related["ted"].text, TED_SUMMARY)  # auto-derived, one hop
        self.assertNotIn("Works in tech", always)  # non-important fact is ranked, not pinned

    def test_relevant_related_person_only(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test"}), mock.patch.object(
            rag_pipeline, "_embed_texts", side_effect=fake_embed_texts
        ):
            sister = rag_pipeline._select_context("john", JOHN, SISTER_QUERY, self.memories_dir)
            fishing = rag_pipeline._select_context("john", JOHN, FISHING_QUERY, self.memories_dir)
        self.assertIn("Sarah", sister)
        self.assertNotIn("Ted", sister)
        self.assertIn("Ted", fishing)
        self.assertNotIn("Sarah", fishing)
        # Unrelated candidates fall below the similarity floor entirely.
        self.assertNotIn("Thanksgiving", sister)
        # Always-include block survives regardless of query.
        for context in (sister, fishing):
            self.assertIn("Relationship: son", context)
            self.assertIn("Important fact: Lives in Austin", context)

    def test_offline_path_never_embeds(self):
        env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
            rag_pipeline, "_embed_texts"
        ) as embed:
            context = rag_pipeline._select_context("john", JOHN, SISTER_QUERY, self.memories_dir)
        embed.assert_not_called()
        self.assertIn("Relationship: son", context)
        # Most-recent-N conversations by timestamp (N=3 default): the
        # oldest of the four must be the one left out.
        self.assertIn("new job is going well", context)
        self.assertNotIn("photo albums", context)
        # Exactly one related person in offline mode.
        self.assertEqual(context.count("Also relevant:"), 1)

    def test_backward_compat_old_flat_schema(self):
        old = {
            "name": "Sarah",
            "relationship": "daughter",
            "last_interaction": "Sunday lunch",
            "facts": ["Is a nurse", "Loves gardening"],
            "recent_conversations": ["Talked about tomatoes"],
        }
        always, candidates = rag_pipeline._build_candidates("sarah", old, self.memories_dir)
        self.assertEqual(len(always), 3)  # no important flags in old schema
        kinds = sorted(c.kind for c in candidates)
        self.assertEqual(kinds, ["conversation", "fact", "fact"])

    def test_cache_hit_skips_embedding(self):
        texts = [SARAH_NOTE, TED_SUMMARY]
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            with mock.patch.object(rag_pipeline, "_embed_texts", side_effect=fake_embed_texts):
                first = rag_pipeline._get_cached_or_embed("john", texts, self.memories_dir)
            self.assertIsNotNone(first)
            self.assertTrue(
                os.path.isfile(rag_pipeline._embedding_cache_path("john", self.memories_dir))
            )
            with mock.patch.object(
                rag_pipeline, "_embed_texts", side_effect=AssertionError("must not embed")
            ):
                second = rag_pipeline._get_cached_or_embed("john", texts, self.memories_dir)
        self.assertIsNotNone(second)
        for text in texts:
            np.testing.assert_allclose(first[text], second[text])

    def test_stale_cache_discarded_on_dimension_change(self):
        path = rag_pipeline._embedding_cache_path("john", self.memories_dir)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": config.EMBEDDING_MODEL,
                    "dimensionality": config.EMBEDDING_DIMENSIONALITY + 1,
                    "vectors": {"deadbeef": [1.0, 2.0]},
                },
                f,
            )
        cache = rag_pipeline._load_embedding_cache("john", self.memories_dir)
        self.assertEqual(cache["vectors"], {})
        self.assertEqual(cache["dimensionality"], config.EMBEDDING_DIMENSIONALITY)


if __name__ == "__main__":
    unittest.main()
