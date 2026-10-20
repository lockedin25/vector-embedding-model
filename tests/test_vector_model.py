"""
Automated test suite for Vector Embedding Model (Qwen3-Embedding-4B).
Tests expression parsing, vector arithmetic, semantic transformations,
vocabulary nearest-neighbor search, and Flask API endpoints.
"""

from __future__ import annotations

import unittest

import numpy as np

from main import (
    apply_transformation,
    compute_similarity,
    evaluate_vector_expression,
    find_similar_words,
    get_embedding,
    parse_vector_expression,
)
from server import app


class TestVectorEmbeddingModel(unittest.TestCase):
    def test_parse_vector_expression(self) -> None:
        self.assertEqual(
            parse_vector_expression("king - man + woman"),
            [("+", "king"), ("-", "man"), ("+", "woman")],
        )
        self.assertEqual(
            parse_vector_expression("uncle + (woman - man)"),
            [("+", "uncle"), ("+", "woman"), ("-", "man")],
        )

    def test_get_embedding_unit_normalized(self) -> None:
        vec = get_embedding("ocean")
        self.assertEqual(len(vec), 2560)
        norm = float(np.linalg.norm(vec))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_evaluate_vector_expression_king_queen(self) -> None:
        res = evaluate_vector_expression("king - man + woman", top_k=5)
        self.assertEqual(res["closest_word"], "queen")
        self.assertGreater(res["closest_similarity"], 0.75)

    def test_apply_transformation_uncle_aunt(self) -> None:
        res = apply_transformation("uncle", "man", "woman", top_k=5)
        self.assertIn(res["closest_word"], {"aunt", "auntie"})
        self.assertGreater(res["closest_similarity"], 0.75)

    def test_compute_similarity_and_neighbors(self) -> None:
        sim = compute_similarity("ocean", "sea")
        self.assertGreater(sim["similarity"], 0.65)
        neighbors = find_similar_words("ocean", top_k=5)
        self.assertEqual(len(neighbors["nearest_words"]), 5)

    def test_flask_api_endpoints(self) -> None:
        client = app.test_client()

        # Reset scene
        res = client.post("/api/reset")
        self.assertEqual(res.status_code, 200)

        # Single word embed
        res = client.post("/api/embed", json={"word": "ocean"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data["points"]), 1)
        self.assertIn("word_info", data)

        # State check
        res = client.get("/api/state")
        self.assertEqual(res.status_code, 200)
        state_data = res.get_json()
        self.assertEqual(len(state_data["points"]), 1)
        self.assertGreaterEqual(state_data["vocab_size"], 2500)

        # Transformation endpoint
        res = client.post(
            "/api/transform",
            json={"base": "uncle", "source": "man", "target": "woman", "plot_components": True},
        )
        self.assertEqual(res.status_code, 200)
        trans_data = res.get_json()
        self.assertIn(trans_data["transformation"]["closest_word"], {"aunt", "auntie"})
        self.assertIsNotNone(trans_data.get("explained_variance"))
        self.assertEqual(len(trans_data["explained_variance"]), 3)
        self.assertIn("similarity_matrix", trans_data)
        self.assertGreaterEqual(len(trans_data["similarity_matrix"]["words"]), 6)

        # Similarity endpoint
        res = client.post("/api/similarity", json={"word_a": "king", "word_b": "queen"})
        self.assertEqual(res.status_code, 200)
        sim_data = res.get_json()
        self.assertGreater(sim_data["similarity"]["similarity"], 0.7)

        # Remove single word
        res = client.post("/api/remove", json={"word": "ocean"})
        self.assertEqual(res.status_code, 200)
        rem_data = res.get_json()
        words_after = [p["word"] for p in rem_data["points"]]
        self.assertNotIn("ocean", words_after)


if __name__ == "__main__":
    unittest.main()
