"""
Web interface for the embedding model — plots words, vector expressions,
and semantic transformation analogies in an interactive 3D coordinate space.

Endpoints:
- GET  /api/state: Returns active 3D scene points, links, and metadata
- POST /api/embed: Embeds a single word or an algebraic expression ("king - man + woman")
- POST /api/transform: Evaluates Base + (Target - Source), finds nearest word, and plots all nodes
- POST /api/similarity: Computes 2560-dim cosine similarity between two words and links them in 3D
- POST /api/remove: Removes a single vector and its associated links from the 3D scene
- POST /api/reset: Clears the 3D scene
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from sklearn.decomposition import PCA

from main import (
    apply_transformation,
    compute_similarity,
    evaluate_vector_expression,
    find_similar_words,
    get_embedding,
    load_model,
)
from vocab_manager import vocab_manager

app = Flask(__name__, static_folder="static", static_url_path="")

RADIUS = 5.0

_embeddings: dict[str, np.ndarray] = {}
_order: list[str] = []
_meta: dict[str, dict[str, Any]] = {}
_links: list[dict[str, Any]] = []

# Deterministic fallback projection for when fewer than 3 words are plotted
_fallback_matrix = np.random.default_rng(42).normal(size=(2560, 3))

_STANDARD_PALETTE = [
    "#58a6ff",  # Technical blue
    "#3fb950",  # Signal green
    "#d29922",  # Ochre / amber
    "#f78166",  # Terracotta
    "#79c0ff",  # Ice blue
    "#56d364",  # Mint
    "#e3b341",  # Gold
    "#ffa657",  # Tangerine
    "#a5d6ff",  # Pale cyan
    "#ff7b72",  # Coral red
]


def _color_for(word: str, node_type: str = "word") -> str:
    """Assign high-contrast instrument palette colors for vector types."""
    if node_type == "result":
        return "#e5533d"  # Vermilion / signal orange
    if node_type == "delta":
        return "#d99b26"  # Technical amber
    if node_type == "nearest":
        return "#2ea073"  # Jade green
    if node_type == "base":
        return "#4b8be5"  # Steel blue
    if node_type == "source":
        return "#d9536f"  # Rose coral
    if node_type == "target":
        return "#36a3b8"  # Cyan

    digest = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16)
    return _STANDARD_PALETTE[digest % len(_STANDARD_PALETTE)]


def _project_with_stats() -> tuple[list[dict[str, Any]], list[float] | None]:
    """Project all active embeddings to 3D space using PCA (or fallback) and return explained variance."""
    if not _order:
        return [], None

    vectors = np.array([_embeddings[w] for w in _order], dtype=np.float32)
    explained_var: list[float] | None = None

    if len(_order) < 3:
        coords = vectors @ _fallback_matrix
    else:
        pca = PCA(n_components=3)
        coords = pca.fit_transform(vectors)
        explained_var = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    max_norm = np.linalg.norm(coords, axis=1).max()
    if max_norm > 0:
        coords = coords / max_norm * RADIUS

    points = []
    for word, c in zip(_order, coords):
        item_meta = _meta.get(word, {})
        node_type = item_meta.get("type", "word")
        points.append(
            {
                "word": word,
                "x": round(float(c[0]), 4),
                "y": round(float(c[1]), 4),
                "z": round(float(c[2]), 4),
                "color": item_meta.get("color") or _color_for(word, node_type),
                "type": node_type,
                "meta": item_meta,
            }
        )
    return points, explained_var


def _project() -> list[dict[str, Any]]:
    """Backwards-compatible helper returning projected 3D points."""
    pts, _ = _project_with_stats()
    return pts


def _compute_similarity_matrix() -> dict[str, Any]:
    """Compute pairwise cosine similarity matrix for all active vectors in <0.1ms."""
    if not _order:
        return {"words": [], "matrix": []}
    mat = np.array([_embeddings[w] for w in _order], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    unit_mat = mat / norms
    sim_mat = np.clip(unit_mat @ unit_mat.T, -1.0, 1.0)
    return {
        "words": list(_order),
        "matrix": [[round(float(val), 4) for val in row] for row in sim_mat],
    }


def _scene_payload(**extra: Any) -> dict[str, Any]:
    """Build standard scene response with 3D points, links, PCA variance, and similarity matrix."""
    pts, explained_var = _project_with_stats()
    payload: dict[str, Any] = {
        "points": pts,
        "links": _get_links(pts),
        "dimension": 2560,
        "vocab_size": len(vocab_manager.words),
        "explained_variance": explained_var,
        "similarity_matrix": _compute_similarity_matrix(),
    }
    payload.update(extra)
    return payload


def _upsert_link(link_item: dict[str, Any]) -> None:
    """Insert or update a link between two nodes without duplicating."""
    for idx, existing in enumerate(_links):
        if existing.get("from") == link_item.get("from") and existing.get("to") == link_item.get("to"):
            _links[idx] = link_item
            return
    _links.append(link_item)


def _get_links(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve 3D coordinates for all active links between nodes."""
    coord_map = {p["word"]: [p["x"], p["y"], p["z"]] for p in points}
    resolved = []
    for link in _links:
        from_id = link.get("from")
        to_id = link.get("to")
        if from_id in coord_map and to_id in coord_map:
            resolved.append(
                {
                    "from": from_id,
                    "to": to_id,
                    "from_coord": coord_map[from_id],
                    "to_coord": coord_map[to_id],
                    "style": link.get("style", "solid"),
                    "color": link.get("color", "#94a3b8"),
                    "label": link.get("label", ""),
                }
            )
    return resolved


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/state", methods=["GET"])
def state():
    """Return current 3D scene points, links, PCA variance, and similarity matrix."""
    return jsonify(_scene_payload())


@app.route("/api/embed", methods=["POST"])
def embed():
    data = request.get_json(force=True) or {}
    raw_query = (data.get("word") or "").strip()
    exclude_inputs = bool(data.get("exclude_inputs", True))
    if not raw_query:
        return jsonify({"error": "word is required"}), 400

    # Ensure model and vocab are ready
    load_model()

    # Check if query is a vector arithmetic expression (contains + or -)
    if ("+" in raw_query or "-" in raw_query) and len(raw_query.split()) > 1:
        try:
            expr_data = evaluate_vector_expression(
                raw_query, top_k=5, exclude_inputs=exclude_inputs
            )
            expr_label = raw_query

            # 1. Register constituent terms so they appear in 3D
            for op, term in expr_data["terms"]:
                if term not in _embeddings:
                    vec = get_embedding(term)
                    _embeddings[term] = np.array(vec, dtype=np.float32)
                    _order.append(term)
                    _meta[term] = {"type": "term"}

            # 2. Register the resulting expression vector
            _embeddings[expr_label] = np.array(expr_data["vector"], dtype=np.float32)
            if expr_label not in _order:
                _order.append(expr_label)
            _meta[expr_label] = {
                "type": "result",
                "color": "#e5533d",
                "nearest_words": expr_data["nearest_words"],
                "closest_word": expr_data["closest_word"],
                "closest_similarity": expr_data["closest_similarity"],
            }

            # 3. Register the closest vocabulary word if available
            closest = expr_data.get("closest_word")
            if closest:
                if closest not in _embeddings:
                    closest_vec = get_embedding(closest)
                    _embeddings[closest] = np.array(closest_vec, dtype=np.float32)
                    _order.append(closest)
                _meta[closest] = {
                    "type": "nearest",
                    "color": "#2ea073",
                    "similarity": expr_data["closest_similarity"],
                }

                # Link expression to nearest word
                _upsert_link(
                    {
                        "from": expr_label,
                        "to": closest,
                        "style": "dotted",
                        "color": "#2ea073",
                        "label": f"cos {expr_data['closest_similarity']:.4f}",
                    }
                )

            return jsonify(_scene_payload(expression=expr_data))

        except Exception as e:
            return jsonify({"error": f"Failed to evaluate expression: {str(e)}"}), 400

    # Single word / phrase
    word_info = find_similar_words(raw_query, top_k=5, exclude_self=True)
    if raw_query not in _embeddings:
        _embeddings[raw_query] = np.array(word_info["vector"], dtype=np.float32)
        _order.append(raw_query)
    _meta[raw_query] = {
        "type": _meta.get(raw_query, {}).get("type", "word"),
        "nearest_words": word_info["nearest_words"],
        "closest_word": word_info["closest_word"],
        "closest_similarity": word_info["closest_similarity"],
    }

    return jsonify(_scene_payload(word_info=word_info))


@app.route("/api/transform", methods=["POST"])
def transform():
    """
    Apply semantic transformation: Base + (Target - Source) -> Nearest Word.
    """
    data = request.get_json(force=True) or {}
    base = (data.get("base") or "").strip()
    source = (data.get("source") or "").strip()
    target = (data.get("target") or "").strip()
    plot_components = bool(data.get("plot_components", True))
    exclude_inputs = bool(data.get("exclude_inputs", True))

    if not base or not source or not target:
        return jsonify({"error": "base, source, and target are all required"}), 400

    load_model()

    try:
        res = apply_transformation(
            base, source, target, top_k=5, exclude_inputs=exclude_inputs
        )

        # Labels
        delta_label = f"{target} - {source}"
        result_label = f"{base} + ({target} - {source})"
        closest = res["closest_word"]

        # Plot components if requested
        if plot_components:
            # 1. Base
            _embeddings[base] = np.array(res["base_vector"], dtype=np.float32)
            if base not in _order:
                _order.append(base)
            _meta[base] = {"type": "base", "color": "#4b8be5"}

            # 2. Source
            v_source = get_embedding(source)
            _embeddings[source] = np.array(v_source, dtype=np.float32)
            if source not in _order:
                _order.append(source)
            _meta[source] = {"type": "source", "color": "#d9536f"}

            # 3. Target
            v_target = get_embedding(target)
            _embeddings[target] = np.array(v_target, dtype=np.float32)
            if target not in _order:
                _order.append(target)
            _meta[target] = {"type": "target", "color": "#36a3b8"}

            # 4. Difference vector delta
            _embeddings[delta_label] = np.array(res["delta_vector"], dtype=np.float32)
            if delta_label not in _order:
                _order.append(delta_label)
            _meta[delta_label] = {"type": "delta", "color": "#d99b26"}

        # 5. Resultant vector
        _embeddings[result_label] = np.array(res["result_vector"], dtype=np.float32)
        if result_label not in _order:
            _order.append(result_label)
        _meta[result_label] = {
            "type": "result",
            "color": "#e5533d",
            "nearest_words": res["nearest_words"],
            "closest_word": closest,
            "closest_similarity": res["closest_similarity"],
        }

        # 6. Closest actual word
        if closest:
            v_closest = get_embedding(closest)
            _embeddings[closest] = np.array(v_closest, dtype=np.float32)
            if closest not in _order:
                _order.append(closest)
            _meta[closest] = {
                "type": "nearest",
                "color": "#2ea073",
                "similarity": res["closest_similarity"],
            }

            # Add guide line: Result vector -> Nearest vocabulary word
            _upsert_link(
                {
                    "from": result_label,
                    "to": closest,
                    "style": "dotted",
                    "color": "#2ea073",
                    "label": f"cos {res['closest_similarity']:.4f}",
                }
            )

        # Add displacement guide line: Base -> Result
        if plot_components:
            _upsert_link(
                {
                    "from": base,
                    "to": result_label,
                    "style": "dashed",
                    "color": "#d99b26",
                    "label": f"+ ({delta_label})",
                }
            )

        return jsonify(_scene_payload(transformation=res))

    except Exception as e:
        return jsonify({"error": f"Transformation failed: {str(e)}"}), 500


@app.route("/api/similarity", methods=["POST"])
def similarity():
    """
    Compute exact cosine similarity between two words/phrases, plot both, and link them in 3D.
    """
    data = request.get_json(force=True) or {}
    word_a = (data.get("word_a") or "").strip()
    word_b = (data.get("word_b") or "").strip()
    if not word_a or not word_b:
        return jsonify({"error": "word_a and word_b are required"}), 400

    load_model()

    try:
        res = compute_similarity(word_a, word_b)
        for w, vec in [(word_a, res["vector_a"]), (word_b, res["vector_b"])]:
            if w not in _embeddings:
                _embeddings[w] = np.array(vec, dtype=np.float32)
                _order.append(w)
                info = find_similar_words(w, top_k=5, exclude_self=True)
                _meta[w] = {
                    "type": "word",
                    "nearest_words": info["nearest_words"],
                    "closest_word": info["closest_word"],
                    "closest_similarity": info["closest_similarity"],
                }

        _upsert_link(
            {
                "from": word_a,
                "to": word_b,
                "style": "dotted",
                "color": "#4b8be5",
                "label": f"cos {res['similarity']:.4f}",
            }
        )

        return jsonify(_scene_payload(similarity=res))
    except Exception as e:
        return jsonify({"error": f"Similarity calculation failed: {str(e)}"}), 500


@app.route("/api/remove", methods=["POST"])
def remove():
    """Remove a single word/vector from the active scene."""
    data = request.get_json(force=True) or {}
    word = (data.get("word") or "").strip()
    if not word:
        return jsonify({"error": "word is required"}), 400

    _embeddings.pop(word, None)
    _meta.pop(word, None)
    if word in _order:
        _order.remove(word)
    _links[:] = [
        lnk for lnk in _links if lnk.get("from") != word and lnk.get("to") != word
    ]

    return jsonify(_scene_payload())


@app.route("/api/reset", methods=["POST"])
def reset():
    _embeddings.clear()
    _order.clear()
    _meta.clear()
    _links.clear()
    return jsonify(_scene_payload())


if __name__ == "__main__":
    load_model()
    app.run(host="127.0.0.1", port=5050, debug=False)
