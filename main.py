"""
Vector Embedding Model — Qwen3-Embedding-4B

Loads the Qwen3-Embedding-4B model and exposes functions to:
1. Turn a word/phrase into an n-dimensional embedding vector (2560-dim).
2. Evaluate vector additions and subtractions (e.g., "woman - man", "king - man + woman").
3. Apply analogical transformations: Base + (Target - Source), e.g., uncle + (woman - man) -> aunt.
4. Find the closest vocabulary word to any arbitrary embedding vector.

Run interactively:
    python main.py

Or evaluate an expression directly:
    python main.py "king - man + woman"
    python main.py transform uncle man woman
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from vocab_manager import vocab_manager

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
CACHE_DIR = Path(__file__).resolve().parent / "models"

# Pick the best available device: Apple Silicon GPU (MPS) > CUDA > CPU.
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

_model: SentenceTransformer | None = None


def load_model() -> SentenceTransformer:
    """Load (once) and cache the Qwen3-Embedding-4B model and vocabulary."""
    global _model
    if _model is None:
        print(f"Loading {MODEL_NAME} on {DEVICE} ... (first run downloads ~8GB, be patient)")
        _model = SentenceTransformer(
            MODEL_NAME,
            device=DEVICE,
            model_kwargs={"torch_dtype": torch.float16},
            cache_folder=str(CACHE_DIR),
        )
        dim = getattr(_model, "get_embedding_dimension", _model.get_sentence_embedding_dimension)()
        print(f"Model loaded. Embedding dimension: {dim}")

        # Ensure vocabulary is loaded or built
        vocab_manager.load_or_build(_model)

    return _model


def get_embedding(text: str) -> list[float]:
    """
    Return the n-dimensional embedding vector for a given word or piece of text.

    Args:
        text: the word/phrase to embed.

    Returns:
        A list of floats representing the unit-normalized embedding vector.
    """
    model = load_model()
    vector = model.encode(text, normalize_embeddings=True)
    vec_list = vector.tolist()
    # Register word in vocabulary index
    vocab_manager.add_word(text, vector)
    return vec_list


def parse_vector_expression(expr: str) -> list[tuple[str, str]]:
    """
    Parse a vector arithmetic expression into signed terms.

    Supports:
        "man + woman" -> [('+', 'man'), ('+', 'woman')]
        "woman - man" -> [('+', 'woman'), ('-', 'man')]
        "king - man + woman" -> [('+', 'king'), ('-', 'man'), ('+', 'woman')]
        "uncle + (woman - man)" -> [('+', 'uncle'), ('+', 'woman'), ('-', 'man')]

    Returns:
        List of (operator, term) tuples where operator is '+' or '-'.
    """
    # Clean string: remove outer parens or brackets
    clean = expr.replace("(", " ").replace(")", " ").strip()
    # Split tokens around + and - while retaining operators
    pattern = re.compile(r"([+-])")
    tokens = [t.strip() for t in pattern.split(clean) if t.strip()]

    terms: list[tuple[str, str]] = []
    current_op = "+"

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in {"+", "-"}:
            current_op = token
            i += 1
            continue

        terms.append((current_op, token))
        current_op = "+"  # default next op
        i += 1

    return terms


def evaluate_vector_expression(
    expr: str,
    top_k: int = 5,
    exclude_inputs: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a vector arithmetic expression (additions and subtractions)
    and find its nearest vocabulary words.

    Returns dict containing:
        - expression: str
        - terms: list of (sign, word)
        - vector: unit-normalized result vector (list of floats)
        - nearest_words: list of {"word": str, "similarity": float}
        - closest_word: str (excluding input terms when exclude_inputs=True)
    """
    load_model()
    terms = parse_vector_expression(expr)
    if not terms:
        raise ValueError(f"Could not parse vector expression: '{expr}'")

    input_words = [t[1] for t in terms] if exclude_inputs else []
    result_vec = np.zeros(2560, dtype=np.float32)

    for op, term in terms:
        vec = np.array(get_embedding(term), dtype=np.float32)
        if op == "+":
            result_vec += vec
        elif op == "-":
            result_vec -= vec

    norm = np.linalg.norm(result_vec)
    if norm > 0:
        result_vec = result_vec / norm

    nearest = vocab_manager.find_nearest(
        result_vec,
        top_k=top_k,
        exclude_words=input_words,
    )

    return {
        "expression": expr,
        "terms": terms,
        "vector": result_vec.tolist(),
        "nearest_words": nearest,
        "closest_word": nearest[0]["word"] if nearest else None,
        "closest_similarity": nearest[0]["similarity"] if nearest else None,
    }


def apply_transformation(
    base: str,
    source: str,
    target: str,
    top_k: int = 5,
    exclude_inputs: bool = True,
) -> dict[str, Any]:
    """
    Apply a semantic transformation vector (Target - Source) to a Base concept:

        Result = Base + (Target - Source)

    Example:
        base = "uncle", source = "man", target = "woman"
        Result = "uncle" + ("woman" - "man") -> nearest: "aunt" / "auntie"

    Args:
        base: The starting base word/phrase (e.g., "uncle").
        source: The source domain anchor word (e.g., "man").
        target: The target domain anchor word (e.g., "woman").
        top_k: Number of nearest neighbors to return.
        exclude_inputs: Whether to exclude base, source, target from nearest words.

    Returns:
        dict containing base_vector, delta_vector, result_vector,
        and nearest_words ranked by cosine similarity.
    """
    load_model()

    vec_base = np.array(get_embedding(base), dtype=np.float32)
    vec_source = np.array(get_embedding(source), dtype=np.float32)
    vec_target = np.array(get_embedding(target), dtype=np.float32)

    # 1. The difference vector (Target - Source)
    delta = vec_target - vec_source

    # 2. Result vector = Base + Delta
    raw_result = vec_base + delta
    norm = np.linalg.norm(raw_result)
    result_vec = raw_result / norm if norm > 0 else raw_result

    # 3. Normalized delta for plotting
    delta_norm = np.linalg.norm(delta)
    unit_delta = delta / delta_norm if delta_norm > 0 else delta

    # 4. Nearest word lookup (excluding the 3 input query words by default)
    exclude_list = [base, source, target] if exclude_inputs else []
    nearest = vocab_manager.find_nearest(
        result_vec,
        top_k=top_k,
        exclude_words=exclude_list,
    )

    closest_word = nearest[0]["word"] if nearest else None
    closest_sim = nearest[0]["similarity"] if nearest else None

    # Also register the result's closest word if present
    if closest_word:
        get_embedding(closest_word)

    return {
        "base": base,
        "source": source,
        "target": target,
        "delta_expr": f"{target} - {source}",
        "full_expr": f"{base} + ({target} - {source})",
        "base_vector": vec_base.tolist(),
        "delta_vector": unit_delta.tolist(),
        "result_vector": result_vec.tolist(),
        "nearest_words": nearest,
        "closest_word": closest_word,
        "closest_similarity": closest_sim,
    }


def compute_similarity(word_a: str, word_b: str) -> dict[str, Any]:
    """
    Compute exact cosine similarity between two words or phrases in 2560-dim space.
    """
    load_model()
    vec_a = np.array(get_embedding(word_a), dtype=np.float32)
    vec_b = np.array(get_embedding(word_b), dtype=np.float32)
    sim = float(np.dot(vec_a, vec_b))
    return {
        "word_a": word_a,
        "word_b": word_b,
        "similarity": round(sim, 4),
        "vector_a": vec_a.tolist(),
        "vector_b": vec_b.tolist(),
    }


def find_similar_words(word: str, top_k: int = 5, exclude_self: bool = True) -> dict[str, Any]:
    """
    Embed a single word/phrase and return its top_k nearest neighbors in the vocabulary.
    """
    load_model()
    vec = np.array(get_embedding(word), dtype=np.float32)
    nearest = vocab_manager.find_nearest(
        vec,
        top_k=top_k,
        exclude_words=[word] if exclude_self else [],
    )
    return {
        "word": word,
        "vector": vec.tolist(),
        "nearest_words": nearest,
        "closest_word": nearest[0]["word"] if nearest else None,
        "closest_similarity": nearest[0]["similarity"] if nearest else None,
    }


def main() -> None:
    # Command-line mode
    if len(sys.argv) > 1:
        load_model()
        first_arg = sys.argv[1].strip()

        # Transformation CLI: `python main.py transform uncle man woman`
        if first_arg == "transform" and len(sys.argv) >= 5:
            base, source, target = sys.argv[2], sys.argv[3], sys.argv[4]
            print(f"\nEvaluating Transformation: {base} + ({target} - {source})")
            res = apply_transformation(base, source, target)
            print(f"Closest Word: '{res['closest_word']}' (Cosine Similarity: {res['closest_similarity']})")
            print("\nTop Candidates:")
            for item in res["nearest_words"]:
                print(f"  • {item['word']:<14} (similarity: {item['similarity']:.4f})")
            return

        # Similarity CLI: `python main.py similarity king queen`
        if first_arg in {"similarity", "sim"} and len(sys.argv) >= 4:
            word_a, word_b = sys.argv[2], sys.argv[3]
            res = compute_similarity(word_a, word_b)
            print(f"\nCosine Similarity ('{word_a}', '{word_b}'): {res['similarity']:.4f}")
            return

        # Expression CLI: `python main.py "king - man + woman"`
        expr = " ".join(sys.argv[1:])
        if "+" in expr or "-" in expr:
            print(f"\nEvaluating Vector Expression: {expr}")
            res = evaluate_vector_expression(expr)
            print(f"Closest Word: '{res['closest_word']}' (Cosine Similarity: {res['closest_similarity']})")
            print("\nTop Candidates:")
            for item in res["nearest_words"]:
                print(f"  • {item['word']:<14} (similarity: {item['similarity']:.4f})")
            return

        # Single word: `python main.py "ocean"`
        res = find_similar_words(expr)
        vector = res["vector"]
        print(f"\n'{expr}' -> {len(vector)}-dim vector")
        print(f"First 10 values: {vector[:10]} ...")
        print("\nNearest Vocabulary Neighbors:")
        for item in res["nearest_words"]:
            print(f"  • {item['word']:<14} (similarity: {item['similarity']:.4f})")
        return

    # Interactive mode
    load_model()
    print("\nEnter a word, expression (e.g. 'king - man + woman'),")
    print("or transformation (e.g. 'transform uncle man woman'):")
    print("(Type 'quit' or 'exit' to stop)")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            print("Exiting.")
            break

        if line.startswith("transform ") and len(line.split()) >= 4:
            parts = line.split()
            base, source, target = parts[1], parts[2], parts[3]
            res = apply_transformation(base, source, target)
            print(f"\n=> Transformation: {base} + ({target} - {source})")
            print(f"   Closest Word: {res['closest_word']} (Similarity: {res['closest_similarity']})")
            for item in res["nearest_words"]:
                print(f"   • {item['word']:<14} ({item['similarity']:.4f})")
            print()
            continue

        if "+" in line or "-" in line:
            res = evaluate_vector_expression(line)
            print(f"\n=> Expression: {line}")
            print(f"   Closest Word: {res['closest_word']} (Similarity: {res['closest_similarity']})")
            for item in res["nearest_words"]:
                print(f"   • {item['word']:<14} ({item['similarity']:.4f})")
            print()
            continue

        vector = get_embedding(line)
        print(f"Dimension: {len(vector)}")
        print(f"Vector (first 10 values): {vector[:10]} ...")


if __name__ == "__main__":
    main()
