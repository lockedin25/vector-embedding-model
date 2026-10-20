"""
Vocabulary Manager for Semantic Nearest-Neighbor Search.

Maintains a searchable corpus of common English words and benchmark analogy concepts.
Computes and caches normalized embeddings in `vocab_cache.npz` so nearest-neighbor
lookups take < 1 ms via a single BLAS matrix-vector dot product.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).resolve().parent / "vocab_cache.npz"

# Core benchmark words for analogical reasoning
CURATED_BENCHMARKS = [
    # Kinship & Gender
    "man", "woman", "men", "women", "boy", "girl", "boys", "girls",
    "father", "mother", "dad", "mom", "parent", "parents",
    "son", "daughter", "brother", "sister", "uncle", "aunt", "auntie",
    "nephew", "niece", "cousin", "husband", "wife", "spouse",
    "grandfather", "grandmother", "grandpa", "grandma", "grandson", "granddaughter",
    "male", "female", "masculine", "feminine", "gentleman", "lady", "sir", "madam",
    # Royalty & Nobility
    "king", "queen", "prince", "princess", "monarch", "sovereign", "ruler",
    "emperor", "empress", "empire", "kingdom", "realm", "throne", "crown",
    "duke", "duchess", "baron", "baroness", "lord", "lady", "knight", "noble",
    # Professions & Roles
    "doctor", "physician", "surgeon", "nurse", "dentist", "paramedic",
    "teacher", "professor", "educator", "instructor", "tutor", "student", "pupil",
    "engineer", "architect", "mechanic", "technician", "programmer", "developer",
    "scientist", "researcher", "chemist", "physicist", "biologist", "mathematician",
    "lawyer", "attorney", "judge", "juror", "prosecutor", "bailiff",
    "police", "officer", "detective", "sheriff", "guard", "soldier", "warrior",
    "pilot", "captain", "sailor", "astronaut", "driver",
    "chef", "cook", "baker", "waiter", "waitress", "bartender",
    "actor", "actress", "director", "producer", "singer", "musician", "artist", "painter",
    "author", "writer", "poet", "journalist", "reporter", "editor",
    "farmer", "gardener", "fisherman", "hunter", "miner", "carpenter",
    "president", "minister", "governor", "mayor", "senator", "diplomat", "ambassador",
    "boss", "manager", "executive", "worker", "employee", "staff",
    # Geography & Capitals
    "france", "paris", "french", "italy", "rome", "italian",
    "germany", "berlin", "german", "spain", "madrid", "spanish",
    "england", "london", "english", "britain", "british", "uk",
    "russia", "moscow", "russian", "china", "beijing", "chinese",
    "japan", "tokyo", "japanese", "india", "delhi", "indian",
    "egypt", "cairo", "egyptian", "greece", "athens", "greek",
    "canada", "ottawa", "canadian", "australia", "canberra", "australian",
    "brazil", "brasilia", "brazilian", "mexico", "america", "american",
    "city", "capital", "country", "nation", "state", "continent", "village", "town",
    # Animals (adult/young, male/female)
    "dog", "puppy", "hound", "canine", "cat", "kitten", "feline",
    "horse", "stallion", "mare", "foal", "pony", "donkey", "mule",
    "cow", "bull", "calf", "cattle", "ox", "sheep", "ram", "ewe", "lamb",
    "pig", "boar", "sow", "piglet", "goat", "kid",
    "lion", "lioness", "cub", "tiger", "tigress", "leopard", "cheetah",
    "bear", "wolf", "fox", "deer", "stag", "doe", "fawn", "moose", "elk",
    "elephant", "giraffe", "zebra", "hippo", "rhino", "camel", "kangaroo",
    "monkey", "ape", "chimpanzee", "gorilla",
    "bird", "eagle", "hawk", "falcon", "owl", "crow", "raven", "pigeon", "dove",
    "duck", "drake", "duckling", "goose", "gander", "gosling", "swan",
    "chicken", "rooster", "hen", "chick", "turkey",
    "fish", "shark", "whale", "dolphin", "salmon", "trout", "tuna",
    "snake", "serpent", "lizard", "crocodile", "alligator", "turtle", "frog",
    # Nature & Elements
    "sun", "moon", "star", "planet", "sky", "earth", "ocean", "sea", "lake", "river",
    "mountain", "valley", "forest", "desert", "island", "beach", "tree", "flower",
    "fire", "water", "air", "wind", "rain", "snow", "ice", "cloud", "storm", "lightning",
    # Tech & Science
    "computer", "software", "hardware", "code", "data", "algorithm", "network", "internet",
    "robot", "machine", "engine", "device", "phone", "camera", "screen", "keyboard",
    "energy", "atom", "molecule", "cell", "gene", "gravity", "quantum", "space", "time",
    # Time & Seasons
    "day", "night", "morning", "afternoon", "evening", "dawn", "dusk", "midnight", "noon",
    "hour", "minute", "second", "week", "month", "year", "decade", "century",
    "spring", "summer", "autumn", "fall", "winter", "yesterday", "today", "tomorrow",
    # Emotions & Abstract concepts
    "love", "hate", "peace", "war", "joy", "sorrow", "happiness", "sadness", "anger", "fear",
    "hope", "despair", "courage", "bravery", "wisdom", "knowledge", "truth", "lie",
    "beauty", "freedom", "justice", "power", "honor", "glory", "wealth", "poverty",
    "life", "death", "birth", "health", "disease", "strength", "weakness",
    # Common Antonyms & Gradable Adjectives
    "good", "bad", "great", "terrible", "wonderful", "awful", "best", "worst",
    "big", "small", "huge", "tiny", "large", "little", "giant", "massive",
    "hot", "cold", "warm", "cool", "freezing", "boiling",
    "fast", "slow", "quick", "rapid", "swift",
    "rich", "poor", "wealthy", "cheap", "expensive",
    "strong", "weak", "powerful", "fragile",
    "heavy", "light", "hard", "soft", "tough",
    "dark", "bright", "shiny", "dim",
    "young", "old", "ancient", "modern", "new", "fresh",
    "clean", "dirty", "pure", "messy",
    "smart", "clever", "intelligent", "stupid", "foolish", "dumb",
    "happy", "sad", "cheerful", "depressed", "glad", "miserable",
    "safe", "dangerous", "easy", "difficult", "simple", "complex",
    "open", "closed", "empty", "full", "alive", "dead",
    "sweet", "sour", "bitter", "salty", "spicy",
    "loud", "quiet", "silent", "noisy",
    "deep", "shallow", "wide", "narrow", "thick", "thin",
    "high", "low", "tall", "short",
    "right", "wrong", "true", "false", "fair", "unfair",
]


class VocabManager:
    """Manages vocabulary embeddings for fast cosine similarity search."""

    def __init__(self) -> None:
        self.words: list[str] = []
        self.embeddings: np.ndarray | None = None  # shape: (N, 2560), unit-normalized
        self._word_to_idx: dict[str, int] = {}
        self._loaded: bool = False

    def load_or_build(self, model: SentenceTransformer, target_vocab_size: int = 2500) -> None:
        """Load precomputed embeddings from cache, or build them if missing."""
        if CACHE_FILE.exists():
            print(f"Loading cached vocabulary embeddings from {CACHE_FILE.name}...")
            data = np.load(CACHE_FILE, allow_pickle=True)
            self.words = list(data["words"])
            self.embeddings = data["embeddings"].astype(np.float32)
            self._word_to_idx = {w.lower(): i for i, w in enumerate(self.words)}
            self._loaded = True
            print(f"Loaded {len(self.words)} vocabulary embeddings.")
            return

        print("No vocabulary cache found. Building vocabulary corpus...")
        words_set: list[str] = []
        seen = set()

        # 1. Add curated benchmarks first
        for w in CURATED_BENCHMARKS:
            clean = w.strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                words_set.append(clean)

        # 2. Add top words from Google 10k or system dictionary
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa-no-swears.txt"
            with urllib.request.urlopen(url, timeout=5) as resp:
                remote_words = [
                    line.decode("utf-8").strip().lower()
                    for line in resp.readlines()
                    if line.decode("utf-8").strip()
                ]
                for w in remote_words:
                    if len(words_set) >= target_vocab_size:
                        break
                    if w.isalpha() and 3 <= len(w) <= 14 and w not in seen:
                        seen.add(w)
                        words_set.append(w)
        except Exception as err:
            print(f"Could not download remote word list ({err}); falling back to local dictionary...")
            dict_path = Path("/usr/share/dict/words")
            if dict_path.exists():
                with open(dict_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if len(words_set) >= target_vocab_size:
                            break
                        w = line.strip().lower()
                        if w.isalpha() and 3 <= len(w) <= 12 and w not in seen:
                            seen.add(w)
                            words_set.append(w)

        print(f"Encoding {len(words_set)} vocabulary words in batches...")
        all_embeddings = model.encode(
            words_set,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.words = words_set
        self.embeddings = np.array(all_embeddings, dtype=np.float32)
        self._word_to_idx = {w.lower(): i for i, w in enumerate(self.words)}

        print(f"Saving vocabulary cache to {CACHE_FILE}...")
        np.savez_compressed(CACHE_FILE, words=np.array(self.words), embeddings=self.embeddings)
        self._loaded = True
        print(f"Vocabulary cache built ({len(self.words)} words) and saved successfully.")

    def add_word(self, word: str, embedding: list[float] | np.ndarray) -> None:
        """Add a single word and its embedding into the active vocabulary if not present."""
        w_clean = word.strip().lower()
        if not w_clean or w_clean in self._word_to_idx:
            return

        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vec = vec.reshape(1, -1)

        if self.embeddings is None or len(self.words) == 0:
            self.embeddings = vec
        else:
            self.embeddings = np.vstack([self.embeddings, vec])

        self._word_to_idx[w_clean] = len(self.words)
        self.words.append(w_clean)

    def find_nearest(
        self,
        query_vector: list[float] | np.ndarray,
        top_k: int = 5,
        exclude_words: list[str] | None = None,
    ) -> list[dict[str, float | str]]:
        """
        Find the top_k closest words in the vocabulary to query_vector using cosine similarity.

        Returns a list of dicts: [{"word": "aunt", "similarity": 0.820}, ...]
        """
        if self.embeddings is None or len(self.words) == 0:
            return []

        vec = np.array(query_vector, dtype=np.float32).ravel()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        scores = self.embeddings @ vec

        exclude_set = {w.strip().lower() for w in (exclude_words or [])}

        ranked_indices = np.argsort(-scores)

        results: list[dict[str, float | str]] = []
        for idx in ranked_indices:
            w = self.words[idx]
            if w in exclude_set:
                continue
            results.append({"word": w, "similarity": round(float(scores[idx]), 4)})
            if len(results) >= top_k:
                break

        return results


# Global singleton instance
vocab_manager = VocabManager()
