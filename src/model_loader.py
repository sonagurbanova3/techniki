from functools import lru_cache
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_semantic_model():
    """Ładuje model SentenceTransformer tylko raz w trakcie działania aplikacji."""
    return SentenceTransformer("all-MiniLM-L6-v2")
