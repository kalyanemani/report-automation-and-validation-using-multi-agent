from __future__ import annotations

"""
QDR·AI — Embedding Manager
────────────────────────────
Manages BGE-M3 embeddings (1024-dim) for hybrid vector search.
Guarantees 1024-dimensional dense vectors to match ChromaDB collections.
"""

import logging
from typing import Optional
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION, EmbeddingModelError

logger = logging.getLogger("qdr_ai.embeddings")

# Global singleton — lazy loaded
_embedding_model = None


def get_embedding_model():
    """
    Get the embedding model instance (singleton).
    Guarantees 1024-dim vectors matching ChromaDB collection schema.
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    # Primary model: BAEI/bge-m3 (1024-dim)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info(f"Loading BGE-M3 embedding model: {EMBEDDING_MODEL_NAME}")

        model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
            },
        )
        test = model.embed_query("test")
        dim = len(test)
        if dim != EMBEDDING_DIMENSION:
            # A dimension mismatch means every similarity score against the
            # existing collections is meaningless, so this is fatal too.
            raise EmbeddingModelError(
                f"{EMBEDDING_MODEL_NAME} produced {dim}-dim vectors but the ChromaDB "
                f"collections are {EMBEDDING_DIMENSION}-dim. Retrieval would be invalid. "
                f"Check EMBEDDING_MODEL / EMBEDDING_DIMENSION in .env."
            )
        logger.info(f"[OK] Loaded {EMBEDDING_MODEL_NAME} — dimension={dim}")
        _embedding_model = model
        return _embedding_model

    except EmbeddingModelError:
        raise
    except Exception as e:
        # Previously this fell through to FakeEmbeddings(size=1024), which
        # returns RANDOM vectors. Retrieval silently became noise while every
        # downstream "retrieved specification" citation still looked genuine.
        # There is no safe way to continue: fail loudly instead.
        logger.error(f"Could not load embedding model {EMBEDDING_MODEL_NAME}: {e}")
        raise EmbeddingModelError(
            f"Failed to load embedding model {EMBEDDING_MODEL_NAME}: {e}. "
            f"RAG retrieval is unavailable — refusing to continue with random "
            f"vectors, which would silently invalidate every retrieved citation."
        ) from e


def reset_embedding_model():
    """Reset the singleton (useful for testing)."""
    global _embedding_model
    _embedding_model = None
