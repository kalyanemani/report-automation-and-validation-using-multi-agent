"""
QDR·AI — ChromaDB Vector Store Manager
────────────────────────────────────────
Manages ChromaDB collections for document chunks.
Supports:
  - Ingesting chunks with rich metadata
  - Similarity search with metadata filtering
  - Parent-child retrieval (retrieve child, return parent context)
"""

import logging
from pathlib import Path
from typing import Optional

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import CHROMA_PERSIST_DIR, EMBED_TOP_K
from src.ingestion.chunker import Chunk
from src.vectorstore.embeddings import get_embedding_model

logger = logging.getLogger("qdr_ai.store")


class VectorStoreManager:
    """
    Manages ChromaDB collections for QDR document chunks.

    Collections:
      - qdr_{guide_code}_children : fine-grained chunks for retrieval
      - qdr_{guide_code}_parents  : larger context chunks for LLM

    Usage:
        manager = VectorStoreManager()
        manager.ingest_chunks("HRD", child_chunks, parent_chunks)
        results = manager.search("HRD", "hardness test acceptance criteria")
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or str(CHROMA_PERSIST_DIR)
        self.embedding_model = get_embedding_model()
        self._stores: dict[str, Chroma] = {}
        self._parent_stores: dict[str, Chroma] = {}

        logger.info(
            f"VectorStoreManager initialized — "
            f"persist_dir={self.persist_dir}, "
            f"embedding={type(self.embedding_model).__name__}"
        )

    def _get_collection_name(self, guide_code: str, is_parent: bool = False) -> str:
        suffix = "parents" if is_parent else "children"
        return f"qdr_{guide_code.lower()}_{suffix}"

    def ingest_chunks(
        self,
        guide_code: str,
        child_chunks: list[Chunk],
        parent_chunks: Optional[list[Chunk]] = None,
    ) -> dict:
        """
        Embed and store chunks in ChromaDB.

        Args:
            guide_code: identifier (e.g., "HRD", "TEN")
            child_chunks: fine-grained chunks for retrieval
            parent_chunks: larger chunks for LLM context (optional)

        Returns:
            Summary dict with counts
        """
        # ── Ingest children ──────────────────────────────────────────
        child_collection = self._get_collection_name(guide_code, is_parent=False)

        child_docs = [
            Document(
                page_content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "chunk_id": chunk.chunk_id,
                    "parent_chunk_id": chunk.parent_chunk_id or "",
                },
            )
            for chunk in child_chunks
            if chunk.content.strip()
        ]

        if child_docs:
            logger.info(f"Embedding {len(child_docs)} child chunks for [{guide_code}]...")

            # Drop existing collection
            self._drop_collection(child_collection)

            vs = Chroma.from_documents(
                documents=child_docs,
                embedding=self.embedding_model,
                collection_name=child_collection,
                persist_directory=self.persist_dir,
            )
            self._stores[guide_code] = vs

            count = vs._collection.count()
            logger.info(f"[OK] [{child_collection}] -> {count} vectors stored")

        # ── Ingest parents ───────────────────────────────────────────
        parent_count = 0
        if parent_chunks:
            parent_collection = self._get_collection_name(guide_code, is_parent=True)

            parent_docs = [
                Document(
                    page_content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "chunk_id": chunk.chunk_id,
                        "child_ids": ",".join(chunk.children_ids),
                    },
                )
                for chunk in parent_chunks
                if chunk.content.strip()
            ]

            if parent_docs:
                self._drop_collection(parent_collection)

                ps = Chroma.from_documents(
                    documents=parent_docs,
                    embedding=self.embedding_model,
                    collection_name=parent_collection,
                    persist_directory=self.persist_dir,
                )
                self._parent_stores[guide_code] = ps
                parent_count = ps._collection.count()
                logger.info(f"[OK] [{parent_collection}] -> {parent_count} parent vectors stored")

        return {
            "guide_code": guide_code,
            "children_stored": len(child_docs),
            "parents_stored": parent_count,
            "embedding_model": type(self.embedding_model).__name__,
        }

    def search(
        self,
        guide_code: str,
        query: str,
        k: int = EMBED_TOP_K,
        filter_metadata: Optional[dict] = None,
        return_parents: bool = False,
    ) -> list[dict]:
        """
        Search for relevant chunks.
        """
        collection_name = self._get_collection_name(guide_code, is_parent=False)
        store = self._stores.get(guide_code)
        
        if not store:
            try:
                store = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embedding_model,
                    persist_directory=self.persist_dir,
                )
                self._stores[guide_code] = store
            except Exception as e:
                logger.warning(f"Could not load vector store [{collection_name}]: {e}")
                return []

        logger.info(
            f"Search [{guide_code}]: query='{query[:60]}...' k={k}"
        )

        # Build search kwargs
        search_kwargs = {"k": k}
        if filter_metadata:
            search_kwargs["filter"] = filter_metadata

        try:
            results = store.similarity_search_with_score(query, **search_kwargs)
        except Exception as e:
            logger.warning(f"Search failed with existing handle for [{collection_name}]: {e}. Reconnecting...")
            try:
                store = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embedding_model,
                    persist_directory=self.persist_dir,
                )
                self._stores[guide_code] = store
                results = store.similarity_search_with_score(query, **search_kwargs)
            except Exception as retry_err:
                logger.error(f"Search failed on retry for [{collection_name}]: {retry_err}")
                return []

        output = []
        for doc, score in results:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }

            # Optionally fetch parent content for LLM context
            if return_parents:
                parent_id = doc.metadata.get("parent_chunk_id", "")
                if parent_id and guide_code in self._parent_stores:
                    parent_content = self._get_parent_content(guide_code, parent_id)
                    if parent_content:
                        result["parent_content"] = parent_content

            output.append(result)

        logger.info(f"Search returned {len(output)} results")
        return output

    def _get_parent_content(self, guide_code: str, parent_id: str) -> Optional[str]:
        """Retrieve parent chunk content by ID."""
        parent_store = self._parent_stores.get(guide_code)
        if not parent_store:
            return None

        # Search for the parent by chunk_id metadata
        results = parent_store.get(where={"chunk_id": parent_id})
        if results and results.get("documents"):
            return results["documents"][0]
        return None

    def _drop_collection(self, collection_name: str):
        """Drop an existing ChromaDB collection."""
        try:
            existing = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_dir,
            )
            count = existing._collection.count()
            existing.delete_collection()
            logger.info(f"Dropped [{collection_name}] ({count} vectors)")
        except Exception:
            pass

    def list_stores(self) -> dict:
        """List all loaded vector stores and their stats."""
        info = {}
        for key, store in self._stores.items():
            try:
                count = store._collection.count()
            except Exception:
                count = 0

            info[key] = {
                "children_count": count,
                "embedding_model": type(self.embedding_model).__name__,
            }

            if key in self._parent_stores:
                try:
                    pcount = self._parent_stores[key]._collection.count()
                except Exception:
                    pcount = 0
                info[key]["parents_count"] = pcount

        return info
