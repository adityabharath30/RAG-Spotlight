"""
Tests for the vector store module.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.vector_store import FAISSVectorStore


class TestFAISSVectorStore:
    """Tests for FAISS vector store."""

    def test_init_with_dimension(self):
        """Store should initialize with given dimension."""
        store = FAISSVectorStore(dim=384)
        assert store.dim == 384
        assert store.index.ntotal == 0

    def test_init_invalid_dimension(self):
        """Store should reject invalid dimensions."""
        with pytest.raises(ValueError):
            FAISSVectorStore(dim=0)
        with pytest.raises(ValueError):
            FAISSVectorStore(dim=-1)

    def test_add_embeddings(self):
        """Store should add embeddings correctly."""
        store = FAISSVectorStore(dim=384)
        
        embeddings = np.random.rand(3, 384).astype("float32")
        metadata = [
            {"text": "doc1", "filename": "a.txt"},
            {"text": "doc2", "filename": "b.txt"},
            {"text": "doc3", "filename": "c.txt"},
        ]
        
        store.add(embeddings, metadata)
        
        assert store.index.ntotal == 3
        assert len(store.metadata) == 3

    def test_add_mismatched_lengths(self):
        """Store should reject mismatched embeddings and metadata."""
        store = FAISSVectorStore(dim=384)
        
        embeddings = np.random.rand(3, 384).astype("float32")
        metadata = [{"text": "doc1"}]  # Only 1 metadata for 3 embeddings
        
        with pytest.raises(ValueError):
            store.add(embeddings, metadata)

    def test_add_wrong_dimension(self):
        """Store should reject embeddings with wrong dimension."""
        store = FAISSVectorStore(dim=384)
        
        embeddings = np.random.rand(3, 256).astype("float32")  # Wrong dim
        metadata = [{"text": f"doc{i}"} for i in range(3)]
        
        with pytest.raises(ValueError):
            store.add(embeddings, metadata)

    def test_search_empty_store(self):
        """Search on empty store should return empty results."""
        store = FAISSVectorStore(dim=384)
        
        query = np.random.rand(1, 384).astype("float32")
        results = store.search(query, k=5)
        
        assert results == []

    def test_search_returns_results(self):
        """Search should return relevant results."""
        store = FAISSVectorStore(dim=384)
        
        # Add some embeddings
        embeddings = np.random.rand(5, 384).astype("float32")
        metadata = [{"text": f"doc{i}", "filename": f"{i}.txt"} for i in range(5)]
        store.add(embeddings, metadata)
        
        # Search with the first embedding (should find itself)
        query = embeddings[0:1]
        results = store.search(query, k=3)
        
        assert len(results) == 3
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)

    def test_search_respects_k(self):
        """Search should return at most k results."""
        store = FAISSVectorStore(dim=384)
        
        embeddings = np.random.rand(10, 384).astype("float32")
        metadata = [{"text": f"doc{i}"} for i in range(10)]
        store.add(embeddings, metadata)
        
        query = np.random.rand(1, 384).astype("float32")
        
        results_3 = store.search(query, k=3)
        results_5 = store.search(query, k=5)
        
        assert len(results_3) == 3
        assert len(results_5) == 5

    def test_save_and_load(self, temp_dir):
        """Store should save and load correctly."""
        store = FAISSVectorStore(dim=384)
        
        embeddings = np.random.rand(5, 384).astype("float32")
        metadata = [{"text": f"doc{i}", "value": i} for i in range(5)]
        store.add(embeddings, metadata)
        
        # Save
        save_path = temp_dir / "test_index"
        store.save(save_path)
        
        # Verify files exist
        assert (save_path.with_suffix(".faiss")).exists()
        assert (save_path.with_suffix(".pkl")).exists()
        
        # Load
        loaded_store = FAISSVectorStore.load(save_path)
        
        assert loaded_store.index.ntotal == 5
        assert len(loaded_store.metadata) == 5
        assert loaded_store.dim == 384

    def test_load_nonexistent(self, temp_dir):
        """Loading nonexistent index should raise error."""
        with pytest.raises(RuntimeError):
            FAISSVectorStore.load(temp_dir / "nonexistent")

    def test_search_scores_ordered(self):
        """Search results should be ordered by score (descending)."""
        store = FAISSVectorStore(dim=384)
        
        embeddings = np.random.rand(10, 384).astype("float32")
        metadata = [{"text": f"doc{i}"} for i in range(10)]
        store.add(embeddings, metadata)
        
        query = np.random.rand(1, 384).astype("float32")
        results = store.search(query, k=10)
        
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
