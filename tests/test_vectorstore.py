import os
import shutil
import tempfile

import pytest

from src.vectorstore import ChromaVectorStoreAdapter, VectorStoreManager
from tests.mock_vectorstore import MockVectorStoreAdapter

# Marker for tests that require a local environment or can be slow
REQUIRES_CHROMA = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping Chroma test in GitHub Actions",
)


def test_vector_store_manager_mock():
    """Test that VectorStoreManager correctly returns a mock adapter."""
    manager = VectorStoreManager(provider="mock")
    adapter = manager.get_adapter()

    assert isinstance(adapter, MockVectorStoreAdapter)
    # Check something to verify it is the mock adapter
    assert adapter.provider == "mock"


def test_mock_adapter_operations():
    """Test basic operations on the mock adapter."""
    manager = VectorStoreManager(provider="mock")
    adapter = manager.get_adapter()

    texts = ["Hello world", "RAG is awesome"]
    metadatas = [{"source": "test1"}, {"source": "test2"}]

    adapter.add_documents(texts, metadatas)

    results = adapter.similarity_search("RAG", k=2)
    assert len(results) == 2
    assert results[0]["text"] == "Hello world"
    assert results[1]["metadata"]["source"] == "test2"

    adapter.delete_collection()
    assert len(adapter.similarity_search("any", k=1)) == 0


def test_invalid_provider():
    """Test that an invalid provider raises a ValueError."""
    manager = VectorStoreManager(provider="invalid")
    with pytest.raises(ValueError, match="Unsupported vector store provider"):
        manager.get_adapter()


@REQUIRES_CHROMA
def test_chroma_adapter_operations():
    """Test basic operations on the Chroma adapter using a temp directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        manager = VectorStoreManager(
            provider="chroma",
            persist_directory=temp_dir,
            collection_name="test_collection",
        )
        adapter = manager.get_adapter()
        assert isinstance(adapter, ChromaVectorStoreAdapter)

        texts = ["Python is a programming language", "ChromaDB is a vector store"]
        metadatas = [{"type": "code"}, {"type": "db"}]

        adapter.add_documents(texts, metadatas)

        results = adapter.similarity_search("What is ChromaDB?", k=1)
        assert len(results) == 1
        # Chroma might return them in different order depending on embedding,
        # but let's check one specifically if it matches
        assert "ChromaDB" in results[0]["text"]

        adapter.delete_collection()
        # After delete_collection in our implementation, it recreates it empty
        assert len(adapter.similarity_search("any", k=1)) == 0
    finally:
        shutil.rmtree(temp_dir)
