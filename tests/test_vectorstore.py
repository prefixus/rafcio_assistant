"""
Tests for the VectorStoreManager and adapters.
"""

import pytest
from src.vectorstore import VectorStoreManager
from tests.mock_vectorstore import MockVectorStoreAdapter


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
