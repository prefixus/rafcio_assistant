"""
Unit tests for the HybridRetriever.
"""

from unittest.mock import MagicMock

from src.retrieval import HybridRetriever
from tests.mock_vectorstore import MockVectorStoreAdapter


def test_rrf_logic():
    """Test the Reciprocal Rank Fusion implementation."""
    # Use a dummy adapter
    adapter = MagicMock()
    retriever = HybridRetriever(vector_adapter=adapter, rrf_k=60)

    # Setup sample results
    vector_res = [
        {"text": "doc1", "metadata": {}},
        {"text": "doc2", "metadata": {}},
    ]
    bm25_res = [
        {"text": "doc2", "metadata": {}},
        {"text": "doc3", "metadata": {}},
    ]

    # RRF scores calculations:
    # doc1: 1/(1+60) = 0.01639
    # doc2: 1/(2+60) + 1/(1+60) = 0.01612 + 0.01639 = 0.03251
    # doc3: 1/(2+60) = 0.01612

    results = retriever._reciprocal_rank_fusion(  # pylint: disable=protected-access
        vector_res, bm25_res, k=3
    )

    assert results[0]["text"] == "doc2"
    assert results[1]["text"] == "doc1"
    assert results[2]["text"] == "doc3"


def test_hybrid_search_flow():
    """Test the full search flow using a Mock adapter."""
    adapter = MockVectorStoreAdapter()
    docs = [
        {"text": "Python is dynamic", "metadata": {"id": 1}},
        {"text": "C++ is fast", "metadata": {"id": 2}},
        {"text": "Rust is safe", "metadata": {"id": 3}},
    ]
    adapter.collection = docs
    # Mock similarity_search to return something consistent
    adapter.similarity_search = MagicMock(return_value=[docs[0]])

    retriever = HybridRetriever(vector_adapter=adapter, top_k=2)

    # "is" appears in all 3, but "dynamic" only in doc1
    # BM25 should rank doc1 highly for "dynamic"
    results = retriever.search("dynamic")

    assert len(results) >= 1
    assert results[0]["text"] == "Python is dynamic"


def test_empty_retrieval():
    """Test behavior with empty document store."""
    adapter = MockVectorStoreAdapter()
    adapter.collection = []
    adapter.similarity_search = MagicMock(return_value=[])

    retriever = HybridRetriever(vector_adapter=adapter)
    results = retriever.search("anything")

    assert results == []
