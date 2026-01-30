"""
Functional tests for retrieval ensuring integration between ChromaDB and HybridRetriever.
"""

import os
import shutil
import tempfile

import pytest

from src.embeddings import LMStudioEmbeddings
from src.retrieval import HybridRetriever
from src.vectorstore import VectorStoreManager

# Reuse the skip marker for local services
REQUIRES_LM_STUDIO = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping local service test in GitHub Actions",
)


@REQUIRES_LM_STUDIO
def test_functional_retrieval_integration():
    """Verify that retrieval works end-to-end with a local ChromaDB instance."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize components
        embeddings = LMStudioEmbeddings()
        manager = VectorStoreManager(
            provider="chroma",
            persist_directory=temp_dir,
            collection_name="functional_test_retrieval",
            embedding_model=embeddings,
        )
        adapter = manager.get_adapter()

        # Add sample data
        texts = [
            "Rafcio Assistant is a tool for developers.",
            "It supports hybrid search combining vector and keywords.",
            "DeepEval is used for quality assessment.",
        ]
        metadatas = [{"source": "manual"}, {"source": "specs"}, {"source": "tests"}]
        adapter.add_documents(texts, metadatas)

        # Initialize Retriever
        retriever = HybridRetriever(vector_adapter=adapter, top_k=2)

        # 1. Test semantic match (Vector should shine)
        # "AI agent" isn't in texts, but "Rafcio Assistant is a tool" is semantically close
        results_semantic = retriever.search("AI assistant tool")
        assert len(results_semantic) > 0
        assert "Rafcio Assistant" in results_semantic[0]["text"]

        # 2. Test keyword match (BM25 should shine)
        results_keyword = retriever.search("DeepEval assessment")
        assert len(results_keyword) > 0
        assert "DeepEval" in results_keyword[0]["text"]

        # 3. Test RRF merging
        results_combined = retriever.search("hybrid search tool")
        # Should find both the tool doc and the hybrid search doc
        assert len(results_combined) == 2

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
