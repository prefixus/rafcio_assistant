"""
Foundation test to verify the core RAG flow:
1. PDF loading
2. Text chunking
3. Embedding generation via LM Studio
4. Storage and retrieval via ChromaDB
"""

import os
import shutil

import pytest
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.embeddings import LMStudioEmbeddings
from src.vectorstore import VectorStoreManager

# NOTE: This test requires 'pypdf' to be installed for PDF loading.
# It also requires LM Studio to be running for embeddings.

REQUIRES_LOCAL_SERVICES = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping foundation test in GitHub Actions",
)


@REQUIRES_LOCAL_SERVICES
def test_foundation_embeddings_to_chroma():
    """
    Verifies that we can process a real PDF, embed it, and query it.
    """
    pdf_path = os.path.join(
        settings.knowledge_base_dir, "Introduction to Agents from Google.pdf"
    )

    # 1. Verify file exists
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"

    # 2. Load PDF (Missing functionality: PDF Loader library)
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_content = "\n".join([doc.page_content for doc in docs])
    except (ImportError, ModuleNotFoundError):
        pytest.fail(
            "Missing dependency: 'pypdf' and 'langchain-community' are required for PDF loading."
        )

    # 3. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = text_splitter.split_text(text_content)
    assert len(chunks) > 0, "No chunks generated from PDF"

    # 4. Embeddings (requires LM Studio)
    embeddings_model = LMStudioEmbeddings()
    try:
        sample_vec = embeddings_model.embed_query("test query")
        assert len(sample_vec) > 0
    except Exception as e:  # pylint: disable=broad-exception-caught
        pytest.skip(f"LM Studio not available: {e}")

    # 5. Vector Store (Missing functionality: passing custom embeddings to adapter)
    manager = VectorStoreManager(
        provider="chroma",
        persist_directory="./data/test_foundation_db",
        collection_name="foundation_test",
    )
    adapter = manager.get_adapter()

    # Note: Current ChromaVectorStoreAdapter uses default Chroma embeddings.
    # It should be updated to accept our LMStudioEmbeddings.

    try:
        adapter.add_documents(chunks[:5])  # Add first 5 chunks for speed

        # 6. Retrieval
        query = "What are agents?"
        results = adapter.similarity_search(query, k=2)

        assert len(results) > 0
        print(f"\n✓ Found {len(results)} relevant chunks in ChromaDB")
        for i, res in enumerate(results):
            print(f"Result {i + 1} snippet: {res['text'][:100]}...")

    finally:
        # Cleanup
        adapter.delete_collection()

        if os.path.exists("./data/test_foundation_db"):
            shutil.rmtree("./data/test_foundation_db")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
