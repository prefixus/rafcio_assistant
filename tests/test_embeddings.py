import os

import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.embeddings import LMStudioEmbeddings

# Marker for tests that require a local model server (LM Studio)
REQUIRES_LM_STUDIO = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping test that requires local LM Studio in GitHub Actions",
)


@REQUIRES_LM_STUDIO
def test_embeddings_connectivity():
    """
    Test if we can connect to the embedding server and get a response.
    Requires LM Studio to be running with an embedding model loaded.
    """
    embeddings = LMStudioEmbeddings()
    try:
        vector = embeddings.embed_query("This is a connectivity test.")
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert isinstance(vector[0], float)
        print(f"\n✓ Successfully generated embedding of size {len(vector)}")
    except RuntimeError as e:
        pytest.skip(f"Skipping connectivity test: {e}. Ensure LM Studio is running.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        pytest.fail(f"Unexpected error during connectivity test: {e}")


@REQUIRES_LM_STUDIO
def test_embeddings_with_chunking():
    """
    Test embedding generation for chunked documents from a file.
    Verifies that the integration with LangChain's text splitters works.
    """
    embeddings = LMStudioEmbeddings()

    # Path to test file
    test_file = os.path.join(settings.knowledge_base_dir, "test_sample.txt")
    if not os.path.exists(test_file):
        pytest.fail(f"Test file not found at {test_file}")

    with open(test_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split content using settings from config
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = text_splitter.split_text(content)

    assert len(chunks) > 0
    print(f"\n✓ Split document into {len(chunks)} chunks")

    # Embed chunks
    try:
        vectors = embeddings.embed_documents(chunks)
        assert len(vectors) == len(chunks)
        for vec in vectors:
            assert len(vec) == len(vectors[0])  # All vectors should have same dimension
        print(f"✓ Successfully embedded {len(vectors)} chunks")
    except RuntimeError as e:
        pytest.skip(f"Skipping embedding test: {e}. Ensure LM Studio is running.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        pytest.fail(f"Failed to embed chunks: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])  # -s to see print statements
