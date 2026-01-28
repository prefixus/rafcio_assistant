from src.config import settings


def test_settings_loading():
    print("--- Settings Validation ---")
    print(f"API URL: {settings.api_url}")
    print(f"LLM Model: {settings.llm_model}")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"Hybrid Search Enabled: {settings.enable_hybrid_search}")
    print(f"Chunk Size: {settings.chunk_size}")
    print("---------------------------")

    # Assertions to ensure .env is actually read (assuming they match .env values)
    assert "http" in settings.api_url
    print("✓ Validation successful: Settings loaded correctly.")


if __name__ == "__main__":
    test_settings_loading()
