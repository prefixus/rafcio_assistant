"""
Configuration management for the RAG Assistant using Pydantic Settings.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # Model Configuration (mapped to current .env)
    api_url: str = Field(default="http://127.0.0.1:1234", alias="API_URL")
    llm_model: str = Field(default="openai/gpt-oss-20b", alias="LLM_MODEL")
    secondary_llm_model: Optional[str] = Field(
        default="zai-org/glm-4.7-flash", alias="SECONDARY_LLM_MODEL"
    )
    embedding_model: str = Field(default="text-embedding-bge-m3", alias="EMB_MODEL")

    # Vector Store Configuration
    vector_store_provider: str = Field(default="chroma", alias="VECTOR_STORE_PROVIDER")
    vector_store_url: Optional[str] = Field(default=None, alias="VECTOR_STORE_URL")
    vector_store_collection: str = Field(
        default="rafcio_assistant", alias="VECTOR_STORE_COLLECTION"
    )

    # Feature Flags (from technical specification)
    enable_hybrid_search: bool = Field(default=True, alias="ENABLE_HYBRID_SEARCH")
    enable_mcp_tools: bool = Field(default=True, alias="ENABLE_MCP_TOOLS")
    enable_conversation_memory: bool = Field(
        default=True, alias="ENABLE_CONVERSATION_MEMORY"
    )
    enable_deepeval: bool = Field(default=False, alias="ENABLE_DEEPEVAL")

    # Retrieval Configuration (from technical specification)
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=128, alias="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, alias="TOP_K_RESULTS")

    # Path Configuration
    knowledge_base_dir: str = "./knowledge_base"
    workspace_dir: str = "./workspace"
    chroma_db_dir: str = "./data/chroma_db"
    conversation_db_path: str = "./data/conversations.db"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# Global settings instance
settings = Settings()
