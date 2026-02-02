"""
Standard LangChain Expression Language (LCEL) chain for basic RAG.
Connects the Retriever, Prompt Template, and LLM.
"""

from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from src.config import settings
from src.retrieval import HybridRetriever


def format_docs(docs: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into a single string."""
    formatted = []
    for doc in docs:
        text = doc.get("text", "")
        # metadata extraction can be more robust if needed
        source = doc.get("metadata", {}).get("source", "unknown")
        formatted.append(f"Content: {text}\nSource: {source}\n---")
    return "\n".join(formatted)


# pylint: disable=too-few-public-methods
class RAGChain:
    """
    Standard RAG Chain implemented using LCEL.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        model_name: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        self.retriever = retriever
        self.model_name = model_name or settings.llm_model
        self.api_url = (api_url or settings.api_url).rstrip("/")
        if not self.api_url.endswith("/v1"):
            self.api_url = f"{self.api_url}/v1"

        # Initialize LLM
        # LM Studio is OpenAI-compatible
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url=self.api_url,
            # LM Studio doesn't require an API key by default
            api_key="not-needed",  # type: ignore[arg-type]
            temperature=0.1,
        )

        # Define Prompt Template
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know,
            don't try to make up an answer.

            Context:
            {context}

            Question: {question}

            Answer:"""
        )

        # Build Chain
        self.chain: Runnable = (
            {
                "context": lambda x: format_docs(self.retriever.search(x["question"])),
                "question": lambda x: x["question"],
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, question: str) -> str:
        """Invoke the RAG chain."""
        return self.chain.invoke({"question": question})
