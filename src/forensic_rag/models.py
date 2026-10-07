"""LLM (Claude via OpenRouter) and embeddings (Voyage AI) factories."""

from functools import lru_cache

import voyageai
from langchain_openai import ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings

from forensic_rag import config


@lru_cache(maxsize=None)
def get_llm(temperature: float | None = None) -> ChatOpenAI:
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set (forensic-rag-mcp/.env)")
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        max_retries=3,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> VoyageAIEmbeddings:
    if not config.VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is not set (forensic-rag-mcp/.env)")
    embeddings = VoyageAIEmbeddings(
        model=config.EMBED_MODEL,
        voyage_api_key=config.VOYAGE_API_KEY,
        batch_size=64,
    )
    # the wrapper builds its client with max_retries=0; retry transient overloads / rate limits
    embeddings._client = voyageai.Client(api_key=config.VOYAGE_API_KEY, max_retries=6)
    return embeddings
