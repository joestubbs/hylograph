import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache


# Defaults
DEFAULT_MODEL_URL = "https://ollama.pods.tacc.develop.tapis.io"
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
DEFAULT_LLM = "llama3.1:8b"
SQLITE_DB_LLM_CACHE_PATH = os.environ.get("SQLITE_DB_LLM_CACHE_PATH", "./hylo-LLM-cache.db")


def get_embedding_model(model=DEFAULT_EMBEDDING_MODEL, 
                        base_url=DEFAULT_MODEL_URL):
    """
    Returns the embedding model used for this graph.
    """
    return OllamaEmbeddings(model=model, base_url=base_url)


def get_llm(model=DEFAULT_LLM, 
            base_url=DEFAULT_MODEL_URL, 
            cache_responses=True,
            database_path=SQLITE_DB_LLM_CACHE_PATH):
    """
    Returns the LLM for this graph.
    """
    if cache_responses:
        set_llm_cache(SQLiteCache(database_path=database_path))
    return ChatOllama(model=model, base_url=base_url)


