from pathlib import Path


# Project Paths (Single Source of Truth)
BASE_DIR = Path(__file__).resolve().parent
BOOK_PATH = BASE_DIR / "data" / "book" / "Png.pdf"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

# Chunking Settings (bekko max_tokens=8192 -> 1000 chars safely inside limit)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding & Vector Store Settings
EMBEDDING_MODEL_NAME = "hotchpotch/bekko-embedding-v1-a8m"
COLLECTION_NAME = "english_grammar_book"

# Retriever Settings (Phase 5 mein use hoga)
TOP_K = 6


# LLM Settings (Groq)
LLM_MODEL_NAME = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0.0