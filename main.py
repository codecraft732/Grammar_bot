from config import (
    BOOK_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_DIR,
)
from src.book_loader import BookLoader
from src.text_chunker import TextChunker
from src.vector_store import VectorStoreManager


def run_ingestion(test_pages=None, overwrite: bool = False) -> None:
    # 1. Load Book Pages
    loader = BookLoader(str(BOOK_PATH))
    pages = loader.load(test_pages=test_pages)
    print(f"Loaded non-empty pages: {len(pages)}")

    # 2. Split into Overlapping Chunks
    chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = chunker.split_documents(pages)
    print(f"Total chunks created: {len(chunks)}")

    # 3. Embed & Save in Local Chroma Vector Store
    manager = VectorStoreManager(
        model_name=EMBEDDING_MODEL_NAME,
        persist_directory=str(VECTOR_STORE_DIR),
        collection_name=COLLECTION_NAME,
    )
    store = manager.build_and_save(chunks, overwrite=overwrite)
    print(f"Total vectors saved in '{VECTOR_STORE_DIR}': {store._collection.count()}")


if __name__ == "__main__":
    # Option 2: Pehle 30 pages quick test (baad mein test_pages=None se poori 362 pages)
    run_ingestion(test_pages=list(range(30)), overwrite=True)
