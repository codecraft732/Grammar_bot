from pathlib import Path
import shutil
from typing import List
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreManager:
    """Creates embeddings and saves/loads a persistent local Chroma vector store."""

    def __init__(
        self,
        model_name: str = "hotchpotch/bekko-embedding-v1-a8m",
        persist_directory: str = "data/vector_store",
        collection_name: str = "english_grammar_book",
    ) -> None:
        self.model_name = model_name
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"model_kwargs": {"dtype": torch.float32}},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _store_exists(self) -> bool:
        return (self.persist_directory / "chroma.sqlite3").exists()

    def build_and_save(self, chunks: List[Document], overwrite: bool = False) -> Chroma:
        if not chunks:
            raise ValueError("Chunks list is empty. Run TextChunker first.")

        if self._store_exists() and not overwrite:
            print(
                "Vector store already exists. Loading existing index (no duplicates)."
            )
            return self.load_vectorstore()

        if overwrite and self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory),
            collection_name=self.collection_name,
        )

    def load_vectorstore(self) -> Chroma:
        if not self._store_exists():
            raise FileNotFoundError(
                f"No Chroma index found in '{self.persist_directory}'. Build it first."
            )
        return Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )


if __name__ == "__main__":
    # Quick local test with 2 sample chunks (full book baad mein ingest karenge)
    sample_chunks = [
        Document(
            page_content="Present Simple Rule: Add 's' or 'es' to the verb for he, she, it.",
            metadata={"page": 10, "source": "Png.pdf"},
        ),
        Document(
            page_content="Past Simple Rule: Use the second form of the verb for completed actions.",
            metadata={"page": 25, "source": "Png.pdf"},
        ),
    ]

    manager = VectorStoreManager(persist_directory="data/vector_store_test")
    store = manager.build_and_save(sample_chunks, overwrite=True)
    print(f"Total vectors stored: {store._collection.count()}")

    loaded_store = manager.load_vectorstore()
    results = loaded_store.similarity_search("When do we add s to a verb?", k=1)
    print(f"Top Match Page: {results[0].metadata['page']}")
    print(f"Top Match Text: {results[0].page_content}")
