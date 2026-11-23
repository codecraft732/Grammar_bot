from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


class GrammarRetriever:
    """Retrieves top-k relevant grammar book chunks from Chroma."""

    def __init__(
        self,
        vectorstore: Chroma,
        k: int = 4,
        search_type: str = "similarity",
    ) -> None:
        self.vectorstore = vectorstore
        self.k = k
        self.search_type = search_type
        self.retriever: VectorStoreRetriever = self.vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs={"k": self.k},
        )

    def get_retriever(self) -> VectorStoreRetriever:
        return self.retriever

    def retrieve(self, query: str) -> List[Document]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        return self.retriever.invoke(query)


if __name__ == "__main__":
    from config import (
        COLLECTION_NAME,
        EMBEDDING_MODEL_NAME,
        TOP_K,
        VECTOR_STORE_DIR,
    )
    from src.vector_store import VectorStoreManager

    # Saved index load karo (dobara ingest nahi!)
    manager = VectorStoreManager(
        model_name=EMBEDDING_MODEL_NAME,
        persist_directory=str(VECTOR_STORE_DIR),
        collection_name=COLLECTION_NAME,
    )
    store = manager.load_vectorstore()

    grammar_retriever = GrammarRetriever(
        vectorstore=store,
        k=TOP_K,
        search_type="similarity",
    )

    test_query = "Present Simple"
    results = grammar_retriever.retrieve(test_query)

    print(f"Query: {test_query} | Retrieved Chunks: {len(results)}\n")
    for idx, doc in enumerate(results, start=1):
        page_num = doc.metadata.get("page", "N/A")
        print(f"--- Chunk {idx} (Page {page_num}) ---")
        print(doc.page_content[:250])
        print()
