from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """Splits page-wise Document objects into overlapping chunks while preserving metadata."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            raise ValueError("Documents list is empty. Load the book before chunking.")
        return self.splitter.split_documents(documents)


if __name__ == "__main__":
    sample_doc = Document(
        page_content=("Present Simple Rule: Add 's' or 'es' with he/she/it. " * 30),
        metadata={"page": 1, "source": "grammar_book.pdf"},
    )
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split_documents([sample_doc])

    print(f"Total Chunks: {len(chunks)}")
    print(f"Chunk 1 Length: {len(chunks[0].page_content)}")
    print(f"Chunk 1 Metadata: {chunks[0].metadata}")