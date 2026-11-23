import re
import pymupdf.layout
import pymupdf4llm
from langchain_core.documents import Document


class BookLoader:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    # [~] CHANGE: 'self' add karein aur neeche wale function ki 3 regex lines yahan merge karein
    def clean_markdown(self, text: str) -> str:
        text = re.sub(r"</?mark>", "", text)  
        text = re.sub(r"<!--\s*Start of picture text\s*-->", "", text)
        text = re.sub(r"<!--\s*End of picture text\s*-->", "", text)
        text = re.sub(r"<br\s*/?>", "\n", text)
        text = re.sub(r"^FURTHER EXAMPLES$", "", text, flags=re.MULTILINE)  
        text = re.sub(r"^\d{1,3}$", "", text, flags=re.MULTILINE)  
        text = re.sub(r"[ \t]+", " ", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    # [~] CHANGE: Function naam 'load' karein aur 'self' lagayein
    def load(self, test_pages=None) -> list[Document]:
        md_pages = pymupdf4llm.to_markdown(
            self.pdf_path,  # [~] CHANGE: pdf_path -> self.pdf_path
            pages=test_pages,
            page_chunks=True,
        )

        documents = []
        for p in md_pages:
            cleaned_text = self.clean_markdown(
                p["text"]
            )  # [~] CHANGE: self.clean_markdown
            if not cleaned_text:
                continue

            doc = Document(
                page_content=cleaned_text,
                metadata={
                    "source": self.pdf_path,  # [~] CHANGE: direct self.pdf_path
                    "page": p["metadata"].get(
                        "page_number"
                    ),  # [~] CHANGE: key 'page_number' -> 'page'
                    "title": p["metadata"].get("title", "English Grammar Guide"),
                },
            )
            documents.append(doc)

        return documents


if __name__ == "__main__":
    pdf_file = r"D:\AndroidProjects\Grammar_bot\data\book\png.pdf"

    loader = BookLoader(pdf_file)
    docs = loader.load()

    for d in docs:
        print(
            f"=== PAGE {d.metadata['page']} ==="
        )  # [~] CHANGE: 'page_number' -> 'page'
        print(d.page_content)
        print("\n" + "=" * 40 + "\n")


