import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.retriever import GrammarRetriever


SYSTEM_PROMPT = """You are 'English Grammar Book Bot', a strict grammar assistant.
Answer the user's question using ONLY the provided Book Context below.

STRICT RULES:
1. Do NOT use any outside knowledge. Use ONLY facts present in the Book Context.
2. If the answer is NOT in the Book Context, reply with ONLY this exact sentence:
Not found in the book.
3. For grammar topic questions found in the Book Context, ALWAYS use this exact structure:

### Rule
(Exact rule from the book)

### Tips & Tricks
(Tips/notes from the book, or write: Not mentioned in the retrieved book pages.)

###  Examples 
(Correct examples from the book, or write: Not mentioned in the retrieved book pages.)

### Common Mistakes
(Common mistakes from the book, or write: Not mentioned in the retrieved book pages.)

Book Context:
{context}

Chat History:
{chat_history}
"""


class RAGChain:
    """Grounded LCEL Conversation Retrieval Chain with strict format, memory, and citations."""

    def __init__(
        self,
        retriever: GrammarRetriever,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_history_turns: int = 5,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY missing in .env file.")

        self.retriever = retriever
        self.max_history_turns = max_history_turns
        self.chat_history: List[Dict[str, str]] = []

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            reasoning_format="parsed",
            reasoning_effort="low",
            max_tokens=2048,
        )

        # Main grounded answer chain (LCEL: Prompt | LLM | StrOutputParser)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

        # Follow-up question rewriter (for FR9 Conversation Memory)
        self.rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Given the chat history and a follow-up question, rewrite the follow-up "
                    "question into a clear standalone question. Return ONLY the rewritten question.",
                ),
                (
                    "human",
                    "Chat History:\n{chat_history}\n\nFollow-up Question: {question}",
                ),
            ]
        )
        self.rewrite_chain = self.rewrite_prompt | self.llm | StrOutputParser()

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            page = doc.metadata.get("page", "N/A")
            formatted.append(f"[Page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def _format_history(self) -> str:
        if not self.chat_history:
            return "No previous conversation."
        recent = self.chat_history[-self.max_history_turns :]
        lines = []
        for turn in recent:
            lines.append(f"Human: {turn['human']}\nAI: {turn['ai']}")
        return "\n\n".join(lines)

    def _get_search_query(self, question: str) -> str:
        if not self.chat_history:
            return question
        return self.rewrite_chain.invoke(
            {"chat_history": self._format_history(), "question": question}
        ).strip()

    def ask(self, question: str) -> Dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # 1. Rewrite follow-up question if history exists, then retrieve top-k chunks
        search_query = self._get_search_query(question)
        sources = self.retriever.retrieve(search_query)

        # 2. Generate strict grounded answer via LCEL chain
        context = self._format_docs(sources)
        history_text = self._format_history()
        answer = self.chain.invoke(
            {
                "context": context,
                "chat_history": history_text,
                "question": question,
            }
        ).strip()
      
        if not answer:
            answer = "Not found in the book."

        # 3. Save turn to conversation memory (Human/AI roles)
        self.chat_history.append({"human": question, "ai": answer})

        return {"answer": answer, "sources": sources}

    def clear_memory(self) -> None:
        self.chat_history = []


if __name__ == "__main__":
    from config import (
        COLLECTION_NAME,
        EMBEDDING_MODEL_NAME,
        LLM_MODEL_NAME,
        LLM_TEMPERATURE,
        TOP_K,
        VECTOR_STORE_DIR,
    )
    from src.vector_store import VectorStoreManager

    manager = VectorStoreManager(
        model_name=EMBEDDING_MODEL_NAME,
        persist_directory=str(VECTOR_STORE_DIR),
        collection_name=COLLECTION_NAME,
    )
    store = manager.load_vectorstore()
    retriever = GrammarRetriever(vectorstore=store, k=TOP_K, search_type="similarity")
    rag = RAGChain(
        retriever=retriever,
        model_name=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
    )

    # Test 1: In-book Grammar Topic (FR5 + FR6 + FR7)
    res1 = rag.ask("Explain Present Simple")
    pages1 = sorted({d.metadata.get("page", "N/A") for d in res1["sources"]})
    print("=== TEST 1: IN-BOOK TOPIC ===")
    print(res1["answer"])
    print(f"Cited Pages: {pages1}\n")

    # Test 2: Follow-up Question using Memory (FR9)
    res2 = rag.ask("Give me examples of its negative form")
    pages2 = sorted({d.metadata.get("page", "N/A") for d in res2["sources"]})
    print("=== TEST 2: FOLLOW-UP MEMORY ===")
    print(res2["answer"])
    print(f"Cited Pages: {pages2}\n")

    # Test 3: Out-of-book Hallucination Check (FR5)
    res3 = rag.ask("Who won the 2022 FIFA World Cup?")
    print("=== TEST 3: OUT-OF-BOOK GUARDRAIL ===")
    print(res3["answer"])
