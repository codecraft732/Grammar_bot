from pathlib import Path
from typing import List

import streamlit as st

from config import (
    BOOK_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    TOP_K,
    VECTOR_STORE_DIR,
)
from src.book_loader import BookLoader
from src.rag_chain import RAGChain
from src.retriever import GrammarRetriever
from src.text_chunker import TextChunker
from src.vector_store import VectorStoreManager


@st.cache_resource(show_spinner="Loading embedding model...")
def get_vector_store_manager() -> VectorStoreManager:
    """Loads the embedding model once and caches it across Streamlit reruns."""
    return VectorStoreManager(
        model_name=EMBEDDING_MODEL_NAME,
        persist_directory=str(VECTOR_STORE_DIR),
        collection_name=COLLECTION_NAME,
    )


class ChatApp:
    """Streamlit UI layer for English Grammar Book Bot."""

    def __init__(self) -> None:
        self.manager = get_vector_store_manager()
        self._init_session_state()

    def _init_session_state(self) -> None:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "active_book_path" not in st.session_state:
            st.session_state.active_book_path = str(BOOK_PATH)
        if "allow_fallback" not in st.session_state:
            st.session_state.allow_fallback = False
        if "rag_chain" not in st.session_state:
            st.session_state.rag_chain = self._try_build_rag_chain()

    def _try_build_rag_chain(self) -> RAGChain | None:
        if not self.manager._store_exists():
            return None
        store = self.manager.load_vectorstore()
        retriever = GrammarRetriever(
            vectorstore=store,
            k=TOP_K,
            search_type="similarity",
        )
        return RAGChain(
            retriever=retriever,
            model_name=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
        )

    def _index_full_book(self, pdf_path: str) -> int:
        loader = BookLoader(pdf_path)
        pages = loader.load()

        chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = chunker.split_documents(pages)

        if self.manager._store_exists():
            store = self.manager.load_vectorstore()
            existing_ids: List[str] = store.get().get("ids", [])
            # Batch delete avoids SQLite parameter limit & Windows file lock
            for i in range(0, len(existing_ids), 500):
                store.delete(ids=existing_ids[i : i + 500])
            for i in range(0, len(chunks), 100):
                store.add_documents(chunks[i : i + 100])
        else:
            store = self.manager.build_and_save(chunks, overwrite=False)

        st.session_state.rag_chain = self._try_build_rag_chain()
        st.session_state.messages = []
        return store._collection.count()

    def _render_sidebar(self) -> None:
        with st.sidebar:
            st.header("Book & Index Control")
            st.caption(f"Model: `{LLM_MODEL_NAME}`")

            # Upload new grammar book (saved locally in data/book/)
            uploaded_file = st.file_uploader(
                "Upload Grammar Book (PDF)",
                type=["pdf"],
                help="Stored locally in data/book/. Never sent to external storage.",
            )
            if uploaded_file is not None:
                save_path = Path(BOOK_PATH).parent / uploaded_file.name
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(uploaded_file.getbuffer())
                st.session_state.active_book_path = str(save_path)
                st.success(f"Saved locally: `{uploaded_file.name}`")

            active_name = Path(st.session_state.active_book_path).name
            st.info(f"Active Book: **{active_name}**")

            if self.manager._store_exists():
                count = self.manager.load_vectorstore()._collection.count()
                st.success(f"Index Ready: **{count} chunks**")
            else:
                st.warning("No vector index found. Click Index Book below.")

            if st.button(
                "Index / Re-index Full Book", type="primary", use_container_width=True
            ):
                try:
                    with st.spinner(
                        "Indexing full grammar book (may take 1–3 mins)..."
                    ):
                        total = self._index_full_book(st.session_state.active_book_path)
                    st.success(f"Full book indexed! Total chunks: {total}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

            st.divider()
            st.session_state.allow_fallback = st.toggle(
                "Allow General AI Fallback",
                value=st.session_state.allow_fallback,
                help="OFF = Strict Book-Only Mode (Recommended). ON = Uses General AI if not in the book.",
            )
            if st.session_state.allow_fallback:
                st.caption("Mode: Book + General AI Fallback (labeled)")
            else:
                st.caption("Mode: Strict Book-Only (Zero Hallucination)")

            st.divider()
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.rag_chain is not None:
                    st.session_state.rag_chain.clear_memory()
                st.rerun()

    def _render_sources(self, sources: list) -> None:
        if not sources:
            return
        pages = sorted({doc.metadata.get("page", "N/A") for doc in sources})
        st.caption(f"Cited Book Pages: {', '.join(str(p) for p in pages)}")
        with st.expander("View Book Sources & Page Excerpts"):
            for idx, doc in enumerate(sources, start=1):
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Source {idx} — Page {page}**")
                st.text(doc.page_content[:500])
                if idx < len(sources):
                    st.divider()

    def _render_chat(self) -> None:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    self._render_sources(msg["sources"])

        user_query = st.chat_input(
            "Ask any grammar topic (e.g., Present Simple, Conditionals)..."
        )
        if not user_query:
            return

        if st.session_state.rag_chain is None:
            st.error("Please index the grammar book from the sidebar first.")
            return

        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your grammar book..."):
                    result = st.session_state.rag_chain.ask(
                        user_query,
                        allow_fallback=st.session_state.allow_fallback,
                    )
                answer = result["answer"]
                sources = result["sources"]
                st.markdown(answer)
                self._render_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                st.error(
                    f"Could not generate answer (check internet/Groq limit): {exc}"
                )

    def run(self) -> None:
        st.set_page_config(
            page_title="English Grammar Book Bot",
            page_icon="📘",
            layout="centered",
        )
        st.title("English Grammar Book Bot")
        st.caption(
            "Strict RAG Chatbot — Answers from your grammar book with page citations (optional fallback in sidebar)."
        )
        self._render_sidebar()
        self._render_chat()
