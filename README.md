# 📘 English Grammar Book Bot (RAG + LangChain + Streamlit)

An end-to-end, modular **Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain**, **Chroma**, **HuggingFace Embeddings**, **Groq LLM**, and **Streamlit**.  
It answers English grammar questions **strictly from a single local grammar book**—providing structured grammar explanations, page-number citations, conversation memory, zero outside-knowledge hallucination (with an optional labeled General AI Fallback toggle in the sidebar).

---

## 🎯 Problem Statement
Learners studying from large English grammar reference books struggle to quickly locate exact **rules, tips & tricks, correct/incorrect examples, and common mistakes** for a specific topic without flipping through hundreds of pages.

**Solution:** A strictly grounded RAG pipeline that indexes the grammar book once into a local persistent vector store and answers every query exclusively from retrieved book pages. If a topic is not present in the book (default Strict Mode), the bot responds:
> `Not found in the book.`

---

## ✨ Key Features (FR1 – FR10)
- **Page-Wise Book Ingestion (FR1):** Extracts text page by page while preserving page numbers and source metadata.
- **Overlapping Text Chunking (FR2):** Splits pages with `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`) without breaking sentence meaning or losing metadata.
- **Local Persistent Vector Store (FR3):** Generates embeddings locally using HuggingFace (`hotchpotch/bekko-embedding-v1-a8m`) and persists them to local **Chroma** (index once, reuse across sessions without duplicate vectors).
- **Semantic Top-k Retrieval (FR4):** Retrieves the top `k=6` most relevant chunks via similarity search.
- **Strict Grounded Generation + Optional Fallback Toggle (FR5):** Uses an LCEL chain (`ChatPromptTemplate | ChatGroq | StrOutputParser`) with `temperature=0.0` and an anti-hallucination guardrail (`"Not found in the book."`). Includes an optional sidebar toggle (**OFF by default**) that, when enabled, provides a clearly labeled `⚠️ Not found in the book — General AI Answer:` response without misleading book citations.
- **Fixed Structured Grammar Output (FR6):** Every in-book topic answer follows a clean 5-part format:
  1. **Rule**
  2. **Tips & Tricks**
  3. **Correct Examples (Right)**
  4. **Incorrect Examples (Wrong)**
  5. **Common Mistakes**
- **Page-Number Citations & Source Viewer (FR7):** Displays exact cited book page numbers (inline `(Page X)` + citation footer) and expandable raw book excerpts for every book-grounded answer.
- **Streamlit Chat UI (FR8):** Clean chat interface with message history, sidebar PDF upload, one-click full book indexing/re-indexing, Strict/Fallback mode toggle, clear chat button, and expandable source viewer.
- **Conversation Memory & Query Rewriting (FR9):** Tracks recent chat turns (`Human` / `AI` roles) and rewrites follow-up queries into standalone search queries for accurate multi-turn retrieval.
- **Human-Checked Evaluation Suite (FR10):** Includes `eval/test_questions.json` and `eval/eval.py` covering in-book topics, follow-up memory, and out-of-book hallucination tests.

---

## 🏗️ System Architecture

### 1. Ingestion Pipeline (Offline — Run Once)
```text
Grammar Book (PDF)
   │
   ▼
BookLoader (src/book_loader.py)          ──► Page-wise Documents + Page Metadata
   │
   ▼
TextChunker (src/text_chunker.py)        ──► Overlapping Chunks (1000 chars / 200 overlap)
   │
   ▼
VectorStoreManager (src/vector_store.py) ──► Local HuggingFace Embeddings + Persistent Chroma DB
```

### 2. Query Pipeline (Online — Every User Question)
```text
User Question (Streamlit ChatApp)
   │
   ▼
Query Rewriter (if chat history exists)  ──► Standalone Search Query
   │
   ▼
GrammarRetriever (src/retriever.py)      ──► Top-k (k=6) Relevant Chunks + Page Numbers
   │
   ▼
RAGChain (src/rag_chain.py — LCEL)       ──► Strict Prompt + Context + History → Groq LLM
   │                                           (Optional labeled Fallback if toggle ON & not in book)
   ▼
Structured 5-Section Answer + Page Citations → Streamlit UI (with Expandable Source Viewer)
```

---

## 📂 Project Structure (Single Responsibility & Separation of Concerns)
```text
Grammar_bot/
├── app.py                   # Thin Streamlit entry point (streamlit run app.py)
├── main.py                  # CLI ingestion runner (BookLoader -> TextChunker -> VectorStoreManager)
├── config.py                # Single source of truth for paths, chunk size, models, and top-k
├── requirements.txt         # Pinned Python dependencies
├── pyproject.toml           # uv project configuration
├── .env.example             # Environment variable template (safe to commit)
├── .env                     # Local secret API keys (git-ignored)
├── .gitignore               # Excludes .env, .venv, data/book/, and data/vector_store/
├── ui/
│   └── chatapp.py           # ChatApp class (Streamlit UI, upload/index, fallback toggle, chat, source viewer)
├── src/
│   ├── __init__.py
│   ├── book_loader.py       # BookLoader: page-wise PDF extraction + markdown cleaning
│   ├── text_chunker.py      # TextChunker: RecursiveCharacterTextSplitter wrapper
│   ├── vector_store.py      # VectorStoreManager: local embeddings + persistent Chroma DB
│   ├── retriever.py         # GrammarRetriever: similarity top-k retriever interface
│   └── rag_chain.py         # RAGChain: LCEL Conversation Retrieval Chain + strict/fallback modes + memory
├── eval/
│   ├── test_questions.json  # 8 evaluation queries (in-book, follow-up, out-of-book)
│   ├── eval.py              # Automated runner saving report to eval/eval_results.json
│   └── eval_results.json    # Generated evaluation output for human groundedness review
└── data/
    ├── book/                # Local private grammar PDF (git-ignored)
    └── vector_store/        # Local persistent Chroma index (git-ignored)
```

---

## 🛠️ Tech Stack & Architectural Trade-Offs

| Stage | Tool / Model | Cost / Mode | Why Chosen & Trade-Offs |
| :--- | :--- | :--- | :--- |
| **Language & UI** | Python + Streamlit | 100% Free, Local | Fast interactive chat UI with `@st.cache_resource` and `st.session_state`; best for single-user/portfolio apps. |
| **RAG Framework** | LangChain (LCEL) | 100% Free, Open Source | Modular loaders, splitters, retrievers, and clean pipe (`\|`) chains. |
| **PDF Loader** | `pymupdf4llm` (PyMuPDF) | 100% Free, Local | Preserves headings/layout far better than raw text on visual grammar books (Viva alternative: `PyPDFLoader`). |
| **Text Splitter** | `RecursiveCharacterTextSplitter` | 100% Free, Local | Splits on paragraphs/newlines first (`1000` chars / `200` overlap) while respecting model token limits. |
| **Embeddings** | `hotchpotch/bekko-embedding-v1-a8m` | 100% Free, Local | Ultra-compact (~8M params), 384-dim, 8,192 max tokens, runs fast on CPU (`float32`), no API rate limits. |
| **Vector Store** | Chroma (`langchain-chroma`) | 100% Free, Local | Persistent local SQLite-backed store with metadata support; single-machine scale. |
| **LLM** | Groq (`openai/gpt-oss-20b`) | Free Tier (Cloud API) | Ultra-fast inference (`temperature=0.0`, `reasoning_format="parsed"`, `reasoning_effort="low"`); requires internet & respects free-tier limits. |
| **Secrets** | `python-dotenv` (`.env`) | 100% Free, Local | Keeps `GROQ_API_KEY` out of source code and version control. |

---

## 🔐 Privacy & Security
- `GROQ_API_KEY` is stored strictly in a local `.env` file (excluded via `.gitignore`).
- The grammar book PDF (`data/book/`) and vector store (`data/vector_store/`) stay 100% local and are never pushed to GitHub.
- Embeddings are computed locally on CPU — the full book is never uploaded to an external embedding API.

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Grammar_bot
```

### 2. Install Dependencies
**Using `uv` (Recommended):**
```bash
uv sync
```
**Or using `pip`:**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and add your free Groq API key from [console.groq.com](https://console.groq.com/):
```bash
cp .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key
```

### 4. Add Your Grammar Book
Place your English grammar PDF inside `data/book/` and update `BOOK_PATH` in `config.py` if your filename differs (or upload it directly from the Streamlit sidebar).

---

## ▶️ How to Run

### Option 1: Launch Streamlit Chat UI (Recommended)
```bash
uv run streamlit run app.py
```
1. Open `http://localhost:8501` in your browser.
2. Click **"Index / Re-index Full Book"** in the sidebar (runs once, takes ~1–3 minutes on CPU).
3. Ask any grammar topic and expand **"View Book Sources & Page Excerpts"** to inspect exact book pages.
4. *(Optional)* Toggle **"Allow General AI Fallback"** in the sidebar if you want labeled general-knowledge answers for topics missing from the book (OFF by default for strict book-only mode).

### Option 2: Run CLI Ingestion
```bash
uv run python main.py
```

### Option 3: Run Evaluation Suite (FR10)
```bash
uv run python eval/eval.py
```
Review `eval/eval_results.json` to inspect retrieved page citations, structured responses, and guardrail pass/fail scores.

---

## 📊 Evaluation Summary (FR10)
Tested on **8 queries** across the full 362-page English grammar book (in default Strict Mode):
- **In-Book Topic Queries (Q1, Q3–Q6):** Retrieved exact book chapters (Present Simple, Past Simple vs. Present Perfect, Future `will`/`going to`, Relative Clauses, Conditionals) with accurate page citations.
- **Follow-Up Memory Query (Q2):** Correctly resolved `"What are its negative form rules?"` to Present Simple Negative using conversation memory and query rewriting.
- **Out-of-Book Guardrail (Q7–Q8):** **2/2 PASS** — replied `"Not found in the book."` with zero hallucination.

---

## ⚠️ Known Limitations & Risks
- **Visual / Image-Based PDF Boxes:** Illustrated grammar books often place some right/wrong example callouts inside images/graphics. Text-only extraction skips image-embedded text unless OCR (e.g., `RapidOCR`) is enabled, which trades off indexing speed on CPU.
- **Cloud LLM Dependency:** Answer generation requires internet access and operates within Groq's free-tier rate limits.
- **Scope Boundaries:** Intentionally scoped to a single local grammar book, single-user local web UI, and text-only Q&A.

---

## 🔮 Future Work
- Optional OCR toggle (`RapidOCR`) for visual/scanned grammar books.
- Multi-book support with book-level metadata filtering.
- Auto-generated grammar practice quizzes & voice Q&A.
- Cloud deployment and user accounts.