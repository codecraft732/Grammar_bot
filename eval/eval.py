import json
import time
from pathlib import Path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    BASE_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    TOP_K,
    VECTOR_STORE_DIR,
)
from src.rag_chain import RAGChain
from src.retriever import GrammarRetriever
from src.vector_store import VectorStoreManager


def run_evaluation() -> None:
    questions_path = BASE_DIR / "eval" / "test_questions.json"
    results_path = BASE_DIR / "eval" / "eval_results.json"

    test_cases = json.loads(questions_path.read_text(encoding="utf-8"))

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

    results = []
    out_of_book_passed = 0
    out_of_book_total = 0

    for item in test_cases:
        if item.get("clear_memory_before", True):
            rag.clear_memory()

        response = rag.ask(item["question"])
        answer = response["answer"]
        cited_pages = sorted(
            {doc.metadata.get("page", "N/A") for doc in response["sources"]}
        )

        auto_check = "HUMAN_CHECK_REQUIRED"
        if item["type"] == "out_of_book":
            out_of_book_total += 1
            passed = answer.strip() == "Not found in the book."
            if passed:
                out_of_book_passed += 1
            auto_check = "PASS" if passed else "FAIL"

        results.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "expected_behavior": item["expected_behavior"],
                "cited_pages": cited_pages,
                "answer": answer,
                "auto_check": auto_check,
                "human_groundedness_verdict": "PENDING (PASS/FAIL)",
            }
        )
        print(
            f"[Q{item['id']} | {item['type']}] Cited Pages: {cited_pages} | Check: {auto_check}"
        )
        time.sleep(2)  # Avoid Groq free-tier burst rate limit

    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved full report to: {results_path}")
    print(f"Out-of-Book Guardrail Score: {out_of_book_passed}/{out_of_book_total}")


if __name__ == "__main__":
    run_evaluation()
