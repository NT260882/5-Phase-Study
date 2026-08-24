import os
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# --- CONFIGURATION ---
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "fastapi_docs"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HybridReRankRetriever:
    def __init__(self, top_k: int = 15, top_n: int = 4):
        self.top_k = top_k  # Candidates fetched per retrieval method
        self.top_n = top_n  # Final chunks sent to LLM

        print("⚡ Initializing ChromaDB vector store...")
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)

        print("⚡ Loading all ingested chunks into BM25 Index...")
        all_docs = self.collection.get(include=["documents", "metadatas"])
        self.corpus_docs = all_docs["documents"]
        self.corpus_metadatas = all_docs["metadatas"]

        # Tokenize documents for BM25
        tokenized_corpus = [doc.lower().split(" ") for doc in self.corpus_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"⚡ Loading Cross-Encoder re-ranker model '{RERANKER_MODEL}'...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

    def bm25_search(self, query: str):
        """Sparse Keyword Search using BM25."""
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get indices of top_k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "content": self.corpus_docs[idx],
                "metadata": self.corpus_metadatas[idx],
                "source_method": "bm25"
            })
        return results

    def vector_search(self, query: str):
        """Dense Semantic Search using ChromaDB."""
        results = self.collection.query(
            query_texts=[query],
            n_results=self.top_k,
            include=["documents", "metadatas"]
        )

        extracted = []
        if results["documents"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                extracted.append({
                    "content": doc,
                    "metadata": meta,
                    "source_method": "vector"
                })
        return extracted

    def retrieve_and_rerank(self, query: str):
        """
        1. Fetches top-K candidates from BM25 and Vector Search.
        2. Deduplicates candidates.
        3. Re-scores candidates using Cross-Encoder.
        4. Returns top-N final candidates.
        """
        # Step 1: Hybrid candidate retrieval
        bm25_candidates = self.bm25_search(query)
        vector_candidates = self.vector_search(query)

        # Step 2: Deduplicate candidate documents by content
        seen_contents = set()
        unique_candidates = []

        for candidate in bm25_candidates + vector_candidates:
            if candidate["content"] not in seen_contents:
                seen_contents.add(candidate["content"])
                unique_candidates.append(candidate)

        if not unique_candidates:
            return []

        # Step 3: Cross-Encoder Scoring (evaluates query & candidate together)
        pairs = [[query, candidate["content"]] for candidate in unique_candidates]
        rerank_scores = self.reranker.predict(pairs)

        # Attach scores to candidates
        for idx, score in enumerate(rerank_scores):
            unique_candidates[idx]["rerank_score"] = float(score)

        # Step 4: Sort by Cross-Encoder score and return top-N
        reranked_results = sorted(
            unique_candidates, key=lambda x: x["rerank_score"], reverse=True
        )[:self.top_n]

        return reranked_results


if __name__ == "__main__":
    # Test query requiring both keyword and conceptual intent
    test_query = "How do I configure background tasks in FastAPI using uvicorn reload?"
    
    retriever = HybridReRankRetriever(top_k=10, top_n=3)
    results = retriever.retrieve_and_rerank(test_query)

    print(f"\n Top {len(results)} Re-Ranked Candidates for Query: '{test_query}'\n" + "="*70)
    for i, res in enumerate(results, 1):
        print(f"Rank {i} | Cross-Encoder Score: {res['rerank_score']:.4f} | Method Source: {res['source_method']}")
        print(f"Source File: {res['metadata'].get('source')}")
        print(f"Content Snippet:\n{res['content'][:200]}...")
        print("-" * 70)