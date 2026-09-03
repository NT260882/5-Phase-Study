import os
import time
import ollama
from langfuse import Langfuse
from retriever import HybridReRankRetriever

# Set your Langfuse API Keys
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-your-public-key"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-your-secret-key"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"  # or https://cloud.langfuse.com

# Initialize Langfuse Client
langfuse = Langfuse()

MODEL_NAME = "qwen3:14b"  # Or your active local model

SYSTEM_PROMPT = """You are a strict Technical Documentation Assistant.
Answer the user's question ONLY using the provided context snippets.
Provide clear file citations [e.g., Source: file.md] for every claim."""

class TracedRAGPipeline:
    def __init__(self):
        self.retriever = HybridReRankRetriever(top_k=10, top_n=3)

    def generate_traced_answer(self, query: str):
        # 1. Start Root Trace
        trace = langfuse.trace(
            name="RAG_Request_Pipeline",
            user_id="dev_user_1",
            metadata={"model": MODEL_NAME, "environment": "production"}
        )

        start_total_time = time.perf_counter()

        # 2. Trace Retrieval & Re-ranking Span
        retrieval_span = trace.span(
            name="hybrid_retrieval_and_rerank",
            input={"query": query, "top_k": 10, "top_n": 3}
        )
        
        start_retrieval = time.perf_counter()
        retrieved_chunks = self.retriever.retrieve_and_rerank(query)
        retrieval_latency = time.perf_counter() - start_retrieval

        formatted_context = "\n".join([
            f"--- [Source: {c['metadata'].get('source')}] ---\n{c['content']}"
            for c in retrieved_chunks
        ])

        retrieval_span.end(
            output={
                "retrieved_count": len(retrieved_chunks),
                "top_chunk_source": retrieved_chunks[0]["metadata"].get("source") if retrieved_chunks else None,
                "top_rerank_score": retrieved_chunks[0].get("rerank_score") if retrieved_chunks else 0.0
            },
            metadata={"latency_s": round(retrieval_latency, 3)}
        )

        # 3. Trace LLM Generation Generation/Span
        user_message = f"QUERY: {query}\n\nCONTEXT:\n{formatted_context}\n\nANSWER:"
        
        generation = trace.generation(
            name="llm_generation_step",
            model=MODEL_NAME,
            model_parameters={"temperature": 0.0},
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )

        start_llm = time.perf_counter()
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            options={"temperature": 0.0}
        )
        llm_latency = time.perf_counter() - start_llm
        
        completion_text = response["message"]["content"]

        generation.end(
            output=completion_text,
            metadata={"llm_latency_s": round(llm_latency, 3)}
        )

        # 4. Finalize Root Trace Metrics
        total_latency = time.perf_counter() - start_total_time
        trace.update(
            output=completion_text,
            metadata={"total_latency_s": round(total_latency, 3)}
        )

        # Flush metrics to Langfuse dashboard
        langfuse.flush()

        return completion_text, total_latency

if __name__ == "__main__":
    pipeline = TracedRAGPipeline()
    print("⚡ Executing Traced RAG Request...")
    
    answer, latency = pipeline.generate_traced_answer(
        "How do I pass BackgroundTasks in FastAPI path functions?"
    )
    
    print(f"\n✅ Finished in {latency:.2f}s!")
    print(f"Response Snippet:\n{answer[:200]}...")