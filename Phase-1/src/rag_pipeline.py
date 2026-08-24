import sys
import ollama
from retriever import HybridReRankRetriever

MODEL_NAME = "qwen3:14b"  # Or mistral, qwen2.5, etc.

SYSTEM_PROMPT = """You are a strict, factual Technical Documentation Assistant.
Your task is to answer the user's question ONLY using the provided documentation snippets below.

CRITICAL RULES:
1. Every claim or factual statement you make MUST be directly supported by the context.
2. For EVERY answer, you MUST cite the source file from the metadata (e.g., [Source: tutorial/background-tasks.md]).
3. IF THE CONTEXT DOES NOT CONTAIN ENOUGH INFORMATION TO ANSWER THE QUERY, YOU MUST EXPLICITLY REFUSE TO ANSWER. Say: "I am sorry, but the provided documentation does not contain enough information to answer this question."
4. DO NOT use external knowledge or fabricate details not present in the provided context.
"""

class ProductionRAGPipeline:
    def __init__(self):
        self.retriever = HybridReRankRetriever(top_k=10, top_n=3)

    def format_context(self, retrieved_chunks):
        """Formats candidates into a structured context string with clear source tags."""
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk["metadata"].get("source", "Unknown Source")
            content = chunk["content"]
            block = f"--- DOCUMENT SNIPPET {i} [Source: {source}] ---\n{content}\n"
            context_blocks.append(block)
        return "\n".join(context_blocks)

    def generate_answer(self, query: str):
        print(f"\n🔍 Retrieving & Re-Ranking context for: '{query}'...")
        chunks = self.retriever.retrieve_and_rerank(query)

        if not chunks:
            return "I am sorry, but no relevant documentation could be retrieved."

        context_str = self.format_context(chunks)

        user_message = f"""USER QUERY: {query}

RELEVANT DOCUMENTATION CONTEXT:
{context_str}

ANSWER (Include source citations):"""

        print("⚡ Sending grounded payload to LLM...")
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            options={"temperature": 0.0}  # Low temperature for deterministic factual answers
        )

        return response["message"]["content"]

if __name__ == "__main__":
    pipeline = ProductionRAGPipeline()

    # Test 1: In-domain query (Should answer with strict file citations)
    valid_query = "How do I pass BackgroundTasks in FastAPI path functions?"
    print(f"\n==================== TEST 1: VALID QUERY ====================")
    ans1 = pipeline.generate_answer(valid_query)
    print(f"\nAI RESPONSE:\n{ans1}\n")

    # Test 2: Out-of-domain query (Should trigger strict refusal guardrail)
    invalid_query = "How do I configure AWS S3 bucket policies in Terraform?"
    print(f"\n==================== TEST 2: OUT-OF-DOMAIN QUERY ====================")
    ans2 = pipeline.generate_answer(invalid_query)
    print(f"\nAI RESPONSE:\n{ans2}\n")