import numpy as np
import pandas as pd
import time
from tracing_rag import TracedRAGPipeline

def run_sre_latency_benchmark(sample_queries: list[str]):
    pipeline = TracedRAGPipeline()
    latencies = []
    success_count = 0
    refusal_count = 0

    print(f"⚡ Running SRE Latency Benchmark over {len(sample_queries)} queries...")

    for i, query in enumerate(sample_queries, 1):
        print(f" Processing query {i}/{len(sample_queries)}: '{query[:40]}...'")
        try:
            ans, total_time = pipeline.generate_traced_answer(query)
            latencies.append(total_time)
            success_count += 1
            
            if "does not contain enough information" in ans:
                refusal_count += 1
        except Exception as e:
            print(f"❌ Query failed with error: {e}")

    # Calculate Percentile Metrics
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    citation_coverage = ((success_count - refusal_count) / len(sample_queries)) * 100

    print("\n================ SRE SYSTEM HEALTH REPORT ================")
    print(f"  • Total Requests Processed : {len(sample_queries)}")
    print(f"  • P50 Latency (Median)     : {p50_latency:.2f} s")
    print(f"  • P95 Latency (Tail / SLA) : {p95_latency:.2f} s")
    print(f"  • Grounded Citation Rate   : {citation_coverage:.1f} %")
    print(f"  • Guardrail Refusal Rate   : {(refusal_count / len(sample_queries)) * 100:.1f} %")
    print("==========================================================")

if __name__ == "__main__":
    test_suite = [
        "How do I pass BackgroundTasks in FastAPI path functions?",
        "What is the CLI flag to enable reload in Uvicorn?",
        "How do I configure AWS S3 bucket policies in Terraform?",  # Refusal test
        "How do I handle background execution tasks after returning an API response?",
        "What is the default port for FastAPI dev server?"
    ]
    run_sre_latency_benchmark(test_suite)