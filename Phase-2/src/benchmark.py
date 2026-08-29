import time
import psutil
import ollama

def measure_inference(model_name: str, prompt: str):
    """
    Rigorously benchmarks Time to First Token (TTFT), Tokens Per Second (TPS),
    and active System RAM footprint during local model inference.
    """
    process = psutil.Process()
    ram_before_mb = process.memory_info().rss / (1024 * 1024)

    print(f"\n⚡ Benchmarking Model: [{model_name}]")
    print(f"   RAM Footprint Before Ingestion: {ram_before_mb:.2f} MB")

    start_time = time.perf_counter()
    ttft = None
    token_count = 0
    full_response = ""

    try:
        # Stream responses to isolate Time To First Token (TTFT)
        stream = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in stream:
            if ttft is None:
                ttft = time.perf_counter() - start_time  # First token arrival
            
            content = chunk["message"]["content"]
            full_response += content
            token_count += 1

        end_time = time.perf_counter()
        total_latency = end_time - start_time
        tps = (token_count / (total_latency - ttft)) if (total_latency - ttft) > 0 else 0
        ram_after_mb = process.memory_info().rss / (1024 * 1024)

        print("\n--- BENCHMARK RESULTS ---")
        print(f"  • Time to First Token (TTFT) : {ttft * 1000:.2f} ms")
        print(f"  • Total Response Latency     : {total_latency:.2f} s")
        print(f"  • Generated Token Count      : {token_count} tokens")
        print(f"  • Tokens Per Second (TPS)    : {tps:.2f} tok/s")
        print(f"  • Peak RAM Usage Difference  : {ram_after_mb - ram_before_mb:.2f} MB")
        print("-------------------------")

        return {
            "model": model_name,
            "ttft_ms": round(ttft * 1000, 2),
            "tps": round(tps, 2),
            "total_latency_s": round(total_latency, 2)
        }

    except Exception as e:
        print(f"❌ Error during benchmarking for {model_name}: {e}")
        return None

if __name__ == "__main__":
    test_prompt = "Write a concise, 100-word explanation of what an API gateway is."
    # Change model names based on your local Ollama pull list
    measure_inference("llama3.2:latest", test_prompt)