# Phase 2: Hardware & Quantization Technical Report

## System Benchmarking Environment
- **Hardware:** Apple Mac Mini (M2, 8GB Unified Memory)
- **Runtime:** Ollama local inference engine
- **Test Prompt:** 100-word standard technical explanation query

## Performance Comparison Matrix

| Model Name | Quantization | Model Size | TTFT (ms) | Speed (TPS) | Peak RAM Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3.2 (1B)** | Q4_K_M | ~1.3 GB | ~120 ms | ~42 tok/s | ~1.4 GB |
| **Llama 3.2 (3B)** | Q4_K_M | ~2.0 GB | ~280 ms | ~28 tok/s | ~2.2 GB |
| **Qwen 2.5 (3B)** | Q4_K_M | ~1.9 GB | ~260 ms | ~30 tok/s | ~2.1 GB |

## Key Findings & Trade-offs
1. **TTFT vs. Model Scale:** Larger models increase initial Time To First Token latency due to memory loading overhead.
2. **Temperature Variance:** Setting temperature to `0.0` ensured 100% Pydantic schema validation success on Attempt 1, whereas `0.8` required re-prompt retry loops on 20% of requests due to structural JSON formatting errors.