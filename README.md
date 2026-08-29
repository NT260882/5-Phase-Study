# 5 AI Engineer Projects to Build in 2026

A structured, 5-phase learning path designed to bridge the gap between basic LLM demos and production-grade AI systems engineering[cite: 1, 3].

---

## Suggested Execution Timeline

- **Weeks 1–3:** Phase 1 (Production-Grade RAG Infrastructure & CI/CD Gating)[cite: 1]
- **Weeks 4–5:** Phase 2 (Local AI & Edge Inference Engineering)[cite: 1]
- **Weeks 6–7:** Phase 3 (AI Observability & SRE Operations)[cite: 1]
- **Weeks 8–10:** Phase 4 (Model Alignment & Parameter-Efficient Fine-Tuning)[cite: 1]
- **Weeks 11–12:** Phase 5 (Real-Time Multimodal Streaming Infrastructure)[cite: 1]

---

## Phase 1: Production-Grade RAG Infrastructure

**Goal:** Move beyond basic chunking to hybrid retrieval, re-ranking, and CI/CD evaluation[cite: 1].

### Foundations & Chunking
* Practice document ingestion across multiple formats (PDFs, Markdown, Web pages)[cite: 1].
* Implement fixed-size text chunking (500–800 tokens) with sliding window overlap (~100 tokens) to preserve contextual boundaries[cite: 1].
* **Tools:** LangChain or LangGraph, ChromaDB / Qdrant / Weaviate[cite: 1].

### Advanced Hybrid Retrieval & Re-ranking
* Combine dense vector semantic search with sparse keyword search (BM25) to catch both contextual intent and exact keyword matches[cite: 1, 3].
* Add a cross-encoder re-ranker to re-score top-$K$ vector matches as query-chunk pairs[cite: 1, 3].
* Implement strict citation enforcement so the model declines to answer when context is insufficient[cite: 1, 3].
* Externalize prompts into versioned configuration files[cite: 1, 3].
* **Tools:** Cohere Rerank or SentenceTransformers cross-encoders[cite: 1, 3].

### Continuous Evaluation & Quality Gating
* Curate a golden dataset of 50–200 verified Q&A ground-truth pairs[cite: 1, 3].
* Build automated offline evaluation scripts measuring faithfulness and context relevance[cite: 1, 3].
* Integrate evaluation into GitHub Actions CI pipelines to block PRs that drop quality below threshold limits[cite: 1, 3].
* **Tools:** Ragas framework[cite: 1, 3].

---

## Phase 2: Local AI & Edge Inference Engineering

**Goal:** Learn offline execution, constrained JSON output validation, and hardware benchmarking[cite: 1, 3].

### Local Model Ingestion & Benchmarking
* Run open-weight models (e.g., Llama 3.2, Mistral 7B) locally[cite: 1, 3].
* Set up API wrappers (e.g., FastAPI) around inference engines[cite: 1, 3].
* Record metrics: Tokens per second (TPS), Time to First Token (TTFT), and memory footprint[cite: 1, 3].
* **Tools:** Ollama or LM Studio[cite: 1, 3].

### Deterministic Output Schemas & Reliability
* Force structured JSON outputs using schema constraints[cite: 1, 3].
* Validate response payloads against schemas and build automatic self-correction / re-prompting loops[cite: 1, 3].
* Measure response variance across low (0.0) vs. high (0.7+) temperature settings[cite: 1, 3].
* **Tools:** Pydantic[cite: 1, 3].

### Comparative Hardware & Quantization Analysis
* Run standardized test prompts across multiple open models[cite: 1, 3].
* Benchmark precision vs. speed trade-offs using GGUF quantization formats (e.g., Q4_K_M vs. Q5_K_M)[cite: 1, 3].
* Document memory consumption and TPS degradation in a comparative technical report[cite: 1, 3].

---

## Phase 3: AI Observability & SRE Operations

**Goal:** Treat LLM applications like production services through tracing, latency SLAs, and regression testing[cite: 1, 3].

### Full-Pipeline Tracing
* Instrument RAG pipelines to log retrieved chunks, re-ranking scores, final prompt payloads, and token consumption per request[cite: 1, 3].
* **Tools:** Langfuse (self-hosted), Langsmith, or Braintrust[cite: 1, 3].

### SLA & Cost Tracking
* Monitor system latencies at P50 and P95 percentiles rather than relying on averages[cite: 1, 3].
* Track dollar-cost-per-request, citation coverage percentage, and failure rates over time[cite: 1, 3].

### Regression Gating & Deployment Guardrails
* Automate evaluation suite runs on every code or prompt commit[cite: 1, 3].
* Version prompt configurations alongside code logic to prevent silent quality regressions[cite: 1, 3].

---

## Phase 4: Model Alignment & Parameter-Efficient Fine-Tuning

**Goal:** Specialize open-weight models on structured extraction or tool call tasks using LoRA/DPO[cite: 1, 3].

### Supervised Fine-Tuning (SFT) with LoRA
* Curate a clean dataset (2,000–10,000 samples) targeting structured extraction or function calling[cite: 1, 3].
* Train low-rank adapters (LoRA / QLoRA) on single-GPU hardware[cite: 1, 3].
* Evaluate model checkpoints on held-out datasets using exact-match accuracy and JSON validation rates[cite: 1, 3].
* **Tools:** Hugging Face TRL, Axolotl, Unsloth[cite: 1, 3].

### Direct Preference Optimization (DPO)
* Construct pairwise comparison datasets containing prompt, chosen response, and rejected response[cite: 1, 3].
* Train DPO alignment on top of your SFT baseline to improve response quality and safety guardrails[cite: 1, 3].
* Plot loss curves and quantify incremental benchmark gains[cite: 1, 3].

---

## Phase 5: Real-Time Multimodal Streaming Infrastructure

**Goal:** Engineer low-latency event-driven applications with speech and visual streaming[cite: 1, 3].

### Streaming Audio/Voice Pipeline
* Build a full-duplex voice assistant over WebSockets: Speech-to-Text (ASR) $\rightarrow$ LLM Reasoning $\rightarrow$ Text-to-Speech (TTS)[cite: 1, 3].
* **Tools:** Deepgram / Whisper (ASR), ElevenLabs / Cartesia (TTS)[cite: 1, 3].

### Latency Budget Breakdown
* Deconstruct latency per step: ASR latency, LLM TTFT, and TTS Time to First Byte (TTFB)[cite: 1, 3].
* Target sub-1.5 second end-to-end response cycles[cite: 1, 3].

### System Resilience & Debugging
* Add timeout handlers, connection fallback strategies, and graceful error messaging[cite: 1, 3].
* Build an offline replay mode to feed recorded audio inputs through the pipeline for deterministic debugging[cite: 1, 3].
