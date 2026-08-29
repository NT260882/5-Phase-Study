# Phase 2: Local AI & Edge Inference Engineering

An end-to-end implementation guide and repository specification for offline LLM execution, deterministic schema enforcement, self-correction loops, and hardware inference benchmarking.

---

## 🎯 What Phase 2 Tries to Achieve

Phase 2 transitions your engineering focus from RAG architectures (Phase 1) to **low-level model runtime mechanics and hardware optimization**. The primary goal is to master offline, edge-capable AI systems that do not depend on external APIs like OpenAI or Anthropic.

### Core Learning Objectives
1. **Offline & Edge Execution:** Deploy open-weight models (Llama 3.2, Qwen 2.5, Mistral 7B) locally using Ollama and wrap them inside production REST interfaces using FastAPI[cite: 1, 3].
2. **Deterministic Outputs & Reliability:** Force non-deterministic LLMs to output strictly valid JSON conforming to Pydantic schemas, and build automatic self-correction / re-prompting loops to handle structural failures[cite: 1, 3].
3. **Hardware & Quantization Telemetry:** Measure Time to First Token (TTFT), Tokens Per Second (TPS), and peak RAM footprint across low-bit GGUF quantization formats (e.g., Q4_K_M vs. Q5_K_M)[cite: 1, 3].

---

## 🏗️ Architecture Overview
                  ┌──────────────────────────────────────────────┐
                  │    FastAPI Gateway Interface (api_server.py) │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │   Validation & Retry Engine (schema_eval.py) │
                  │  - Pydantic Schema Validation                │
                  │  - Self-Correction Loop on Failure           │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │     Local Inference Engine (Ollama / API)    │
                  │     (Llama 3.2 3B / Qwen 2.5 / Gemma)        │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │   Telemetry & Benchmark Suite (benchmark.py) │
                  │  - TTFT, TPS, Memory, Quantization (GGUF)    │
                  └──────────────────────────────────────────────┘

---

## 📂 Repository File Breakdown

| File | Purpose & Responsibilities |
| :--- | :--- |
| `src/benchmark.py` | **Hardware Telemetry Engine:** Measures streaming latency to isolate Time To First Token (TTFT), calculates generated Tokens Per Second (TPS), and monitors active system RAM usage via `psutil`[cite: 1, 3]. |
| `src/schema_eval.py` | **Reliability & Validation Layer:** Enforces Pydantic schema constraints on raw LLM outputs, executes automated re-prompting loops when JSON parsing fails, and measures output variance at low (`0.0`) vs. high (`0.8`) temperatures[cite: 1, 3]. |
| `src/api_server.py` | **Local REST Gateway:** Wraps the local Ollama inference runtime inside a FastAPI application with structured input/output schemas and execution metrics[cite: 1, 3]. |
| `PHASE2_REPORT.md` | **Comparative Analysis:** Standardized technical report documenting hardware metrics, RAM footprints, and quantization trade-offs across open-weight models[cite: 1, 3]. |

---

## 🚀 Steps to Execute Phase 2

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally[cite: 1, 3]

### Step 1: Install Dependencies
```bash
pip install fastapi uvicorn pydantic ollama psutil requests
```

### Step 2: Pull Target Open-Weight Models
Pull the local models you plan to benchmark via your terminal:
```bash
ollama pull llama3.2
ollama pull qwen2.5:3b
```

### Step 3: Run the Hardware Benchmarking Suite
Execute benchmark.py to collect TTFT, TPS, and RAM usage metrics:
```bash
python3 src/benchmark.py
```

### Step 4: Test Deterministic JSON Schemas & Self-Correction
Run schema_eval.py to verify Pydantic validation, temperature variance (0.0 vs. 0.8), and automated retry loops[cite: 1, 3]:
```bash
python3 src/schema_eval.py
```

### Step 5: Launch the FastAPI Local Gateway
Start the local API gateway server:
```bash
python3 src/api_server.py
```
Access the interactive OpenAPI documentation in your browser at http://localhost:8000/docs to test endpoint responses[cite: 1, 3].