import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama

app = FastAPI(
    title="Local LLM Inference API Gateway",
    description="Production-ready FastAPI wrapper around local Ollama inference engine.",
    version="1.0.0"
)

class GenerationRequest(BaseModel):
    prompt: str
    model: str = "qwen2.5-coder:14b"
    temperature: float = 0.0

class GenerationResponse(BaseModel):
    model: str
    response: str
    ttft_ms: float
    total_latency_s: float

@app.get("/health")
def health_check():
    """Service health probe endpoint."""
    return {"status": "healthy", "engine": "Ollama local inference"}

@app.post("/v1/generate", response_model=GenerationResponse)
def generate_text(request: GenerationRequest):
    """
    Inference endpoint wrapping local LLM generation with metrics measurement.
    """
    start_time = time.perf_counter()
    
    try:
        response = ollama.chat(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            options={"temperature": request.temperature}
        )
        
        latency = time.perf_counter() - start_time
        
        return GenerationResponse(
            model=request.model,
            response=response["message"]["content"],
            ttft_ms=round(latency * 1000, 2), # Approximated for non-streamed HTTP
            total_latency_s=round(latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("⚡ Starting Local FastAPI Inference Gateway on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)