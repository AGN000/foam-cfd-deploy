"""
FastAPI inference server for mesh generation and CFD simulation.

Endpoints:
    POST /generate          — prompt → gmsh script (raw)
    POST /mesh              — prompt → validated mesh file path
    POST /simulate          — prompt → mesh + OpenFOAM run + result PNG
    GET  /health            — health check
    GET  /models            — list available models

Usage:
    python -m inference.server --model checkpoints/unified/merged --port 8000
"""
import argparse
import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from .mesh_pipeline import MeshPipeline
from simulation.case_builder import build_case
from simulation.foam_runner import run_simulation
from simulation.results_viz import visualize_results
from rag.store import VectorStore
from rag.retriever import RAGRetriever
from rag.llm_case_generator import LLMCaseGenerator
from rag.rag_case_builder import build_case_rag
from rag.dataset_collector import DatasetCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global singletons (loaded once at startup)
_pipeline: Optional[MeshPipeline] = None
_rag_retriever: Optional[RAGRetriever] = None
_llm_generator: Optional[LLMCaseGenerator] = None
_dataset_collector: Optional[DatasetCollector] = None

RAG_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "rag", "store")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _rag_retriever, _llm_generator, _dataset_collector
    model_path = os.environ.get("MESH_MODEL_PATH", "checkpoints/unified/merged")
    logger.info(f"Loading model from {model_path} ...")
    _pipeline = MeshPipeline(model_path=model_path)
    logger.info("Model loaded.")

    # Init RAG retriever
    try:
        store = VectorStore(RAG_STORE_DIR)
        stats = store.stats()
        if stats["total"] > 0:
            _rag_retriever = RAGRetriever(store)
            _llm_generator = LLMCaseGenerator()
            _dataset_collector = DatasetCollector()
            logger.info(f"RAG pipeline ready ({stats['total']} chunks indexed).")
        else:
            logger.warning("RAG store is empty — run `python3 -m rag.build_index` to build it.")
    except Exception as e:
        logger.warning(f"RAG init failed: {e} — using fallback case_builder")

    logger.info("Server ready.")
    yield
    _pipeline = None
    _rag_retriever = None
    _llm_generator = None


app = FastAPI(
    title="Mesh Generator API",
    description="Natural language → OpenFOAM mesh using a local LLM",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Natural language mesh description")
    max_retries: int = Field(3, ge=1, le=10)
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    max_new_tokens: int = Field(2048, ge=256, le=8192)


class GenerateResponse(BaseModel):
    script: str
    attempts: int
    valid: bool
    error: Optional[str] = None


class MeshRequest(BaseModel):
    prompt: str = Field(..., description="Natural language mesh description")
    output_format: str = Field("msh2", description="msh2 | msh4 | vtk")
    max_retries: int = Field(3, ge=1, le=10)
    temperature: float = Field(0.1, ge=0.0, le=1.0)


class MeshResponse(BaseModel):
    script: str
    mesh_path: str
    attempts: int
    valid: bool
    check_mesh_output: Optional[str] = None
    error: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.get("/models")
async def models():
    if _pipeline is None:
        return {"models": []}
    return {"models": [{"id": _pipeline.model_path, "type": "mesh-generator"}]}


@app.post("/generate", response_model=GenerateResponse)
async def generate_script(req: GenerateRequest):
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded")

    result = _pipeline.generate_script(
        prompt=req.prompt,
        max_retries=req.max_retries,
        temperature=req.temperature,
        max_new_tokens=req.max_new_tokens,
    )
    return GenerateResponse(
        script=result["script"],
        attempts=result["attempts"],
        valid=result["valid"],
        error=result.get("error"),
    )


@app.post("/mesh", response_model=MeshResponse)
async def generate_mesh(req: MeshRequest):
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded")

    result = _pipeline.generate_mesh(
        prompt=req.prompt,
        output_format=req.output_format,
        max_retries=req.max_retries,
        temperature=req.temperature,
    )

    if not result["valid"]:
        raise HTTPException(422, detail=f"Mesh generation failed after {result['attempts']} attempts: {result.get('error')}")

    return MeshResponse(**result)


@app.get("/download/{filename}")
async def download_mesh(filename: str):
    """Download a previously generated mesh file."""
    path = os.path.join("/tmp/meshgen", filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Mesh file not found")
    return FileResponse(path, filename=filename)


# ── OpenAI-compatible chat endpoint (used by LLMCaseGenerator) ───────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    temperature: float = 0.05
    max_tokens: int = 1024
    stop: Optional[list[str]] = None

class ChatChoice(BaseModel):
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"

class ChatResponse(BaseModel):
    choices: list[ChatChoice]
    model: Optional[str] = None

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    """OpenAI-compatible chat endpoint — delegates to the loaded LLM."""
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded")

    system = next((m.content for m in req.messages if m.role == "system"), "")
    user   = next((m.content for m in req.messages if m.role == "user"),   "")

    text = _pipeline._call_llm(
        system=system,
        user=user,
        temperature=req.temperature,
        max_new_tokens=req.max_tokens,
    )
    return ChatResponse(
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=text))],
        model=_pipeline.model_path,
    )


# ── /simulate ─────────────────────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    prompt: str = Field(..., description="Natural language geometry + flow description")
    output_png: Optional[str] = Field(None, description="Path for result image (auto if omitted)")
    max_retries: int = Field(5, ge=1, le=10)
    sim_timeout: int = Field(600, ge=30, le=3600, description="Solver timeout [s]")


class SimulateResponse(BaseModel):
    prompt: str
    mesh_path: str
    case_dir: str
    patches: list
    is_2d: bool
    sim_params: dict
    iterations: int
    check_mesh_ok: bool
    result_image: Optional[str] = None
    log_path: str
    residuals: dict
    ok: bool
    error: Optional[str] = None


@app.post("/simulate", response_model=SimulateResponse)
async def simulate(req: SimulateRequest):
    """
    End-to-end pipeline: natural language → mesh → OpenFOAM case → run → PNG.
    """
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded")

    # ── 1. Generate mesh ──────────────────────────────────────────────────────
    mesh_result = _pipeline.generate_mesh(
        prompt=req.prompt,
        output_format="msh2",
        max_retries=req.max_retries,
        temperature=0.1,
    )
    if not mesh_result["valid"]:
        raise HTTPException(
            422,
            detail=f"Mesh generation failed: {mesh_result.get('error')}",
        )
    mesh_path = mesh_result["mesh_path"]

    # ── 2. Build OpenFOAM case (RAG + LLM if available, else fallback) ───────
    case_dir = tempfile.mkdtemp(prefix="foam_case_")
    build_result = build_case_rag(
        case_dir, mesh_path, req.prompt,
        rag_retriever=_rag_retriever,
        llm_generator=_llm_generator,
        fallback=True,
    )
    if build_result.get("error"):
        raise HTTPException(
            422,
            detail=f"Case setup failed: {build_result['error']}",
        )

    # ── 3. Run simulation ─────────────────────────────────────────────────────
    sim_result = run_simulation(case_dir, timeout=req.sim_timeout)

    # ── 4. Visualize results ──────────────────────────────────────────────────
    output_png = req.output_png
    if output_png is None:
        os.makedirs("/home/ubuntu/foam-cfd-ai/outputs", exist_ok=True)
        import hashlib, time
        slug = hashlib.md5(req.prompt.encode()).hexdigest()[:8]
        output_png = f"/home/ubuntu/foam-cfd-ai/outputs/sim_{slug}.png"

    viz_result = visualize_results(
        case_dir,
        output_png,
        prompt=req.prompt,
        residuals=sim_result.get("residuals", {}),
    )

    # ── 5. Collect training data from successful RAG runs ────────────────────
    if (_dataset_collector is not None
            and build_result.get("rag_used")
            and any(build_result["rag_used"].values())):
        try:
            from rag.retriever import _infer_geometry, _infer_turb
            _dataset_collector.record_success(
                prompt=req.prompt,
                sim_params=build_result.get("sim_params", {}),
                retrieved_context={},   # context already baked into generated text
                generated_files={
                    slot: {"text": open(os.path.join(case_dir, slot)).read(), "valid": True}
                    for slot, used in build_result.get("rag_used", {}).items()
                    if used and os.path.exists(os.path.join(case_dir, slot))
                },
                residuals=sim_result.get("residuals", {}),
                case_dir=case_dir,
                patches=build_result.get("patches", []),
                is_2d=build_result.get("is_2d", False),
                geometry_type=_infer_geometry(req.prompt),
                turb_model=_infer_turb(req.prompt, build_result.get("sim_params")),
            )
        except Exception as e:
            logger.warning(f"Dataset collection failed: {e}")

    return SimulateResponse(
        prompt=req.prompt,
        mesh_path=mesh_path,
        case_dir=case_dir,
        patches=build_result.get("patches", []),
        is_2d=build_result.get("is_2d", False),
        sim_params=build_result.get("sim_params", {}),
        iterations=sim_result.get("iterations", 0),
        check_mesh_ok=("Mesh OK" in sim_result.get("check_mesh", "")),
        result_image=viz_result.get("output_png") if viz_result.get("ok") else None,
        log_path=sim_result.get("log_path", ""),
        residuals=sim_result.get("residuals", {}),
        ok=sim_result.get("ok", False),
        error=sim_result.get("error"),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default="checkpoints/unified/merged")
    parser.add_argument("--host",   type=str, default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    os.environ["MESH_MODEL_PATH"] = args.model
    uvicorn.run(
        "inference.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
