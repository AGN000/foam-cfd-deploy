# foam-cfd-ai — Deployment Guide

Single Qwen2.5-Coder-7B model fine-tuned for CFD: Gmsh mesh generation,
OpenFOAM BC file generation, and boundary-condition JSON patches.

---

## Requirements

- Linux (Ubuntu 20.04+ recommended)
- NVIDIA GPU with ≥8 GB VRAM (H100/A100/3090 recommended)
- CUDA 12.x
- OpenFOAM 11 (for simulation)
- Python 3.10+

---

## 1. Install system dependencies

```bash
# Required by gmsh on headless/server installs
sudo apt-get install -y libglu1-mesa libgl1-mesa-glx
```

## 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Get the trained model

**Option A — Download from HuggingFace (after upload):**
```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('arungovindneelan/foam-cfd-unified-7b',
                  local_dir='checkpoints/unified/merged')
"
```

**Option B — Copy from training machine:**
```bash
# On training machine, after training completes:
rsync -av --progress checkpoints/unified/merged/ user@deploy-machine:/path/to/foam-cfd-ai/checkpoints/unified/merged/
```

**Option C — Train from scratch (if you have GPU):**
```bash
python3 training/train.py --config training/config_unified.yaml
# Model saved to: checkpoints/unified/merged/
```

---

## 3. Build the RAG index

The RAG index requires OpenFOAM tutorial files. Run once after installation:

```bash
python3 -m rag.build_index
```

---

## 4. Start the server

```bash
# Simple start:
python3 -m inference.server --model checkpoints/unified/merged --port 8000

# Or use the convenience script:
./run_and_simulate.sh
```

---

## 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Generate a mesh script
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "2D backward-facing step, height 0.1m, step at x=0.2m"}'

# Full mesh + simulate
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "flow over a cylinder, Re=100, diameter 0.1m"}'
```

---

## Folder structure

```
foam-cfd-ai/
  checkpoints/unified/merged/   ← trained 7B model (add after training/download)
  data/
    unified_train.jsonl          ← 20,505 training examples (mesh+foam+patch)
    unified_val.jsonl            ← 2,277 validation examples
    merge_datasets.py            ← script to rebuild unified dataset
  inference/
    server.py                    ← FastAPI server (POST /generate /mesh /simulate)
    mesh_pipeline.py             ← Gmsh script generation + validation
  rag/
    build_index.py               ← build vector store from OpenFOAM tutorials
    llm_case_generator.py        ← LLM-driven BC file generator
    rag_case_builder.py          ← top-level: RAG + LLM → OpenFOAM case dir
    retriever.py                 ← vector search over tutorial chunks
    store/                       ← pre-built RAG index (chunks.db + vectors.npy)
  simulation/
    case_builder.py              ← fallback hardcoded case builder
    foam_runner.py               ← runs OpenFOAM (foamRun -solver incompressibleFluid)
    results_viz.py               ← matplotlib residual + mesh visualisation
  training/
    train.py                     ← QLoRA fine-tuning with Unsloth
    config_unified.yaml          ← unified 7B training config
  requirements.txt
```

---

## Model info

| Field | Value |
|---|---|
| Base model | Qwen2.5-Coder-7B-Instruct |
| Fine-tuning | QLoRA (r=64, 4-bit) via Unsloth |
| Training data | 20,505 examples (mesh-gen + foam-gen + patch-gen + raft-patch-gen) |
| Tasks | Gmsh .geo script gen, OpenFOAM BC files, JSON patch specs |
| VRAM required | ~6 GB (4-bit) |
