# foam-cfd-deploy

Minimal deployment package for the **foam-cfd-unified-7b** model.
Clone this repo, download the model, and run CFD simulations from plain-English prompts. This is in the development phase. To convert this into a powerful tool, I need a variety of OpenFOAM examples. If you are interested in sharing a variety of test cases, please contact me at arungovindneelan@gmail.com.

**Model:** [`arungovindneelan/foam-cfd-unified-7b`](https://huggingface.co/arungovindneelan/foam-cfd-unified-7b)

---

## What's included

| Folder | Purpose |
|---|---|
| `inference/` | FastAPI server + Gmsh mesh generation pipeline |
| `rag/` | RAG retriever + LLM-driven OpenFOAM case builder |
| `simulation/` | OpenFOAM runner + results visualisation |
| `dataset/generators/` | Deterministic geometry fast-paths (airfoil, cylinder, etc.) |
| `demo.py` | End-to-end demo script |
| `run_and_simulate.sh` | Convenience script: starts server + runs a simulation |

---

## Requirements

- Linux (Ubuntu 20.04+)
- Python 3.10+
- NVIDIA GPU with **≥ 8 GB VRAM**
- CUDA 12.x
- OpenFOAM 11

---

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/AGN000/foam-cfd-deploy
cd foam-cfd-deploy
pip install -r requirements.txt
```

### 2. Download the model (~15 GB)

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'arungovindneelan/foam-cfd-unified-7b',
    local_dir='checkpoints/unified/merged'
)
"
```

### 3. Install system dependencies

```bash
# Required by gmsh on headless servers
sudo apt-get install -y libglu1-mesa libgl1-mesa-glx
```

### 4. Install OpenFOAM 11

```bash
wget -q -O - https://dl.openfoam.org/gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://dl.openfoam.org/ubuntu focal main"
sudo apt-get update && sudo apt-get install -y openfoam11
source /opt/openfoam11/etc/bashrc
```

### 5. Build the RAG index (run once)

```bash
python3 -m rag.build_index
```

---

## Run

### Option A — Demo script (easiest)

```bash
python3 demo.py "Lid driven cavity, Re=1000"
python3 demo.py "Flow over a cylinder, diameter 0.05m, Re=100"
python3 demo.py "NACA 0012 airfoil, chord 1m, AoA 5 degrees"
python3 demo.py "Backward-facing step, step height 0.05m, Re=500"
```

### Option B — Start the API server

```bash
python3 -m inference.server --model checkpoints/unified/merged --port 8000
```

Then call the API:

```bash
# Health check
curl http://localhost:8000/health

# Full simulation
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Lid driven cavity, Re=1000, 0.1m x 0.1m"}'
```

### Option C — Convenience script

```bash
./run_and_simulate.sh
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server status |
| POST | `/generate` | Prompt → Gmsh `.geo` script |
| POST | `/mesh` | Prompt → validated mesh file |
| POST | `/simulate` | Full pipeline: mesh → OpenFOAM → results |

---

## Supported Geometries

| Geometry | Example Prompt |
|---|---|
| Lid-driven cavity | `"Lid driven cavity, Re=1000, 0.1m x 0.1m"` |
| Pipe flow | `"Circular pipe, radius 0.05m, length 1m, Re=500"` |
| Cylinder in crossflow | `"Flow over a cylinder, diameter 0.05m, Re=100"` |
| Backward-facing step | `"Backward-facing step, step height 0.05m, Re=500"` |
| NACA airfoil | `"NACA 0012 airfoil, chord 1m, AoA 5 degrees, Re=1e6"` |
| Annular pipe | `"Annular pipe, inner radius 0.02m, outer 0.05m, length 0.5m"` |

---

## Troubleshooting

**GPU memory / OOM:**
```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
```

**OpenFOAM not found:**
```bash
source /opt/openfoam11/etc/bashrc
```

**Slow download:**
```bash
pip install hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('arungovindneelan/foam-cfd-unified-7b', local_dir='checkpoints/unified/merged')
"
```
