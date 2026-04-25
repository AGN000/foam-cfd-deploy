# foam-cfd-deploy v2

**An LLM-driven OpenFOAM agent.** Give it a natural-language CFD prompt and
it produces a runnable, validated OpenFOAM v2412 case — mesh, boundary
conditions, transport / turbulence / thermophysical properties, solver
control, and a logged simulation.

```text
"2D lid-driven cavity Re=1000, 2m square, water"
              │
              ▼
  ┌─────────────────────┐
  │ Qwen-14B fine-tune  │  fluid lookup, Re-consistency,
  │ → CFDParams (JSON)  │  geometry routing
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │ gmsh template (×13) │  domain sizing, boundary
  │ → polyMesh          │  tagging, wall refinement
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │ case_writer.py      │  controlDict, fvSchemes,
  │ → 0/, constant/,    │  fvSolution, U, p, k, ω,
  │    system/          │  T, alpha.water, ...
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │ simpleFoam / ...    │  → final residuals, mass
  │ runs to convergence │    conservation, BC sanity
  └──────────┬──────────┘
             ▼
        score (0-1)
        + saved case
```

## What's new in v2

- **Fine-tuned 14 B model** ([`arungovindneelan/foam-cfd-unified-14b`](https://huggingface.co/arungovindneelan/foam-cfd-unified-14b))
  — published, MIT-licensed, replaces the v1 7 B checkpoint
- **13 hand-written gmsh templates** (was 8): added periodic hill, multi-hill,
  S-bend, diffuser, 3 D sphere, Ahmed body, T-junction, convergent-divergent
  nozzle, 90° elbow; airfoil generalised to any NACA-4-digit code
- **Re·U·L·ν consistency enforcement** — fluid lookup table (water, oil,
  glycerine, air, CO2, …), characteristic length per geometry, regex
  fallback for `Re=`/`chord=`/`D=` patterns
- **Keyword-decisive physics flags** — eliminates LLM over-prediction of
  `is_transient` / `is_multiphase`
- **Interactive REPL** with arrow-key editing & persistent history
- **End-to-end eval**: 110/110 (100 %) PASS, 106/110 (96.4 %) solver-match
  on a fresh held-out OOD set (see `data/eval/ood_100_v2.json`)

## Repository layout

```
foam-cfd-deploy/
├── openfoam_agent/         # Python package (the agent itself)
│   ├── agent.py            # OpenFOAMAgent — orchestrator
│   ├── prompt_refiner.py   # natural-language → CFD-precise text
│   ├── param_extractor.py  # text → CFDParams (JSON-schema enforced)
│   ├── solver_selector.py  # CFDParams → solver name + numerical policy
│   ├── gmsh_generator.py   # CFDParams → polyMesh (13 templates)
│   ├── case_writer.py      # CFDParams + mesh → OpenFOAM dictionaries
│   ├── runner.py           # invoke solver, capture log + residuals
│   ├── reward.py           # score residual decay, mass conservation
│   ├── failure_diagnosis.py # residual / log analysis → retry context
│   ├── numerical_policy.py # y+-aware boundary-layer sizing
│   ├── prompt_catalog.py   # 252 in-distribution prompts (training)
│   ├── knowledge_base.py   # ChromaDB RAG over OpenFOAM tutorials
│   ├── schemas.py          # CFDParams / GeometryType / RunResult
│   └── config.py           # paths, vLLM init, env-var knobs
├── scripts/
│   ├── ask.sh              # one-shot: bash ask.sh "your prompt"
│   ├── repl.sh             # interactive REPL (arrow keys, history)
│   ├── run_agent.py        # CLI entry point (run / collect / train / ui)
│   ├── full_test_parallel.py  # multi-shard catalog/OOD evaluator
│   ├── train_qlora.py      # QLoRA fine-tune the base model
│   ├── merge_adapter.py    # adapter → merged bf16 weights
│   ├── build_hf_dataset.py # validated runs → HF dataset (per-file + chat)
│   └── ...                 # validation, RAG indexing, etc.
├── data/eval/              # held-out OOD prompt sets (ground truth)
├── pyproject.toml
├── requirements.txt
└── README.md               # (this file)
```

> **Not in this repo** (intentionally): generated cases (`data/cases/`,
> ~10 GB), checkpoints (`data/checkpoints/`, ~15 GB), the training dataset
> (`data/dataset/`, 36 MB — published on HF instead), ChromaDB indexes
> (`data/chroma_db/`). All of those are reproducible from the model + scripts
> in this repo.

## Public artefacts

| Artefact | Size | Where |
|---|---|---|
| Fine-tuned model (Qwen2.5-Coder-14B + QLoRA, merged bf16) | 9.3 GB | [HF: `arungovindneelan/foam-cfd-unified-14b`](https://huggingface.co/arungovindneelan/foam-cfd-unified-14b) |
| Training dataset (per-file + per-case chat configs) | 25 MB | [HF: `arungovindneelan/openfoam-Agent-Dataset`](https://huggingface.co/datasets/arungovindneelan/openfoam-Agent-Dataset) |
| Runnable case bundle (211 validated cases) | 525 MB | [GH: `AGN000/FoamAgentCases`](https://github.com/AGN000/FoamAgentCases) |
| Earlier 7 B checkpoint | 14 GB | [HF: `arungovindneelan/foam-cfd-unified-7b`](https://huggingface.co/arungovindneelan/foam-cfd-unified-7b) |

## Installation

### 1. OpenFOAM v2412

```bash
# Conda (works on Linux + WSL2)
conda create -n openfoam2412 -c conda-forge openfoam=2412 -y
echo 'source $(conda info --base)/envs/openfoam2412/etc/profile.d/conda.sh' >> ~/.bashrc
echo 'conda activate openfoam2412' >> ~/.bashrc
```

Verify with `which simpleFoam`.

### 2. Python environment

```bash
conda create -n vllm_env python=3.10 -y
conda activate vllm_env
pip install -r requirements.txt
pip install -e .   # installs the openfoam-agent package
```

GPU note: vLLM needs CUDA 12. The merged model is 9.3 GB in bf16 — fits on
any GPU with ≥ 30 GB free (H100, A100, RTX 6000 Ada, A6000).

### 3. Pull the fine-tuned model from HuggingFace

The agent doesn't auto-download — you fetch the model once into the project's
`data/checkpoints/` directory and point [`openfoam_agent/config.py`](openfoam_agent/config.py)
at it.

```bash
# One-time download (~9.3 GB)
mkdir -p data/checkpoints
hf download arungovindneelan/foam-cfd-unified-14b \
    --local-dir data/checkpoints/qwen_coder_14b_merged \
    --local-dir-use-symlinks False
```

(If `hf` is not yet installed: `pip install huggingface_hub[cli]` and then
`hf auth login` if the model is gated — this one is public, so no login
needed.)

Then in `openfoam_agent/config.py` make sure `LLM_MODEL` points at the
local copy:

```python
LLM_MODEL = "/abs/path/to/foam-cfd-deploy/data/checkpoints/qwen_coder_14b_merged"
```

### 4. (Optional) Pull the training dataset

Only needed if you intend to retrain or build a RAG index:

```bash
hf download arungovindneelan/openfoam-Agent-Dataset --repo-type dataset \
    --local-dir data/dataset
```

## Running the agent

### Easiest — interactive REPL

```bash
bash scripts/repl.sh
```

Loads vLLM once (~60 s the first time, model files are mmap'd on subsequent
runs), then accepts prompts in a loop:

```text
prompt> 2D lid-driven cavity Re=1000, 2m square, water
[repl] running... (timeout=300s, retries=1)

  ✓ score   : 0.87
    solver  : simpleFoam
    case    : data/cases/repl_001_attempt0
    feedback: OK
    elapsed : 38s

prompt> NACA 4412 airfoil chord 0.5m Re=1e6 AoA 4 deg air
prompt> Ahmed body 25 degrees slant air freestream 40 m/s
prompt> dam break 4m × 2m water column 1m wide interFoam
prompt> last           # re-print the last result
prompt> cases          # list 10 most recent case dirs
prompt> quit
```

Arrow keys, ↑/↓ history (persisted to `~/.openfoam_agent_repl_history`),
Ctrl-A/E for line start/end, Ctrl-R for reverse search.

### One-shot — single prompt

```bash
bash scripts/ask.sh "flow over circular cylinder D=10cm Re=1000 water"
```

Auto-picks a free GPU (≥ 30 GB free), sources OpenFOAM, runs the full
pipeline, prints the case directory at the end. Env-var knobs:

```bash
GPU=4                   bash scripts/ask.sh "..."   # specific GPU
TIMEOUT=600 RETRIES=2   bash scripts/ask.sh "..."   # longer solver timeout
VLLM_GPU_MEM_FRAC=0.7   bash scripts/ask.sh "..."   # if GPU is dedicated
```

### Programmatic — Python API

```python
from openfoam_agent.agent import OpenFOAMAgent

agent = OpenFOAMAgent(use_llm=True)
result = agent.run(
    prompt="3D pipe flow water D=4cm L=40cm Re=12000 turbulent",
    max_retries=2,
    sim_timeout=300,
)
print(f"score   : {result.score:.2f}")
print(f"solver  : {result.solver}")
print(f"case    : {result.case_dir}")
print(f"params  : {result.params.model_dump_json(indent=2)}")
```

### What you get

Every prompt produces a directory under `data/cases/<name>_attempt0/` with
the standard OpenFOAM tutorial layout:

```
0/                # initial / boundary conditions (U, p, k, ω, T, ...)
constant/
    polyMesh/     # gmsh-generated mesh, converted to polyMesh
    transportProperties
    turbulenceProperties
    thermophysicalProperties   # compressible cases only
    g                          # buoyant cases only
system/
    controlDict
    fvSchemes
    fvSolution
agent.log         # full solver stdout/stderr
```

To rerun or post-process the case standalone:

```bash
cd data/cases/<case_id>_attempt0
simpleFoam                 # rerun the solver
foamLog log.simpleFoam     # extract residual histories
paraFoam                   # open in ParaView
```

## Supported geometries

13 hand-written, parametric gmsh templates. The fine-tuned model routes
prompts to these by setting `geometry_type` in `CFDParams`:

| Family | Routes from prompts like… | Patches emitted |
|---|---|---|
| `lid_driven_cavity` | "lid-driven cavity Re=…" | movingWall, fixedWalls, frontAndBack |
| `pipe` | "pipe flow D=… L=…" | inlet, outlet, wall, frontAndBack (or front/back for 3D) |
| `cylinder` | "flow over a cylinder D=…" | inlet, outlet, cylinder, top, bottom, frontAndBack |
| `channel` / `box` | "channel water height=…" | inlet, outlet, walls, frontAndBack |
| `backward_facing_step` | "BFS step=…" | inlet, outlet, walls, frontAndBack |
| `airfoil` (NACA-4-digit) | "NACA 4412 chord=… AoA=…" | freestream, outlet, airfoil, frontAndBack |
| `wedge` | "axisymmetric pipe wedge D=… L=…" | inlet, outlet, wall, axis, front, back |
| `sphere` (3 D) | "flow over a sphere D=…" | inlet, outlet, sphere, walls |
| `periodic_hill` | "Wu/Mellen periodic hill, H=…" | inlet, outlet, topWall, bottomWall, frontAndBack |
| `multi_hill` | "three periodic hills in series" | as above |
| `s_bend` | "S-bend duct, half-height=…" | inlet, outlet, walls, frontAndBack |
| `diffuser` | "expanding duct inlet=…" | inlet, outlet, walls, frontAndBack |
| `ahmed_body` (3 D) | "Ahmed body 25° slant" | inlet, outlet, body, ground, top, sides |
| `t_junction` | "T-junction main + branch" | mainInlet, branchInlet, outlet, walls, frontAndBack |
| `cd_nozzle` | "convergent-divergent nozzle Mach=…" | inlet, outlet, walls, frontAndBack |
| `elbow` | "90° elbow duct" | inlet, outlet, walls, frontAndBack |

## Evaluation

The 110-prompt held-out OOD set in [`data/eval/ood_100_v2.json`](data/eval/ood_100_v2.json)
is the canonical regression benchmark. Run it after any change:

```bash
# Single shard (slow)
python scripts/full_test_parallel.py --shard 0/1 \
    --ood-file data/eval/ood_100_v2.json \
    --end-time 3 --timeout 120 \
    --out /tmp/eval.jsonl

# Or 8-shard parallel on 8 GPUs (~12 min for 110 prompts)
bash scripts/run_full_test_parallel.sh   # uses the in-distribution catalog
```

| Metric | Score |
|---|---|
| Cases that ran end-to-end | **110/110 (100 %)** |
| Solver-pick exact match | **106/110 (96.4 %)** |
| `simpleFoam` | 72/74 (97 %) |
| `buoyantSimpleFoam`, `icoFoam`, `interFoam`, `rhoSimpleFoam`, `rhoPimpleFoam` | **30/30 (100 %)** |
| `pimpleFoam` | 4/6 (67 %) — the 2 misses are low-Re transient cylinder, defensible icoFoam pick |

## Retraining

If you want to fine-tune from your own data:

```bash
# 1. Pull the per-case-chat training file from HF
hf download arungovindneelan/openfoam-Agent-Dataset --repo-type dataset \
    --local-dir data/dataset

# 2. QLoRA fine-tune (~25 min on H100)
python scripts/train_qlora.py \
    --jsonl data/dataset/expert_train.jsonl \
    --output data/checkpoints/qwen_coder_14b_lora \
    --epochs 3 --batch 1 --grad-accum 8 --lora-r 64 --lora-alpha 128

# 3. Merge adapter into base model (bf16, single file)
python scripts/merge_adapter.py \
    --adapter data/checkpoints/qwen_coder_14b_lora/final_adapter \
    --output  data/checkpoints/qwen_coder_14b_merged

# 4. Update openfoam_agent/config.py LLM_MODEL to the new path
```

## Generating fresh training data

If you want to expand the corpus:

```bash
# Run the prompt catalog through the agent (5-10 hrs on 8 GPUs at full end_time)
python scripts/generate_training_data.py --timeout 300

# Convert validated runs (score >= 0.5) to HF-compatible JSONL
python scripts/build_hf_dataset.py
# → data/dataset/foam_openfoam_dataset.jsonl  (per-file rows)
# → data/dataset/expert_train.jsonl           (per-case chat rows for SFT)
```

## License

MIT.

## Citation

```bibtex
@misc{foam_cfd_deploy_v2_2026,
  title  = {foam-cfd-deploy v2: a 14 B Qwen-Coder agent for OpenFOAM v2412 case authoring},
  author = {Neelan, Arun Govind},
  year   = {2026},
  howpublished = {GitHub repository},
  url    = {https://github.com/AGN000/foam-cfd-deploy}
}
```
