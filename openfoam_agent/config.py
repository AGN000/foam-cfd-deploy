from __future__ import annotations

from pathlib import Path
from typing import Optional

OPENFOAM_BASHRC = "/home/nvidia/miniconda3/envs/openfoam2412/etc/bashrc"
TUTORIALS_DIR = Path("/data/foamllm2/github/OpenFOAM_Tutorials_")
PROJECT_ROOT = Path("/data/foamllm3/openfoam_agent")

CASES_DIR = PROJECT_ROOT / "data/cases"
LOGS_DIR = PROJECT_ROOT / "data/logs"
DATASET_DIR = PROJECT_ROOT / "data/dataset"
CHECKPOINTS_DIR = PROJECT_ROOT / "data/checkpoints"
CHROMA_DIR = PROJECT_ROOT / "data/chroma_db"

LLM_MODEL = "/data/foamllm3/openfoam_agent/data/checkpoints/qwen_coder_14b_merged"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MAX_SEQ_LEN = 8192
import os as _os
VLLM_GPU_MEMORY_FRACTION = float(_os.environ.get("VLLM_GPU_MEM_FRAC", "0.85"))
VLLM_MAX_NUM_SEQS = int(_os.environ.get("VLLM_MAX_NUM_SEQS", "256"))

_llm_instance = None


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        from vllm import LLM
        _llm_instance = LLM(
            model=LLM_MODEL,
            max_model_len=MAX_SEQ_LEN,
            gpu_memory_utilization=VLLM_GPU_MEMORY_FRACTION,
            max_num_seqs=VLLM_MAX_NUM_SEQS,
            dtype="bfloat16",
        )
    return _llm_instance


def get_unsloth_model(load_for_training: bool = False):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LLM_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    if load_for_training:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    return model, tokenizer
