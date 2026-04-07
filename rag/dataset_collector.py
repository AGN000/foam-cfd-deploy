"""
Collect successful RAG-generated cases into a fine-tuning dataset.
"""
import json
import logging
import os
from pathlib import Path

from .prompt_assembler import assemble_prompt, SLOT_SYSTEM_PROMPTS
from .chunker import chunk_case

logger = logging.getLogger(__name__)

DATASET_PATH = "/home/ubuntu/mesh-gen-ai/data/rag_finetune_dataset.jsonl"
RESIDUAL_THRESHOLD = 0.01  # accept if any residual field reached below this


def _final_residuals_ok(residuals: dict) -> bool:
    """Return True if p or Ux converged below threshold."""
    if not residuals:
        return False
    for field, values in residuals.items():
        if values and values[-1] < RESIDUAL_THRESHOLD:
            return True
    return False


class DatasetCollector:
    def __init__(self, dataset_path: str = DATASET_PATH):
        self.path = Path(dataset_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_success(
        self,
        prompt: str,
        sim_params: dict,
        retrieved_context: dict,    # {slot: context_text}
        generated_files: dict,      # {slot: {"text": str, "valid": bool}}
        residuals: dict,
        case_dir: str,
        patches: list = None,
        is_2d: bool = False,
        geometry_type: str = "generic",
        turb_model: str = "laminar",
    ) -> int:
        """
        Write one training record per file slot that was validly generated
        and where the simulation converged.
        Returns number of records written.
        """
        if not _final_residuals_ok(residuals):
            logger.info("Skipping dataset record: residuals did not converge")
            return 0

        written = 0
        with open(self.path, "a") as f:
            for slot, result in generated_files.items():
                if not result.get("valid"):
                    continue
                context = retrieved_context.get(slot, "")
                _, user_msg = assemble_prompt(
                    slot, prompt, context, sim_params,
                    patches or [], is_2d
                )
                record = {
                    "messages": [
                        {"role": "system",    "content": SLOT_SYSTEM_PROMPTS.get(slot, "")},
                        {"role": "user",      "content": user_msg},
                        {"role": "assistant", "content": result["text"]},
                    ],
                    "metadata": {
                        "slot":          slot,
                        "geometry_type": geometry_type,
                        "turb_model":    turb_model,
                        "case_dir":      case_dir,
                        "prompt":        prompt,
                    },
                }
                f.write(json.dumps(record) + "\n")
                written += 1

        logger.info(f"Wrote {written} training records to {self.path}")
        return written

    def add_to_vector_store(self, store, case_dir: str, metadata: dict = None):
        """
        After a successful simulation, add the generated case files as new
        chunks to the vector store (self-improving loop).
        """
        chunks = chunk_case(case_dir, source="working")
        if metadata:
            for c in chunks:
                c.update({k: v for k, v in metadata.items() if k in c})
        store.build(chunks, verbose=False)
        logger.info(f"Added {len(chunks)} chunks from {case_dir} to vector store")

    def dataset_size(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path) as f:
            return sum(1 for _ in f)
