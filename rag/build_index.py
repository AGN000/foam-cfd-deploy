"""
Build (or update) the RAG vector index from OpenFOAM tutorial cases.

Usage:
    python -m rag.build_index
    python -m rag.build_index --tutorials /opt/openfoam11/tutorials --store-dir rag/store
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chunker import walk_tutorial_cases, walk_working_cases
from rag.store import VectorStore

TUTORIALS_ROOT = "/opt/openfoam11/tutorials"
STORE_DIR      = str(Path(__file__).parent / "store")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutorials",  default=TUTORIALS_ROOT)
    ap.add_argument("--store-dir",  default=STORE_DIR)
    ap.add_argument("--working-cases", nargs="*", default=[])
    ap.add_argument("--extra-dirs", nargs="*", default=[],
                    help="Additional directories to walk for OF cases (e.g. cloned GitHub repos)")
    ap.add_argument("--rebuild", action="store_true", help="Drop existing index first")
    args = ap.parse_args()

    store_dir = Path(args.store_dir)

    if args.rebuild:
        for f in ["vectors.npy", "chunks.db", "index_meta.json"]:
            p = store_dir / f
            if p.exists():
                p.unlink()
        print("Cleared existing index.")

    print(f"Walking tutorials: {args.tutorials}")
    t0 = time.time()
    chunks = walk_tutorial_cases(args.tutorials)
    print(f"  Found {len(chunks)} chunks from {args.tutorials} ({time.time()-t0:.1f}s)")

    if args.working_cases:
        wchunks = walk_working_cases(args.working_cases)
        chunks += wchunks
        print(f"  + {len(wchunks)} chunks from {len(args.working_cases)} working cases")

    for extra_dir in (args.extra_dirs or []):
        extra_path = Path(extra_dir)
        if not extra_path.is_dir():
            print(f"  WARN: --extra-dirs path not found, skipping: {extra_dir}")
            continue
        extra_chunks = walk_tutorial_cases(str(extra_path))
        chunks += extra_chunks
        print(f"  + {len(extra_chunks)} chunks from extra dir: {extra_dir}")

    print(f"Building vector store at {store_dir} ...")
    store = VectorStore(str(store_dir))
    store.build(chunks, verbose=True)

    stats = store.stats()
    print(f"\nIndex statistics:")
    print(f"  Total chunks : {stats['total']}")
    print(f"\n  By file slot:")
    for slot, n in sorted(stats["by_slot"].items()):
        print(f"    {slot:<40} {n}")
    print(f"\n  By geometry type:")
    for geom, n in sorted(stats["by_geometry"].items()):
        print(f"    {geom:<20} {n}")

    meta = {
        "built_at":       time.strftime("%Y-%m-%d %Human:%M:%S"),
        "tutorials_root": args.tutorials,
        "total_chunks":   stats["total"],
        "by_slot":        stats["by_slot"],
        "by_geometry":    stats["by_geometry"],
    }
    with open(store_dir / "index_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s. Metadata saved to {store_dir}/index_meta.json")


if __name__ == "__main__":
    main()
