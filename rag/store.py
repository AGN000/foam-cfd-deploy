"""
Persistent vector store: numpy array (cosine search) + SQLite (metadata + text).
No external vector DB dependency.
"""
import json
import os
import sqlite3
import numpy as np
from pathlib import Path

from .vectorizer import vectorize, TOTAL_DIM


class VectorStore:
    def __init__(self, store_dir: str):
        self.store_dir  = Path(store_dir)
        self.vec_path   = self.store_dir / "vectors.npy"
        self.db_path    = self.store_dir / "chunks.db"
        self.meta_path  = self.store_dir / "index_meta.json"
        self._vectors: np.ndarray | None = None
        self._ids: list = []
        self._db: sqlite3.Connection | None = None
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._open_db()
        self._load_vectors()

    # ── persistence ───────────────────────────────────────────────────────────

    def _open_db(self):
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id            TEXT PRIMARY KEY,
                file_slot     TEXT,
                case_name     TEXT,
                geometry_type TEXT,
                turb_model    TEXT,
                regime        TEXT,
                is_2d         INTEGER,
                source        TEXT,
                text          TEXT
            )
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_slot ON chunks(file_slot)")
        self._db.commit()

    def _load_vectors(self):
        if self.vec_path.exists():
            self._vectors = np.load(str(self.vec_path))
            rows = self._db.execute("SELECT id FROM chunks ORDER BY rowid").fetchall()
            self._ids = [r[0] for r in rows]
        else:
            self._vectors = np.empty((0, TOTAL_DIM), dtype=np.float32)
            self._ids = []

    def _save_vectors(self):
        np.save(str(self.vec_path), self._vectors)

    # ── build / update ────────────────────────────────────────────────────────

    def build(self, chunks: list, verbose: bool = True):
        """Vectorize and index a list of chunk dicts. Idempotent."""
        existing = {r[0] for r in self._db.execute("SELECT id FROM chunks").fetchall()}
        new_chunks = [c for c in chunks if c["id"] not in existing]

        if not new_chunks:
            if verbose:
                print(f"  Store up to date ({len(existing)} chunks already indexed).")
            return

        vecs = []
        rows = []
        for c in new_chunks:
            vec = vectorize(c)
            vecs.append(vec)
            rows.append((
                c["id"], c["file_slot"], c["case_name"], c["geometry_type"],
                c["turb_model"], c["regime"], int(c["is_2d"]), c["source"], c["text"]
            ))

        new_mat = np.stack(vecs).astype(np.float32)
        self._vectors = np.vstack([self._vectors, new_mat]) if self._vectors.shape[0] else new_mat
        self._db.executemany(
            "INSERT OR IGNORE INTO chunks VALUES (?,?,?,?,?,?,?,?,?)", rows
        )
        self._db.commit()
        self._ids = [r[0] for r in self._db.execute("SELECT id FROM chunks ORDER BY rowid").fetchall()]
        self._save_vectors()

        if verbose:
            print(f"  Indexed {len(new_chunks)} new chunks ({len(self._ids)} total).")

    def add_chunk(self, chunk: dict):
        """Add a single chunk (used by DatasetCollector)."""
        self.build([chunk], verbose=False)

    # ── search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        file_slot: str = None,
        geometry_type: str = None,
        turb_model: str = None,
        regime: str = None,
        top_k: int = 3,
    ) -> list:
        if self._vectors.shape[0] == 0:
            return []

        # 1. Hard filter by file_slot via SQLite
        if file_slot:
            rows = self._db.execute(
                "SELECT id, case_name, geometry_type, turb_model, regime, is_2d, source, text "
                "FROM chunks WHERE file_slot = ?",
                (file_slot,)
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id, case_name, geometry_type, turb_model, regime, is_2d, source, text "
                "FROM chunks"
            ).fetchall()

        if not rows:
            return []

        # Build aligned (row, vector_index) pairs — skip rows not in current index
        id_to_idx = {cid: i for i, cid in enumerate(self._ids)}
        aligned = [(row, id_to_idx[row[0]]) for row in rows if row[0] in id_to_idx]
        if not aligned:
            return []

        aligned_rows, vec_idxs = zip(*aligned)
        cand_vecs = self._vectors[list(vec_idxs)]  # (M, D)

        # 2. Cosine similarity
        q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        scores = (cand_vecs @ q).copy()  # (M,)

        # 3. Metadata boosts (aligned_rows[i] matches scores[i])
        for i, row in enumerate(aligned_rows):
            if geometry_type and row[2] == geometry_type:
                scores[i] += 0.15
            if turb_model and row[3] == turb_model:
                scores[i] += 0.10
            if regime and row[4] == regime:
                scores[i] += 0.05

        # 4. Top-k
        top_idxs = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idxs:
            row = aligned_rows[i]
            results.append({
                "id":            row[0],
                "case_name":     row[1],
                "geometry_type": row[2],
                "turb_model":    row[3],
                "regime":        row[4],
                "is_2d":         bool(row[5]),
                "source":        row[6],
                "text":          row[7],
                "score":         float(scores[i]),
                "file_slot":     file_slot or "unknown",
            })
        return results

    def stats(self) -> dict:
        total = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        by_slot = dict(self._db.execute(
            "SELECT file_slot, COUNT(*) FROM chunks GROUP BY file_slot"
        ).fetchall())
        by_geom = dict(self._db.execute(
            "SELECT geometry_type, COUNT(*) FROM chunks GROUP BY geometry_type"
        ).fetchall())
        return {"total": total, "by_slot": by_slot, "by_geometry": by_geom}
