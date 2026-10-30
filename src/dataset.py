"""Reader for the BOBSL FS23K fingerspelling LMDB.

Each record is one subtitle segment that overlaps >=1 FS23K fingerspelling
interval. The payload holds the segment's SMPL-X params plus the fingerspelled
words and their segment-relative frame spans:

    smplx        dict of arrays (body_pose 6D, hands/jaw Euler ZYX, expression,
                 eyelid_params, betas, shape, global_orient, transl, K, RTs)
    text         subtitle text
    fs_words     list[str]            fingerspelled words in this segment
    fs_spans     list[[start, end]]   per-word frame span (relative to the clip)
    video        episode id
    order        start frame in the episode (25 fps)
    num_frames   T

Open read-only; safe to share across workers (``lock=False``).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import lmdb
import numpy as np


class FS23KDataset:
    """Light read-only view over ``bobsl_fs23k.lmdb``."""

    def __init__(self, lmdb_path: str | Path):
        self.path = Path(lmdb_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.env = lmdb.open(str(self.path), subdir=False, readonly=True,
                             lock=False, readahead=False)
        with self.env.begin() as txn:
            self.meta = pickle.loads(txn.get(b"__meta__"))
        self.clip_ids: list[str] = self.meta["clip_ids"]

    def __len__(self) -> int:
        return len(self.clip_ids)

    def _read(self, clip_id: str) -> dict:
        with self.env.begin() as txn:
            raw = txn.get(clip_id.encode("utf-8"))
        if raw is None:
            raise KeyError(clip_id)
        rec = pickle.loads(raw)
        # de-fp16 the stored pose tensors to float32 for downstream math
        rec["smplx"] = {k: np.asarray(v, dtype=np.float32)
                        for k, v in rec["smplx"].items()}
        return rec

    def __getitem__(self, idx: int | str) -> dict:
        clip_id = idx if isinstance(idx, str) else self.clip_ids[idx]
        return self._read(clip_id)

    def summary(self) -> dict:
        """Dataset-level stats from ``__meta__`` (no record scan)."""
        m = self.meta
        return {
            "clips": m["n"],
            "episodes": len(m["by_video"]),
            "fs_intervals": m.get("n_fs_intervals"),
            "fs_vocab_size": len(m.get("fs_vocab", [])),
            "dtype": m.get("keypoints_dtype"),
            "layout": m.get("layout"),
            "fps": m.get("fps"),
            "source": m.get("source"),
        }

    def top_words(self, k: int = 20) -> list[tuple[str, int]]:
        counts = self.meta.get("fs_counts", {})
        return sorted(counts.items(), key=lambda kv: -kv[1])[:k]

    def close(self) -> None:
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
