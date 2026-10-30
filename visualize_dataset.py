#!/usr/bin/env python3
"""Inspect the BOBSL FS23K LMDB: print stats, dump a clip, optionally plot.

Examples:
    python visualize_dataset.py --lmdb /media/.../BOBSL/bobsl_fs23k_train.lmdb
    python visualize_dataset.py --lmdb ... --clip 0
    python visualize_dataset.py --lmdb ... --plot out/fs23k_stats.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src import FS23KDataset

_DEFAULT_LMDB = "/media/ks0085/storage/ks0085/datasets/BSL/BOBSL/bobsl_fs23k_train.lmdb"


def print_summary(ds: FS23KDataset) -> None:
    print("=== FS23K dataset summary ===")
    for k, v in ds.summary().items():
        print(f"  {k:14s}: {v}")
    print(f"\n  top fingerspelled words:")
    for word, n in ds.top_words(20):
        print(f"    {word:20s} {n}")


def dump_clip(ds: FS23KDataset, ref: str) -> dict:
    rec = ds[int(ref)] if ref.isdigit() else ds[ref]
    cid = ds.clip_ids[int(ref)] if ref.isdigit() else ref
    print(f"\n=== clip {cid} ===")
    print(f"  video      : {rec['video']}")
    print(f"  order      : {rec['order']}  (start frame @25fps)")
    print(f"  num_frames : {rec['num_frames']}")
    print(f"  text       : {rec['text']!r}")
    print(f"  fs_words   : {rec['fs_words']}")
    print(f"  fs_spans   : {rec['fs_spans']}  (clip-relative frames)")
    print(f"  smplx arrays:")
    for k, v in rec["smplx"].items():
        print(f"    {k:16s} {tuple(v.shape)} {v.dtype}")
    return rec


def make_plot(ds: FS23KDataset, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = ds.meta.get("fs_counts", {})
    word_freq = np.array(sorted(counts.values(), reverse=True))

    # per-clip length distribution (sample the first N records for speed)
    n_frames = np.array([ds[i]["num_frames"] for i in range(min(len(ds), 2000))])

    top = ds.top_words(20)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].hist(n_frames, bins=40, color="#4C72B0")
    ax[0].set(title=f"Clip length (frames, n={len(n_frames)} sampled)",
              xlabel="frames", ylabel="clips")

    ax[1].loglog(np.arange(1, len(word_freq) + 1), word_freq, color="#C44E52")
    ax[1].set(title="FS word frequency (Zipf)", xlabel="rank", ylabel="count")

    words, freqs = zip(*top)
    ax[2].barh(range(len(words)), freqs, color="#55A868")
    ax[2].set_yticks(range(len(words)))
    ax[2].set_yticklabels(words, fontsize=8)
    ax[2].invert_yaxis()
    ax[2].set(title="Top 20 fingerspelled words", xlabel="count")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved plot -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmdb", default=_DEFAULT_LMDB)
    ap.add_argument("--clip", help="clip index (int) or id to dump in detail")
    ap.add_argument("--plot", type=Path, help="save a stats PNG to this path")
    args = ap.parse_args()

    with FS23KDataset(args.lmdb) as ds:
        print_summary(ds)
        if args.clip is not None:
            dump_clip(ds, args.clip)
        if args.plot:
            make_plot(ds, args.plot)


if __name__ == "__main__":
    main()
