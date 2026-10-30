#!/usr/bin/env python3
"""Render the SMPL-X mesh of one FS23K clip to an MP4, labelled with its FS words.

Fully standalone — uses only the pip ``smplx`` + ``pyrender`` packages (no
external project). The clip's stored params are converted to the SMPL-X layer's
rotation-matrix format by :func:`fs23k.smplx_to_render_params`, forwarded to
vertices by :class:`fs23k.SMPLXMesh`, and rendered by :func:`fs23k.render_video`.
The fingerspelled word is overlaid on the frames where it occurs (``fs_spans``).

Needs an env with ``smplx``, ``pyrender``, ``trimesh``, ``torch``, ``opencv`` and
the SMPL-X model file (``<dir>/smplx/SMPLX_NEUTRAL.npz``).

Examples:
    python visualize_mesh.py --lmdb ... --clip 0 --out out/clip0.mp4
    python visualize_mesh.py --lmdb ... --clip 105 --device cuda:0
    python visualize_mesh.py --lmdb ... --clip 105 --model-dir /path/to/body_models
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import FS23KDataset, smplx_to_render_params
from src.body_model import SMPLXMesh
from src.render import frame_labels, render_video

_DEFAULT_LMDB = "/media/ks0085/storage/ks0085/datasets/BSL/BOBSL/bobsl_fs23k_train.lmdb"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmdb", default=_DEFAULT_LMDB)
    ap.add_argument("--clip", default="0", help="clip index (int) or clip id")
    ap.add_argument("--out", type=Path, default=Path("out/clip.mp4"))
    ap.add_argument("--device", default="cpu", help="cpu or cuda:N")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--model-dir", default=None,
                    help="dir containing smplx/SMPLX_NEUTRAL.npz (else $SMPLX_MODEL_DIR)")
    ap.add_argument("--keep-global-orient", action="store_true",
                    help="keep the fit's camera-space orientation (default: canonical)")
    ap.add_argument("--keep-transl", action="store_true",
                    help="keep the fit's translation (default: centred)")
    ap.add_argument("--neutral-face", action="store_true",
                    help="zero the FLAME shape/eyelids (default: use the fit's)")
    args = ap.parse_args()

    with FS23KDataset(args.lmdb) as ds:
        cid = ds.clip_ids[int(args.clip)] if args.clip.isdigit() else args.clip
        rec = ds[cid]

    print(f"Rendering {cid}: {rec['num_frames']} frames, "
          f"fs_words={rec['fs_words']}, text={rec['text']!r}")

    params = smplx_to_render_params(
        rec["smplx"],
        zero_global_orient=not args.keep_global_orient,
        zero_transl=not args.keep_transl,
    )

    mesh = SMPLXMesh(model_dir=args.model_dir, device=args.device)
    verts = mesh.vertices(
        params,
        shape=None if args.neutral_face else rec["smplx"].get("shape"),
        eyelid_params=None if args.neutral_face else rec["smplx"].get("eyelid_params"),
    )

    labels = frame_labels(rec["fs_words"], rec["fs_spans"], rec["num_frames"])
    out = render_video(verts, mesh.faces, args.out, labels=labels,
                       image_size=args.image_size, fps=args.fps)
    print(f"Saved mesh video -> {out}")


if __name__ == "__main__":
    main()
