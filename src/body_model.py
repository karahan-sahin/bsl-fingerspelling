"""SMPLFX forward pass -> mesh vertices, via the pip ``sltk`` (``signlangtk``) package.

The FS23K clips were fitted with **SMPLFX** (SMPL-X + FLAME face shape + eyelids),
so we reconstruct meshes with the same model rather than vanilla SMPL-X — this
keeps the face/eyelids/head identity faithful to the fit.

We build it exactly as the fit did (``use_pca=False``, ``flat_hand_mean=True``,
``use_euler_angles=True``, 50 expression + 10 body betas + 300 FLAME betas) and
drive it with ``pose2rot=False`` rotation matrices (the only safe format for a
``use_euler_angles=True`` model — the Euler path would misread matrices). Model
and FLAME asset files are resolved through sltk's weight registry (env overrides:
``SLTK_SMPLX_MODEL`` etc.).

Install: ``pip install signlangtk`` (imported as ``sltk``) + its weights
(``sltk download-weights``).
"""

from __future__ import annotations

import numpy as np

_NUM_EXPRESSION = 50
_NUM_BETAS = 10
_NUM_BETAS_FLAME = 300


def _build_smplfx(device):
    """Construct the neutral SMPLFX model the FS23K fits used."""
    from sltk.extraction.smplfx_deps.smpl_fx.body_models import SMPLFX
    from sltk.extraction.smplfx_deps.utils_smplx import convertSMPLX2SMPLFXKwargs
    from sltk.extraction.weights import get_weight_path

    smplx_npz = get_weight_path("smplx_model")          # .../body_models/smplx/SMPLX_NEUTRAL.npz
    if smplx_npz is None:
        raise FileNotFoundError(
            "SMPL-X model not found. Run `sltk download-weights` or set $SLTK_SMPLX_MODEL.")
    model_path = str(smplx_npz.parent.parent)           # .../body_models/
    masks = get_weight_path("smplfx_flame_masks")
    corr = get_weight_path("smplfx_flame_corr")

    kwargs = convertSMPLX2SMPLFXKwargs(
        {
            "model_path": model_path,
            "model_type": "smplx",
            "gender": "neutral",
            "use_face_contour": False,       # vertices are identical; skips an extra asset
            "num_betas": _NUM_BETAS,
            "num_expression_coeffs": _NUM_EXPRESSION,
            "use_pca": False,
            "ext": "npz",
            "batch_size": 1,
            "flat_hand_mean": True,
            "use_euler_angles": True,
            "gender_flame": "generic",
            "ext_flame": "pkl",
        },
        num_betas_flame=_NUM_BETAS_FLAME,
        v_mask_path_flame=str(masks) if masks else None,
        corr_fname=str(corr) if corr else None,
    )
    return SMPLFX(**kwargs).to(device).eval()


class SMPLXMesh:
    """Build the SMPLFX model once; forward render-params -> vertices."""

    def __init__(self, model_dir: str | None = None, device: str = "cpu"):
        import torch

        if model_dir:                                    # optional explicit override
            import os
            os.environ.setdefault("SLTK_SMPLX_MODEL",
                                  os.path.join(model_dir, "smplx", "SMPLX_NEUTRAL.npz"))
        self.torch = torch
        self.device = torch.device(device)
        self.model = _build_smplfx(self.device)
        self.faces = np.asarray(self.model.faces, dtype=np.int32)

    def vertices(self, params: dict, shape: np.ndarray | None = None,
                 eyelid_params: np.ndarray | None = None,
                 batch_size: int = 64) -> np.ndarray:
        """``pose2rot=False`` matrix dict -> (T, V, 3) vertices.

        ``params`` holds the keys from ``smplx_to_render_params`` (rotation
        matrices + betas/expression/transl). ``shape`` (FLAME, (1 or T, 300)) and
        ``eyelid_params`` ((1 or T, 2)) are optional face extras; pass the clip's
        own for a faithful head, or ``None`` for a neutral face.
        """
        torch = self.torch
        pose_keys = ["global_orient", "body_pose", "left_hand_pose",
                     "right_hand_pose", "jaw_pose", "leye_pose", "reye_pose",
                     "betas", "expression", "transl"]
        T = params["body_pose"].shape[0]

        def _to_T(x):
            x = np.asarray(x, np.float32)
            if x.ndim == 1:
                x = x[None]
            return np.broadcast_to(x, (T, *x.shape[1:])).copy() if x.shape[0] == 1 else x

        tens = {k: torch.from_numpy(np.asarray(params[k], np.float32)).to(self.device)
                for k in pose_keys if k in params}
        if shape is not None:
            tens["shape"] = torch.from_numpy(_to_T(shape)[:, :_NUM_BETAS_FLAME]).to(self.device)
        if eyelid_params is not None:
            tens["eyelid_params"] = torch.from_numpy(_to_T(eyelid_params)).to(self.device)

        out = []
        for i in range(0, T, batch_size):
            batch = {k: v[i:i + batch_size] for k, v in tens.items()}
            with torch.no_grad():
                res = self.model(**batch, pose2rot=False, return_verts=True)
            out.append(res.vertices.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)
