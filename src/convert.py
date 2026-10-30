"""Turn stored FS23K SMPL-X params into a renderer-ready rotation-matrix dict.

The SMPLFX body model is driven with ``pose2rot=False``, i.e. it expects every
pose entry as (..., 3, 3) rotation matrices rather than angles. This module does
that conversion once per clip:

    body_pose   6D       -> (T, 21, 3, 3)
    global_orient Euler  -> (T, 3, 3)   (per-clip value broadcast over T)
    jaw_pose    Euler    -> (T, 3, 3)
    left/right_hand_pose Euler -> (T, 15, 3, 3)
    leye/reye            -> identity

``expression``/``betas``/``transl`` pass straight through (optionally broadcast
to T). The result matches the dict that :class:`fs23k.body_model.SMPLXMesh`
(pip ``smplx`` ``SMPLXLayer``) consumes.
"""

from __future__ import annotations

import numpy as np

from .rotations import euler_angles_to_matrix, rotation_6d_to_matrix

_ID3 = np.eye(3, dtype=np.float32)


def _broadcast_T(x: np.ndarray, T: int) -> np.ndarray:
    """(1, D) or (D,) -> (T, D); leave (T, D) untouched."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[None]
    if x.shape[0] == 1 and T > 1:
        x = np.broadcast_to(x, (T, x.shape[1])).copy()
    return x


def smplx_to_render_params(
    smplx: dict,
    *,
    zero_global_orient: bool = True,
    zero_transl: bool = True,
) -> dict:
    """FS23K ``smplx`` record dict -> ``pose2rot=False`` matrix dict.

    By default ``global_orient`` and ``transl`` are zeroed so the figure sits in
    SMPL-X canonical space (upright, facing the camera) — which is what the
    repo's renderer camera is set up for. Pass ``zero_*=False`` to keep the
    original BOBSL camera-space orientation/placement from the fit.
    """
    body = np.asarray(smplx["body_pose"], np.float32)          # (T, 126)
    T = body.shape[0]

    body_mat = rotation_6d_to_matrix(body.reshape(T, 21, 6))   # (T, 21, 3, 3)
    jaw_mat = euler_angles_to_matrix(
        np.asarray(smplx["jaw_pose"], np.float32).reshape(T, 3))       # (T, 3, 3)
    lh_mat = euler_angles_to_matrix(
        np.asarray(smplx["left_hand_pose"], np.float32).reshape(T, 15, 3))
    rh_mat = euler_angles_to_matrix(
        np.asarray(smplx["right_hand_pose"], np.float32).reshape(T, 15, 3))

    if zero_global_orient or "global_orient" not in smplx:
        go_mat = np.broadcast_to(_ID3, (T, 3, 3)).copy()
    else:
        go = np.asarray(smplx["global_orient"], np.float32).reshape(-1, 3)[0]
        go_mat = np.broadcast_to(euler_angles_to_matrix(go), (T, 3, 3)).copy()

    if zero_transl or "transl" not in smplx:
        transl = np.zeros((T, 3), np.float32)
    else:
        transl = _broadcast_T(smplx["transl"], T)

    betas = _broadcast_T(smplx.get("betas", np.zeros((1, 10), np.float32)), T)[:, :10]
    expression = np.asarray(smplx.get("expression", np.zeros((T, 50))), np.float32)
    eye_mat = np.broadcast_to(_ID3, (T, 3, 3)).copy()

    return {
        "pose2rot": False,
        "betas": betas,                      # (T, 10)
        "transl": transl,                    # (T, 3)
        "global_orient": go_mat,             # (T, 3, 3)
        "body_pose": body_mat,               # (T, 21, 3, 3)
        "left_hand_pose": lh_mat,            # (T, 15, 3, 3)
        "right_hand_pose": rh_mat,           # (T, 15, 3, 3)
        "jaw_pose": jaw_mat,                 # (T, 3, 3)
        "leye_pose": eye_mat,                # (T, 3, 3)
        "reye_pose": eye_mat,                # (T, 3, 3)
        "expression": expression,            # (T, 50)
    }
