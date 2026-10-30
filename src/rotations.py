"""Small, dependency-free rotation helpers (NumPy).

The FS23K LMDB stores SMPL-X poses exactly as the SMPLFX fit produced them:

    body_pose        (T, 126)   rotation-6D, 21 joints  (IS_BODY_POSE_6D)
    global_orient    (1 or T,3) Euler ZYX
    jaw_pose         (T, 3)     Euler ZYX
    left_hand_pose   (T, 45)    Euler ZYX  (15 MANO joints x 3)
    right_hand_pose  (T, 45)    Euler ZYX

The SMPL-X body model consumes rotation *matrices* (``pose2rot=False``), so we
convert both encodings to (..., 3, 3) matrices here. Conventions follow PyTorch3D
(Gram-Schmidt for 6D; intrinsic Euler ZYX, matching SMPLFX ``use_euler_angles``),
so meshes reproduce the original fit.
"""

from __future__ import annotations

import numpy as np


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """(..., 6) rotation-6D -> (..., 3, 3) matrix (Zhou et al., Gram-Schmidt).

    The two 3-vectors are the first two rows of the matrix; the third row is
    their cross product. Stacked as rows, matching ``matrix_to_rotation_6d``.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def _axis_rotation(axis: str, angle: np.ndarray) -> np.ndarray:
    """Elementary rotation about one axis -> (..., 3, 3)."""
    c, s = np.cos(angle), np.sin(angle)
    one, zero = np.ones_like(angle), np.zeros_like(angle)
    if axis == "X":
        flat = (one, zero, zero, zero, c, -s, zero, s, c)
    elif axis == "Y":
        flat = (c, zero, s, zero, one, zero, -s, zero, c)
    elif axis == "Z":
        flat = (c, -s, zero, s, c, zero, zero, zero, one)
    else:
        raise ValueError(f"invalid axis {axis!r}")
    return np.stack(flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler: np.ndarray, convention: str = "ZYX") -> np.ndarray:
    """(..., 3) Euler angles -> (..., 3, 3). Default intrinsic ZYX (SMPLFX)."""
    mats = [_axis_rotation(ax, euler[..., i]) for i, ax in enumerate(convention)]
    out = mats[0]
    for m in mats[1:]:
        out = out @ m
    return out
