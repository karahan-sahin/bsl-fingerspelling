"""Render a sequence of SMPL-X meshes to frames / an MP4, using only pyrender.

Self-contained: a dark-background scene with 3-point lighting and a fixed camera
is built once, and the mesh node is swapped per frame. The sequence is centred by
its mean vertex so the figure stays put. A per-frame caption (the current
fingerspelled word) is drawn with OpenCV.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")   # headless GL

_BG = [0.0, 0.0, 0.0, 1.0]
_AMBIENT = [0.45, 0.45, 0.45]
# Camera looking down -Z at the origin (the clip is mean-centred there). Pulled
# in and raised to frame the upper body + hands, where fingerspelling happens.
_CAM_POSE = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.35],
    [0.0, 0.0, 1.0, 1.7],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float32)


def _raymond_lights():
    """3-point directional rig (PyRender's standard Raymond lights)."""
    import pyrender

    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes = []
    for theta, phi in zip(thetas, phis):
        xp, yp, zp = (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                      np.cos(theta))
        z = np.array([xp, yp, zp]);  z /= np.linalg.norm(z)
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
        x = x / np.linalg.norm(x) if np.linalg.norm(x) else np.array([1.0, 0.0, 0.0])
        y = np.cross(z, x)
        pose = np.eye(4);  pose[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=pose))
    return nodes


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    if not text:
        return img
    img = img.copy()
    cv2.putText(img, text, (16, img.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def frame_labels(words, spans, n_frames: int) -> dict[int, str]:
    """{frame -> word} from per-word [start, end] spans (clip-relative)."""
    labels: dict[int, str] = {}
    for word, (s, e) in zip(words, spans):
        for t in range(max(0, s), min(n_frames, e)):
            labels[t] = word
    return labels


def render_video(verts: np.ndarray, faces: np.ndarray, out_path: str | Path,
                 labels: dict[int, str] | None = None, image_size: int = 512,
                 fps: int = 25, yfov: float = 0.76794) -> Path:
    """(T, V, 3) vertices -> MP4 at ``out_path``. Returns the path."""
    import cv2
    import pyrender
    import trimesh

    labels = labels or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    center = verts.reshape(-1, 3).mean(0)        # one fixed offset for the clip

    scene = pyrender.Scene(bg_color=_BG, ambient_light=_AMBIENT)
    scene.add(pyrender.PerspectiveCamera(yfov=yfov), pose=_CAM_POSE)
    for node in _raymond_lights():
        scene.add_node(node)
    renderer = pyrender.OffscreenRenderer(image_size, image_size)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (image_size, image_size))

    mesh_node = None
    for t in range(verts.shape[0]):
        tm = trimesh.Trimesh(verts[t] - center, faces, process=False)
        tm.visual.vertex_colors = [200, 200, 210, 255]
        if mesh_node is not None:
            scene.remove_node(mesh_node)
        mesh_node = scene.add(pyrender.Mesh.from_trimesh(tm, smooth=True))
        color, _ = renderer.render(scene)
        img = _caption(color, labels.get(t, ""))
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    writer.release()
    renderer.delete()
    return out_path
