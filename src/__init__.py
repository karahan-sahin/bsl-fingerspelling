"""BOBSL FS23K fingerspelling dataset: LMDB reader, param conversion, viz.

Data I/O needs only ``lmdb``/``numpy``; the mesh renderer uses the pip ``sltk``
(``signlangtk``) SMPLFX model + ``pyrender``/``trimesh``/``torch``/``opencv``.
"""

from .dataset import FS23KDataset
from .convert import smplx_to_render_params

__all__ = ["FS23KDataset", "smplx_to_render_params"]
