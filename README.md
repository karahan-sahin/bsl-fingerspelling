# BSL Fingerspelling (BOBSL FS23K)

Minimal tools to read and visualize the **BOBSL FS23K** fingerspelling LMDBs. Each
clip is **one fingerspelling interval** — the SMPL-X params sliced to just the
frames where a word is fingerspelled (the clip *is* the fingerspelling
subsequence, not the whole subtitle segment) — labelled with that word and the
parent subtitle.

## Splits

Episode-level, no leakage (train and test share no episodes):

| LMDB                    | source                                   | clips  | episodes | words |
|-------------------------|------------------------------------------|--------|----------|-------|
| `bobsl_fs23k_train.lmdb`| FS23K `FSK23letters.json` (automatic)    | 20,813 | 1,494    | 8,334 |
| `bobsl_fs23k_test.lmdb` | manual `fingerspelling_test.csv`         | 6,617  | 250      | 3,459 |

(FS23K has no episodes in BOBSL's official `test`/`challenge_test`; the manual
fingerspelling test CSV is the dedicated held-out test set.)

Data I/O needs only `lmdb`/`numpy`. The mesh renderer uses the pip `sltk`
(`signlangtk`) **SMPLFX** body model — the same model the clips were fitted with,
so faces/eyelids stay faithful — plus `pyrender` for offscreen rendering.

## Layout

```
src/
  dataset.py     FS23KDataset — read-only LMDB view (stats, clips)
  rotations.py   6D / Euler-ZYX -> matrix helpers (NumPy, no deps)
  convert.py     stored SMPL-X params -> SMPLXLayer rotation-matrix dict
  body_model.py  SMPLXMesh — pip sltk SMPLFX forward -> (T, V, 3) vertices
  render.py      pyrender offscreen render -> captioned MP4
visualize_dataset.py   print stats, dump a clip, plot distributions
visualize_mesh.py      render one clip's SMPL-X mesh to an MP4
```

## Record schema

Each clip (`<episode>.seg<NNNN>.<startframe>_<endframe>`) stores:

| field        | meaning |
|--------------|---------|
| `smplx`      | sliced to the FS interval: `body_pose (T,126)` 6D · `left/right_hand_pose (T,45)` Euler ZYX · `jaw_pose (T,3)` · `expression (T,50)` · `eyelid_params (T,2)` — plus per-clip `global_orient (1,3)` · `betas (1,10)` · `shape (1,300)` · `transl (1,3)` · camera `K`/`RTs` |
| `word`       | the fingerspelled word (train: FS23K `predword`; test: `corresponding_word`) |
| `letters`    | letters actually produced (test CSV `fingerspelled_letters`); `None` for train |
| `text`       | parent subtitle text |
| `fs_words`   | `[word]` · `fs_spans` `[[0, T]]` (the clip is the fingerspelling) |
| `video`      | episode id · `order` absolute start frame @25fps · `num_frames` T |

Poses are stored exactly as the SMPLFX fit produced them: **body in rotation-6D,
hands/jaw/global_orient as Euler ZYX**. `convert.smplx_to_render_params` turns
them into the `pose2rot=False` rotation-matrix dict the renderer expects.

## Environments

- **Dataset viz** (`visualize_dataset.py`): just `lmdb`, `numpy`, `matplotlib`.
- **Mesh viz** (`visualize_mesh.py`): `sltk` (`pip install signlangtk`),
  `pyrender`, `trimesh`, `torch`, `opencv-python` + the SMPL-X/FLAME weights
  (`sltk download-weights`). Conda **`base-v1`** has all of these.

The renderer rebuilds the exact **SMPLFX** model the FS23K clips were fitted with
(SMPL-X + FLAME face + eyelids; `use_pca=False`, `flat_hand_mean=True`,
`use_euler_angles=True`, 50 expression + 10 body + 300 FLAME betas), driven with
`pose2rot=False` rotation matrices. Model/FLAME files are resolved via sltk's
weight registry (env overrides `SLTK_SMPLX_MODEL` etc.); `--model-dir` can point
at a directory containing `smplx/SMPLX_NEUTRAL.npz`. `--neutral-face` zeros the
FLAME shape/eyelids.

## Usage

```bash
conda activate base-v1
LMDB=/media/ks0085/storage/ks0085/datasets/BSL/BOBSL/bobsl_fs23k_train.lmdb   # or _test

# stats + a detailed clip dump
python visualize_dataset.py --lmdb $LMDB --clip 0

# save distribution plots (clip length, Zipf, top words)
python visualize_dataset.py --lmdb $LMDB --plot out/fs23k_stats.png

# render clip 105's mesh (captioned with the FS word) to MP4
python visualize_mesh.py --lmdb $LMDB --clip 105 --out out/clip105.mp4
# ...on GPU, keeping the fit's original camera orientation:
python visualize_mesh.py --lmdb $LMDB --clip 105 --device cuda:0 --keep-global-orient
```

`--clip` accepts an integer index or a clip id (e.g. `5093160768934007504.seg0794`).
By default the mesh is rendered in SMPL-X canonical space (upright, facing the
camera); pass `--keep-global-orient` / `--keep-transl` for the original fit pose.

## Programmatic use

```python
from fs23k import FS23KDataset, smplx_to_render_params

with FS23KDataset(LMDB) as ds:
    print(ds.summary())           # {'clips': 20206, 'episodes': 1494, ...}
    rec = ds[0]                   # or ds["<episode>.seg0021"]
    params = smplx_to_render_params(rec["smplx"])   # pose2rot=False matrices
```
