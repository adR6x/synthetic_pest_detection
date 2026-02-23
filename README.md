# Synthetic Data Generation for Pest Detection

A Blender-based synthetic video generator that overlays animated 3D pests (mice, rats, cockroaches) onto kitchen images using depth-aware placement, producing labeled frames for training a Vision Transformer classifier.

## Setup

Requires Python 3.12+ and Blender 3.6+ on PATH (Blender only needed for generation).

### macOS / Linux / WSL

```bash
bash setupUNIX.sh   # installs Poetry, shell plugin, and all packages
poetry shell        # activate the environment
```

### Windows (PowerShell)

```powershell
.\setupPC.ps1       # installs Poetry, shell plugin, and all packages
poetry shell        # activate the environment
```

If PowerShell blocks script execution, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setupPC.ps1
```

## Running the App

Inside `poetry shell`:

```bash
flask run
# Open http://localhost:5000
# Upload a kitchen image -> generates video + 10 labeled frames
```

## Training the Classifier

```bash
python -m training.train
# Trains a ViT classifier on generated frames in outputs/
# Prints "no data" if nothing has been generated yet
```

## Project Structure

```
├── app/
│   ├── main.py               # Flask routes: upload, generate, serve results
│   ├── templates/index.html  # Upload form, frame player, bbox overlay, previews
│   └── static/style.css
│
├── generator/
│   ├── config.py             # Render settings, pest parameters, output paths
│   ├── pipeline.py           # Orchestrator: depth estimation + Blender subprocess + MP4 assembly
│   ├── depth_estimator.py    # Metric3D v2 (depth+normals), gravity estimation, probability placement maps
│   ├── blender_script.py     # Entry point inside Blender
│   ├── scene_setup.py        # Background plane, orthographic camera, lighting (bpy)
│   ├── pest_models.py        # 3D pest geometry from UV sphere primitives (bpy)
│   ├── pest_animation.py     # Random-walk keyframe animation with mask sampling (bpy)
│   └── labeler.py            # 3D->2D bbox projection + JSON labels (bpy)
│
├── training/
│   ├── config.py             # Hyperparameters, label mapping
│   ├── dataset.py            # PyTorch Dataset (frames + JSON labels)
│   ├── model.py              # ViT-base-patch16-224 wrapper (HuggingFace)
│   └── train.py              # Training loop: AdamW, CrossEntropy, per-epoch logging
│
├── setupUNIX.sh              # One-command setup for macOS/Linux/WSL (Poetry)
├── setupPC.ps1               # One-command setup for Windows PowerShell (Poetry)
├── pyproject.toml            # Poetry dependencies
└── outputs/                  # Generated data (gitignored)
    ├── uploads/
    ├── frames/{job_id}/
    ├── videos/{job_id}.mp4
    └── labels/{job_id}/
```

## Architecture

```
User uploads kitchen.jpg
  -> Flask saves to outputs/uploads/
  -> pipeline.py:
       1. Feed-forward inference (parallel on CPU / multi-GPU, sequential on single GPU):
          a. Metric3D v2 (ViT-small) — metric depth (H x W, metres) +
                                        surface normals (H x W x 3) in one pass
          b. Gravity estimation      — camera-up vector via vanishing-point detection
       2. Probability map    — sigmoid slope + normal coherence per pest type
       3. Pest sampling      — positions drawn from probability map
       4. Blender subprocess — renders 10 frames at 640x480 via EEVEE
          -> scene_setup.py      background plane, camera, sun light
          -> pest_models.py      ellipsoid meshes
          -> pest_animation.py   random-walk keyframes + mask rejection sampling
          -> labeler.py          3D->2D bbox projection + JSON
       5. OpenCV             — assembles PNGs into MP4 at 2 FPS
  -> Flask serves video + frame gallery + depth/normal/gravity/mask previews
```

## Models

### Depth + Surface Normals — Metric3D v2 (ViT-small)
Single forward pass yields **geometrically consistent metric depth and surface normals**,
replacing the earlier two-model setup (Depth Anything V2 + Omnidata DPT-Hybrid).
**Paper:** Hu et al., *Metric3D v2: A Versatile Monocular Geometric Foundation Model
for Zero-Shot Metric Depth and Surface Normal Estimation*, IEEE TPAMI 2024.
[arXiv:2404.15506](https://arxiv.org/abs/2404.15506)
**Why chosen:** Joint depth+normal training ensures geometric consistency (important
when combining both in the placement mask). Ranks #1 on NYUv2 / KITTI over Depth
Anything and Marigold. ViT-small variant is lightweight (~22 M params).
Loaded via `torch.hub.load("YvanYin/Metric3D", "metric3d_vit_small")`.
Alternatives considered:
- *Depth Anything V2* (Yang et al., NeurIPS 2024, [arXiv:2406.09414](https://arxiv.org/abs/2406.09414)) +
  *Omnidata DPT-Hybrid* (Eftekhar et al., ICCV 2021, [arXiv:2110.04994](https://arxiv.org/abs/2110.04994)) —
  previous two-model setup; depth and normals were independently estimated and
  could be geometrically inconsistent.
- *DepthPro* (Bochkovskii et al., 2024, [arXiv:2410.02073](https://arxiv.org/abs/2410.02073)) —
  depth + focal length but no normals.
- *UniDepth v2* (Piccinelli et al., CVPR 2024 + 2025, [arXiv:2502.20110](https://arxiv.org/abs/2502.20110)) —
  depth + camera intrinsics but no normals.

### Gravity / Camera-Up Estimation — Classical Vanishing-Point Detection
Estimates the vertical vanishing point (and hence the world-up direction in camera
space) using OpenCV's Line Segment Detector (LSD) followed by RANSAC.
**LSD reference:** von Gioi et al., *LSD: A Fast Line Segment Detector with a False
Detection Control*, TPAMI 2010.
[DOI:10.1109/TPAMI.2008.300](https://doi.org/10.1109/TPAMI.2008.300)
**Why chosen:** Zero extra model weight, runs in <200 ms on CPU, robust for
rectangular indoor scenes (kitchen cabinets, door frames, counters). Falls back
to a level-camera prior ([0, 1, 0]) when the VP cannot be found reliably.
Alternatives considered:
- *PerspectiveFields* (Jin et al., CVPR 2023, [arXiv:2212.03239](https://arxiv.org/abs/2212.03239)) —
  per-pixel perspective field including gravity; very accurate but uses a
  SegFormer-MiT-B3 backbone (~47 M params) plus ConvNeXt-tiny ParamNet (~29 M params),
  which would add ~76 M parameters to an already heavy pipeline.
- *GeoCalib* (Veicht et al., ECCV 2024, [arXiv:2409.06704](https://arxiv.org/abs/2409.06704)) —
  learns intrinsics + gravity jointly via Levenberg-Marquardt optimisation on top of a
  DNN; accurate but heavyweight for this use case.
- *UprightNet* — lighter gravity estimator; lacks a maintained pip-installable release.

### Classifier — Vision Transformer (ViT-Base/16)
Trained on the generated synthetic frames to classify pest type.
**Paper:** Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image
Recognition at Scale*, ICLR 2021.
[arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

## Label Format

Per-frame JSON in `outputs/labels/{job_id}/frame_XXXX.json`:

```json
{
  "frame": 1,
  "width": 640,
  "height": 480,
  "annotations": [
    {
      "pest_type": "mouse",
      "bbox": [x_min, y_min, x_max, y_max],
      "confidence": 1.0
    }
  ]
}
```

## Classes

| ID | Label |
|----|-------|
| 0 | background |
| 1 | mouse |
| 2 | rat |
| 3 | cockroach |
