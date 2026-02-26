# Synthetic Data Generation for Pest Detection

A Blender-based synthetic video generator that overlays animated 3D pests (mice, rats, cockroaches) onto kitchen images using depth-aware placement, producing labeled frames for training a Vision Transformer classifier.

## Setup

Requires Python 3.12+ and Blender 3.6+ on PATH (Blender only needed for generation).

### macOS / Linux / WSL

```bash
bash setupUNIX.sh   # installs Poetry, shell plugin, all packages, and mmcv stub
poetry shell        # activate the environment
```

### Windows (PowerShell)

```powershell
.\setupPC.ps1       # installs Poetry, shell plugin, all packages, and mmcv stub
poetry shell        # activate the environment
```

If PowerShell blocks script execution, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setupPC.ps1
```

> **Note on mmcv:** Metric3D v2 internally imports `mmcv` (OpenMMLab), which has
> no pre-built wheels for PyTorch 2.7 + Python 3.12. The setup scripts install a
> minimal `mmcv_stub/` shim that delegates the two symbols used at inference time
> (`Config`, `DictAction`) to `mmengine`, the official mmcv successor. No separate
> `mmcv` installation is needed.

## Running the App

Inside `poetry shell`:

```bash
flask run
# Open http://localhost:5000
# Upload a kitchen image -> generates video + 100 labeled frames
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
│   ├── pest_models.py        # 3D pest geometry: imported .obj/.glb or UV-sphere fallback (bpy)
│   ├── pest_animation.py     # Random-walk keyframe animation, movement-aligned rotation (bpy)
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
├── mmcv_stub/                # Minimal mmcv shim (delegates to mmengine; see Setup note)
└── outputs/                  # Generated data (gitignored)
    ├── uploads/
    ├── frames/{job_id}/
    ├── videos/{job_id}.mp4
    └── labels/{job_id}/annotations.json   # COCO-format, all 100 frames
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
       4. Depth-based scale  — blender_scale = real_size × fx × PLANE_WIDTH / (depth × RENDER_WIDTH)
       5. Blender subprocess — renders 100 frames at 640x480 via EEVEE
          -> scene_setup.py      background plane, camera, sun light
          -> pest_models.py      load_pest(): import .obj/.glb or fallback to ellipsoid, apply scale
          -> pest_animation.py   random-walk keyframes + heading aligned to movement direction
          -> labeler.py          3D->2D bbox projection + JSON
       6. ffmpeg (H.264)     — assembles PNGs into MP4 at 10 FPS; OpenCV mp4v as fallback
  -> Flask serves video + frame gallery + depth/normal/gravity/mask previews
```

## 3D Pest Models

By default the generator uses procedural geometry (UV-sphere ellipsoids) for all pest types.
You can replace these with downloaded 3D models for more realistic renders.

### Obtaining free models

| Source | License | Notes |
|--------|---------|-------|
| [Sketchfab – Free downloads](https://sketchfab.com/features/free-3d-models) | CC (varies per asset) | Filter by "Downloadable", export as `.glb` |
| [Blendswap](https://www.blendswap.com) | CC0 / CC-BY | `.blend` files; re-export as `.obj` from Blender |
| [Smithsonian 3D Digitization](https://3d.si.edu) | CC0 | High-accuracy photogrammetry scans of real insects |
| [Free3D](https://free3d.com) | Free (check per asset) | Mixed formats; `.obj` works best |

Recommended search terms: `"cockroach 3d model"`, `"house mouse 3d"`, `"rat low poly"`.
Prefer low-poly (< 10k faces) models for fast EEVEE rendering.

### Installing a model

1. Download the model as `.obj` or `.glb`.
2. Place it in `generator/models/` (create the directory if needed).
3. Open `generator/config.py` and set the path:
   ```python
   PEST_MODEL_PATHS = {
       "cockroach": "generator/models/cockroach.glb",
       "mouse":     "generator/models/mouse.obj",
       "rat":       None,   # still uses procedural
   }
   ```
4. Open the model once in Blender GUI to check which local axis points toward the head,
   then set `PEST_FORWARD_AXIS` accordingly (default `"X"`; other options: `"-X"`, `"Y"`, `"-Y"`).

If `model_path` is `None` or the file is missing, the pipeline falls back to the
procedural ellipsoid automatically.

### Depth-based size scaling

Each pest is scaled so it appears at its correct physical size given the estimated depth
of the scene at the placement pixel.  The formula is:

```
blender_scale = real_body_length_m × fx × PLANE_WIDTH
                ─────────────────────────────────────────
                      depth_at_placement_m × RENDER_WIDTH
```

where:
- `real_body_length_m` — physical body length from `PEST_REAL_SIZES_M` in `config.py`
- `fx` — estimated horizontal focal length in pixels (output of Metric3D v2 preprocessing)
- `depth_at_placement_m` — metric depth at the pest's placement pixel (Metric3D v2)
- `PLANE_WIDTH / RENDER_WIDTH` — world-units-per-pixel ratio of the orthographic render

The result is clamped to [0.4×, 3×] of the procedural default to guard against
unreliable depth estimates at image borders or textureless regions.

## Models

### Depth + Surface Normals — Metric3D v2 (ViT-small)
A **single forward pass** yields both geometrically consistent metric depth
(H × W, metres) and surface normals (H × W × 3), replacing the earlier
two-model setup (Depth Anything V2 + Omnidata DPT-Hybrid). Both the
*Predicted Depth* and *Surface Normals* previews shown on the results page
come entirely from this one model.

**Paper:** Hu et al., *Metric3D v2: A Versatile Monocular Geometric Foundation Model
for Zero-Shot Metric Depth and Surface Normal Estimation*, IEEE TPAMI 2024.
[arXiv:2404.15506](https://arxiv.org/abs/2404.15506)

**Why chosen:** Joint depth+normal training ensures geometric consistency (important
when combining both in the placement mask). Ranks #1 on NYUv2 / KITTI over Depth
Anything and Marigold. ViT-small variant is lightweight (~22 M params).
Loaded via `torch.hub.load("YvanYin/Metric3D", "metric3d_vit_small")`.

**CPU compatibility:** Metric3D's RAFT decoder (`RAFTDepthNormalDPTDecoder5`) has
`device='cuda'` hardcoded in `get_bins()` and `create_mesh_grid()`. At load time
we monkey-patch these methods to use `next(self.parameters()).device` instead, so
the model runs correctly on CPU when no GPU is available.

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

A single COCO-format JSON at `outputs/labels/{job_id}/annotations.json`:

```json
{
  "images": [
    {"id": 1, "file_name": "frame_0001.png", "width": 640, "height": 480},
    ...
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": width * height,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "mouse",      "supercategory": "pest"},
    {"id": 2, "name": "rat",        "supercategory": "pest"},
    {"id": 3, "name": "cockroach",  "supercategory": "pest"}
  ]
}
```

Note: `bbox` follows COCO convention `[x, y, width, height]` (top-left origin), not `[x_min, y_min, x_max, y_max]`.

## Classes (ViT Classifier)

Label mapping used by `training/config.py` (different from COCO category IDs above, which start at 1):

| ID | Label |
|----|-------|
| 0 | background |
| 1 | mouse |
| 2 | rat |
| 3 | cockroach |
