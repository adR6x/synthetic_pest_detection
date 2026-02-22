# Synthetic Data Generation for Pest Detection

A Blender-based synthetic video generator that overlays animated 3D pests (mice, rats, cockroaches) onto kitchen images, producing labeled video frames for training a Vision Transformer classifier.

## Architecture

```
User uploads kitchen.jpg
  -> Flask saves to outputs/uploads/
  -> pipeline.py invokes Blender as subprocess
     -> blender_script.py orchestrates:
        1. scene_setup.py   — background plane, orthographic camera, sun light
        2. pest_models.py   — ellipsoid meshes from UV sphere primitives
        3. pest_animation.py — keyframed random-walk scurrying
        4. labeler.py       — 3D->2D bounding box projection + JSON
        5. Renders 10 frames at 640x480 via EEVEE
  -> pipeline.py assembles PNGs into MP4 (OpenCV)
  -> Flask serves video + frame gallery

Training pipeline:
  outputs/frames/*.png + outputs/labels/*.json
  -> PestDetectionDataset (resize 640x480 -> 224x224, ImageNet normalize)
  -> ViTForImageClassification (google/vit-base-patch16-224, 4 classes)
  -> CrossEntropy + AdamW
```

## Image Size Pipeline

| Stage | Size | Format |
|-------|------|--------|
| Generator output | 640x480 | PNG |
| Bounding box coords | 640x480 pixel space | JSON |
| Training input (after resize) | 224x224 | Tensor |
| ViT patches | 14x14 grid of 16x16 patches | Internal |

## Project Structure

```
├── generator/              # Blender-based synthetic data generation
│   ├── config.py           # Resolution, FPS, pest parameters (no bpy)
│   ├── pipeline.py         # Orchestrator: Blender subprocess + MP4 assembly
│   ├── blender_script.py   # Blender entry point (runs inside Blender)
│   ├── scene_setup.py      # Background plane, camera, lighting (bpy)
│   ├── pest_models.py      # 3D pest geometry from primitives (bpy)
│   ├── pest_animation.py   # Random-walk keyframe animation (bpy)
│   └── labeler.py          # 3D->2D bbox projection + JSON (bpy)
│
├── app/                    # Flask web application
│   ├── main.py             # Routes: upload, generate, serve outputs
│   ├── templates/index.html
│   └── static/style.css
│
├── training/               # Training pipeline
│   ├── config.py           # Hyperparameters, label mapping
│   ├── dataset.py          # PyTorch Dataset (frames + JSON labels)
│   ├── model.py            # ViT wrapper (HuggingFace)
│   └── train.py            # Training loop
│
└── outputs/                # Generated data (gitignored)
    ├── uploads/
    ├── frames/{job_id}/
    ├── videos/{job_id}.mp4
    └── labels/{job_id}/
```

## Quickstart for Teammates

### 1. Clone and switch to the working branch

```bash
git clone git@github.com:Mirsaid-ai/Synthetic-Data-Generation-for-Pest-Detection.git synthetic_data_gen_pest
cd synthetic_data_gen_pest
git checkout anubhav_v1
```

Or if you already have the repo cloned:

```bash
cd synthetic_data_gen_pest
git fetch origin
git checkout anubhav_v1
git pull
```

### 2. Create your own branch (to avoid stepping on each other)

```bash
git checkout -b yourname_v1
git push --set-upstream origin yourname_v1
```

To pull in latest changes from `anubhav_v1` into your branch later:

```bash
git fetch origin
git merge origin/anubhav_v1 -m "merge latest from anubhav_v1"
```

### 3. Install dependencies

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Duke Cluster (Singularity):**
```bash
singularity shell --nv --bind /work:/work,/cwork:/cwork /opt/apps/containers/oit/jupyter/courses-jupyter-cuda.sif
pip install --user -r requirements.txt
```

### 4. Install Blender (needed for video generation only)

- Download from https://www.blender.org/download/ (version 3.6+)
- Make sure `blender` is on your PATH:
  - **Windows:** Add the Blender install folder (e.g. `C:\Program Files\Blender Foundation\Blender 4.0`) to your system PATH
  - **Mac:** `brew install --cask blender` or add `/Applications/Blender.app/Contents/MacOS` to PATH
  - **Linux:** `sudo apt install blender` or `sudo snap install blender --classic`
- Verify: `blender --version`

> **Note:** Blender is only needed for the generation step. Training works without it.

### 5. Run the web app

```bash
python -m app.main
# Open http://localhost:5000
# Upload any kitchen image -> generates video + 10 labeled frames
```

### 6. Train the classifier

```bash
python -m training.train
# Loads generated frames from outputs/, trains ViT classifier
# (prints "no data" message if nothing generated yet)
```

## Prerequisites (summary)

- Python 3.10+
- Blender 3.6+ installed and `blender` on PATH (for generation only)
- `pip install -r requirements.txt`

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
