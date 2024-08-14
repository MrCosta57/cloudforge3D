# CloudForge3D

## Installation
```bash
# [OPTIONAL] create conda environment
conda create -n myenv python=3.11
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## Camera calibration
```bash
python camera_calibration/src/calibrate.py
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for calibrate.py</span></summary>
#### --source_path / -s
Path to the source directory containing a COLMAP or Synthetic NeRF data set.
</details>