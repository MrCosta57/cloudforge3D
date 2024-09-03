# CloudForge3D

## Description
This repository contains the implementation of a simple 3D laser scanner using off-the-shelf components. The scanner consists of a motorized turntable with a custom-designed fiducial marker and a planar marker placed behind it. The system projects a laser line on the target object, capturing the scene with a camera multiple times during rotation.<br/>
The output is a 3D point cloud representing the object's geometry.

The repository also includes code for camera calibration using a video file with a calibration chessboard marker.


## Installation
```bash
# [OPTIONAL] create conda environment
conda create -n myenv python=3.11
conda activate myenv

# install requirements
pip install -r requirements.txt
```

## Project structure
```bash
project-folder/
├── camera_calibration/
│   ├── data/
│       └── markers/
|   ├── output/
│   └── src/
├── point_cloud_generation/
│   ├── data/
│       └── markers/
|   ├── output/
│   └── src/
|── utils/
|── README.md
└── requirements.txt
```


## Camera calibration
1. Put the video file for camera calibration in the folder `camera_calibration/data` directory. The video should contain the calibration chessboard marker
1. Run the following command to calibrate the camera
    ```bash
    python camera_calibration/src/calibrate.py
    ```
1. The camera calibration parameters will be saved in the `camera_calibration/output` directory

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for calibrate.py</span></summary>

  #### --debug
  Debug mode (display additional information)
  #### --chessboard_size
  Size of the chessboard
  #### --window_scaling_factor
  Scaling factor for the window size
  #### --time_skip
  Time interval to skip in milliseconds
  #### --video_dir
  Directory containing the video file
  #### --video_name
  Name of the video file
  #### --output_dir
  Directory to save the output file
  #### --output_name
  Name of the output file
</details>


## Point cloud generation
1. Put the video file for point cloud generation in the folder `point_cloud_generation/data` directory. The video should contain the back marker and the plate marker for a correct program execution
1. Run the following command to generate the point cloud
    ```bash
    python point_cloud_generation/src/generate.py
    ```
1. The point cloud will be saved in the `point_cloud_generation/output` directory
<br/>

<p>
If debug mode is enabled, the program will display different windows:
<ol>
    <li><b>Shapes frame</b>: provide more information on the object contours and the geometric shapes fitted during the video processing</li>
    <li><b>Projection frame</b>: shows the data related to point projections, for checking the correctness of markers' coordinate systems and their extrinsic parameter identification</li>
    <li><b>Laser frame</b>: contains all the points detected by the procedure for fitting the laser plane</li>
</ol>

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for generate.py</span></summary>

  #### --debug
  Debug mode (display additional information)
  #### --window_scaling_factor
  Scaling factor for the window size
  #### --video_dir
  Directory containing the video file
  #### --video_name
  Name of the video file
  #### --output_dir
  Directory to save the output file
  #### --camera_params_dir
  Directory containing the camera parameters file
  #### --camera_params_name
  Name of the camera parameters file
  #### --plate_marker_info
  Information about the object marker (seq_string, min_pattern_len, marker_radius_cm)
  #### --back_marker_size
  Size of the back marker (width_cm, height_cm)
</details>