import os, sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)
import argparse, json
import numpy as np
import cv2
from termcolor import colored
from utils.general_utils import *
from laser import *
from utils.scanner_utils import *
from back_marker import *
from plate_marker import *


def main(args: argparse.Namespace):
    print("***Point cloud generation script***")
    debug = args.debug
    print(colored("Debug mode is on", "yellow") if debug else "")
    plate_marker_info = args.plate_marker_info
    back_marker_size = args.back_marker_size
    window_scaling_factor = args.window_scaling_factor
    camera_params_path = os.path.join(args.camera_params_dir, args.camera_params_name)
    video_path = os.path.join(args.video_dir, args.video_name)
    output_path = os.path.join(args.output_dir, args.output_name)

    cap = cv2.VideoCapture(video_path)
    # Sanity checks
    if not cap.isOpened():
        print(colored("Error opening video file.", "red"))
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert height > width, "Video frame is not in portrait mode"
    camera_matrix = None
    dist_coeffs = None
    with open(camera_params_path, "r") as f:
        camera_params = json.load(f)
        camera_matrix = np.array(camera_params["camera_matrix"])
        dist_coeffs = np.array(camera_params["distortion_coefficients"])
        error = camera_params["total_error"]
        print(
            colored(
                f"Camera parameters loaded successfully",
                "green",
            )
        )
        print(colored(f"Calibration error: {error:.4f}", "dark_grey"))

    while True:
        ret, original_frame = cap.read()
        if not ret:
            break
        original_frame = get_undistorted_frame(
            frame=original_frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs
        )
        palette_frame = original_frame.copy()

        gray_frame = find_black_objects(original_frame)
        gray_contours, _ = cv2.findContours(
            gray_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        rectangle = fit_marker_rectangle(gray_contours)
        compute_back_marker_extrinsic(
            rectangle,
            camera_matrix,
            dist_coeffs,
            back_marker_size,
            palette_frame,
        )

        dot_centers = find_plate_marker_cand_dot_centers(gray_contours, width, height)
        ellipse = fit_marker_ellipse(dot_centers)
        compute_plate_marker_extrinsic(
            ellipse,
            dot_centers,
            camera_matrix,
            dist_coeffs,
            plate_marker_info,
            original_frame,
            palette_frame,
        )

        original_frame_resized = get_resized_frame(
            original_frame,
            width=width,
            height=height,
            scaling_factor=window_scaling_factor,
        )
        palette_frame_resized = get_resized_frame(
            palette_frame,
            width=width,
            height=height,
            scaling_factor=window_scaling_factor,
        )

        cv2.imshow("Original frame", original_frame_resized)
        cv2.imshow("Palette frame", palette_frame_resized)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            print(colored("User interrupted the process", "red"))
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that perform point cloud generation from a provided video"
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=True,
        help="Enable debug mode (print additional information)",
    )
    parser.add_argument(
        "--window_scaling_factor",
        type=float,
        default=0.4,
        help="Scaling factor for the window size",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="point_cloud_generation/data",
        help="Directory containing the video file",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="cube.mov",
        help="Name of the video file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="point_cloud_generation/output",
        help="Directory to save the output file",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="camera_params.json",
        help="Name of the output file",
    )
    parser.add_argument(
        "--camera_params_dir",
        type=str,
        default="camera_calibration/output",
        help="Directory containing the camera parameters file",
    )
    parser.add_argument(
        "--camera_params_name",
        type=str,
        default="camera_params.json",
        help="Name of the camera parameters file",
    )
    parser.add_argument(
        "--plate_marker_info",
        type=str,
        nargs=5,
        default=("YWMBMMCCCYWBMYWBYWBC", 20, 5, 4, 7.5),
        help="Information about the object marker (seq_string, seq_len, seq_vocabulary_len, min_pattern_len, marker_radius_cm)",
    )
    parser.add_argument(
        "--back_marker_size",
        type=float,
        nargs=2,
        default=(13.0, 23.0),
        help="Size of the back marker (width_cm, height_cm)",
    )

    args = parser.parse_args()
    args.plate_marker_info = tuple(args.plate_marker_info)
    args.plate_marker_info = (
        str(args.plate_marker_info[0]),
        int(args.plate_marker_info[1]),
        int(args.plate_marker_info[2]),
        int(args.plate_marker_info[3]),
        float(args.plate_marker_info[4]),
    )
    args.back_marker_size = tuple(args.back_marker_size)

    assert (
        len(args.plate_marker_info[0]) == args.plate_marker_info[1]
    ), "Invalid sequence length"
    assert (
        len(set(args.plate_marker_info[0])) == args.plate_marker_info[2]
    ), "Invalid sequence vocabulary length"
    assert args.plate_marker_info[3] > 0, "Invalid plate marker radius"
    assert (
        args.back_marker_size[0] > 0 and args.back_marker_size[1] > 0
    ), "Invalid back marker size"

    main(args)
