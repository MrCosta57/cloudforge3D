import os, sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)
import argparse, json
import numpy as np
import cv2
from utils.general_utils import *


def main(args: argparse.Namespace):
    print("***Point cloud generation script***")
    window_size = args.window_size
    camera_params_path = os.path.join(args.camera_params_dir, args.camera_params_name)
    video_path = os.path.join(args.video_dir, args.video_name)
    output_path = os.path.join(args.output_dir, args.output_name)

    cap = cv2.VideoCapture(video_path)
    # Sanity checks
    if not cap.isOpened():
        print("Error opening video file.")
        return
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    assert height > width, "Video frame is not in portrait mode"
    camera_matrix = None
    dist_coeffs = None
    with open(camera_params_path, "r") as f:
        camera_params = json.load(f)
        camera_matrix = np.array(camera_params["camera_matrix"])
        dist_coeffs = np.array(camera_params["distortion_coefficients"])
        error = camera_params["total_error"]
        print(f"Camera parameters loaded successfully. Calibration error: {error:.4f}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = get_undistorted_frame(
            frame=frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs
        )

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Red can have two distinct hue ranges in the HSV space (0-10 and 160-180)
        # Thresholds here are fixed to take into account some color variations
        mask1 = cv2.inRange(frame_hsv, np.array((0, 70, 230)), np.array((15, 255, 255)))
        mask2 = cv2.inRange(
            frame_hsv, np.array((155, 70, 230)), np.array((180, 255, 255))
        )
        red_mask = cv2.bitwise_or(mask1, mask2)
        # Find contours in the mask
        contours, _ = cv2.findContours(
            red_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

        resized_frame = get_resized_frame(
            frame=frame, window_size=window_size, width=width, height=height
        )
        cv2.imshow("Frame", resized_frame)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            print("User interrupted the process")
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
        "--window_size",
        type=int,
        nargs=2,
        default=(500, 700),
        help="Size of the window to display the video (width, height)",
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
        default="ball.mov",
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
        "--obj_marker_info",
        type=str,
        nargs=3,
        default=("YWMBMMCCCYWBMYWBYWBC", 20, 5),
        help="Information about the object marker (seq_string, seq_len, seq_vocabulary_len, min_pattern_len)",
    )

    args = parser.parse_args()
    args.window_size = tuple(args.window_size)
    args.obj_marker_info = tuple(args.obj_marker_info)

    assert (
        len(args.obj_marker_info[0]) == args.obj_marker_info[1]
    ), "Invalid sequence length"
    assert (
        len(set(args.obj_marker_info[0])) == args.obj_marker_info[2]
    ), "Invalid sequence vocabulary length"

    main(args)
