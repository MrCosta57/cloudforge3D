import os, sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)
import argparse
import numpy as np
import cv2
import json
from utils.general_utils import get_resized_frame


def main(args: argparse.Namespace):
    print("***Camera calibration script***")
    chessboard_size = args.chessboard_size
    window_size = args.window_size
    time_skips = args.time_skip
    video_path = os.path.join(args.video_dir, args.video_name)
    output_path = os.path.join(args.output_dir, args.output_name)

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(
        -1, 2
    )
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    cap = cv2.VideoCapture(video_path)
    # Sanity checks
    if not cap.isOpened():
        print("Error opening video file.")
        return
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    assert height > width, "Video frame is not in portrait mode"

    print("Start analyzing video file...")
    skip_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners for debugging purposes
            cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)

        resized_frame = get_resized_frame(frame, window_size, width, height)
        cv2.imshow("Frame", resized_frame)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            print("User interrupted the process")
            cap.release()
            cv2.destroyAllWindows()
            return

        # Skip to the next frame based on the time interval
        cap.set(cv2.CAP_PROP_POS_MSEC, (skip_count * time_skips))
        skip_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Finish analyzing video file")

    print("Start calibrating camera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("Camera matrix:")
    print(mtx)
    print("Distortion coefficients:")
    print(dist)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    total_error = mean_error / len(objpoints)
    print("Total calibration error: {}".format(total_error))

    # save the camera parameters to json file
    camera_params = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "total_error": total_error,
    }
    with open(output_path, "w") as f:
        json.dump(camera_params, f)
    print(f"Camera parameters saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that perform camera calibration from a provided video"
    )
    parser.add_argument(
        "--chessboard_size",
        type=int,
        nargs=2,
        default=(9, 6),
        help="Size of the chessboard",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        nargs=2,
        default=(500, 700),
        help="Size of the window to display the video (width, height)",
    )
    parser.add_argument(
        "--time_skip",
        type=int,
        default=500,
        help="Time interval to skip in milliseconds",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="camera_calibration/data",
        help="Directory containing the video file",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="calibration.mov",
        help="Name of the video file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="camera_calibration/output",
        help="Directory to save the output file",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="camera_params.json",
        help="Name of the output file",
    )
    args = parser.parse_args()
    args.chessboard_size = tuple(args.chessboard_size)
    args.window_size = tuple(args.window_size)
    args.time_skip = float(args.time_skip)
    main(args)
