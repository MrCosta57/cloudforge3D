from typing import Tuple
import numpy as np
import random
import cv2


def seed_everything(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)


def get_resized_frame(
    frame: np.ndarray, width: int, height: int, scaling_factor: float
):
    # Resize the frame for display it
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame


def get_undistorted_frame(
    frame: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
):
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    # Undistort
    dst = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)
    # Crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    return dst


def marker2camera(r, t):
    rot_matr = cv2.Rodrigues(r)[0]
    mtx = np.concatenate(
        [
            np.concatenate([rot_matr, t], axis=1),
            np.array([[0, 0, 0, 1]]),
        ],
        axis=0,
    )
    return mtx


def camera2marker(r, t):
    rot_matr = cv2.Rodrigues(r)[0]
    mtx = np.concatenate(
        [
            np.concatenate(
                [
                    cv2.transpose(rot_matr),
                    -cv2.transpose(rot_matr) @ t,
                ],
                axis=1,
            ),
            np.array([[0, 0, 0, 1]]),
        ],
        axis=0,
    )
    return mtx


def skip_to_time(cap: cv2.VideoCapture, target_minute=0, target_second=0):
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Calculate the target frame number
    target_time_in_seconds = target_minute * 60 + target_second
    target_frame = int(target_time_in_seconds * fps)
    # Set the video capture to the target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    return cap
