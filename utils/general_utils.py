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
                    np.array(cv2.transpose(rot_matr)),
                    np.array(-cv2.transpose(rot_matr) @ t),
                ],
                axis=1,
            ),
            np.array([[0, 0, 0, 1]]),
        ],
        axis=0,
    )
    return mtx
