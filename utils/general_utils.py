from typing import Tuple
import numpy as np
import cv2


def get_marker_seq_start(seq_string: str, pattern: str, min_pattern_len: int) -> int:
    assert len(pattern) >= min_pattern_len, "Invalid pattern length"
    return seq_string.index(pattern)


def get_resized_frame(
    frame: np.ndarray, window_size: Tuple[int, int], width: float, height: float
):
    # Resize the frame for display it
    scaling_factor = min(window_size[0] / width, window_size[1] / height)
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
    # undistort
    dst = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    return dst
