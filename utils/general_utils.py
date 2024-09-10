from typing import Tuple
import numpy as np
import random
import cv2
from cv2.typing import MatLike


def seed_everything(seed: int = 123):
    """
    Seed everything for reproducibility

    :param seed: Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)


def get_resized_frame(frame: MatLike, width: int, height: int, scaling_factor: float):
    # Resize the frame for display it
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame


def get_undistorted_frame(
    frame: MatLike, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
):
    """
    Undistort the frame using the camera matrix and distortion coefficients

    :param frame: Frame to undistort
    :param camera_matrix: Camera matrix
    :param dist_coeffs: Distortion coefficients
    :return: Undistorted frame
    """
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    # Undistort the frame
    dst = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)
    # Crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    return dst


def marker2camera(r, t):
    """
    Convert marker pose to camera pose

    :param r: Rotation vector
    :param t: Translation vector
    :return: Camera pose matrix
    """
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
    """
    Convert camera pose to marker pose

    :param r: Rotation vector
    :param t: Translation vector
    :return: Marker pose matrix
    """
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


def plane_marker2plane_camera(
    point: np.ndarray, normal: np.ndarray, r: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a point and normal vector from the plane coordinate system to the camera coordinate system

    :param point: Point in the plane coordinate system
    :param normal: Normal vector in the plane coordinate system
    :param r: Rotation vector of the marker
    :param t: Translation vector of the marker
    :return: Point and normal vector in the camera coordinate system
    """
    rot_mat = cv2.Rodrigues(r)[0]
    t = t.reshape(3, 1)

    new_point = point.reshape(3, 1)
    new_point = rot_mat @ new_point + t
    new_normal = normal.reshape(3, 1)
    new_normal = rot_mat @ new_normal
    new_normal = new_normal.squeeze()
    new_point = new_point.squeeze()
    return new_point, new_normal


def skip_to_time(cap: cv2.VideoCapture, target_minute=0, target_second=0):
    """
    Skip to the target time in the video capture

    :param cap: Video capture
    :param target_minute: Target minute
    :param target_second: Target second
    :return: Video capture at the target time
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Calculate the target frame number
    target_time_in_seconds = target_minute * 60 + target_second
    target_frame = int(target_time_in_seconds * fps)
    # Set the video capture to the target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    return cap
