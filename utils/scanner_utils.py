import cv2
import numpy as np
from cv2.typing import MatLike
from typing import Tuple


def get_marker_seq_start(seq_string: str, pattern: str) -> int:
    """
    Find the start index of the pattern in the sequence string.
    If the pattern is empty or not found, return -1.
    :param seq_string: Sequence string to search for the pattern
    :param pattern: Pattern to search for in the sequence string
    :return: Start index of the pattern in the sequence string
    """
    return seq_string.find(pattern) if pattern != "" else -1


def get_world_points_from_cm(size: float) -> int:
    """
    Convert centimeters to world points.
    :param size: Size in centimeters
    :return: Size in world points
    """
    return int(size * 10)


def find_black_objects(frame: MatLike, threshold: int = 120) -> np.ndarray:
    """
    Find black objects in the frame by thresholding.
    :param frame: Frame to find black objects in
    :param threshold: Threshold value for the frame
    :return: Frame with black objects (white) and the rest (black)
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # image to gray scale
    # Smooth the image
    frame_gray = cv2.GaussianBlur(frame_gray, (3, 3), 0)
    # Find "black" objects/contours
    _, frame_gray = cv2.threshold(frame_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    # Dilate to ensure lines without gaps
    frame_gray = cv2.morphologyEx(
        frame_gray, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8)
    )
    return frame_gray


def get_point_color(frame: MatLike, frame_point: Tuple[float, float]) -> str | None:
    """
    Get the color of the point in the frame.
    :param frame: Frame to get the color from
    :param frame_point: Point in the frame to get the color from
    :return: Color of the point in the frame
    """
    # HSV color space is more suitable for object detection
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Smooth the frame
    frame_hsv = cv2.GaussianBlur(frame_hsv, (3, 3), 0)

    # frame_point are encoded as (x, y) but frame uses (y, x)
    frame_value = np.array(frame_hsv[round(frame_point[1]), round(frame_point[0])])

    # Check the color of the point
    if (np.array([140, 100, 100]) <= frame_value).all() and (
        frame_value <= np.array([170, 255, 255])
    ).all():
        return "M"
    elif (np.array([80, 100, 100]) <= frame_value).all() and (
        frame_value <= np.array([180, 255, 200])
    ).all():
        return "C"
    elif (np.array([10, 100, 100]) <= frame_value).all() and (
        frame_value <= np.array([40, 255, 255])
    ).all():
        return "Y"
    elif (np.array([0, 0, 0]) <= frame_value).all() and (
        frame_value <= np.array([180, 255, 90])
    ).all():
        return "B"
    elif (np.array([0, 0, 180]) <= frame_value).all() and (
        frame_value <= np.array([180, 99, 255])
    ).all():
        return "W"
    else:
        return None
