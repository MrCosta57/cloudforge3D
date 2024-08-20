import math
import cv2
import numpy as np


def get_marker_seq_start(seq_string: str, pattern: str) -> int:
    return seq_string.find(pattern)


def get_world_points_from_cm(size: float):
    return int(size * 10)


def find_black_objects(frame: np.ndarray, threshold: int = 110) -> np.ndarray:
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # image to gray scale

    # Find "black" objects/contours
    _, frame_gray = cv2.threshold(frame_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    # Smooth the image
    frame_gray = cv2.GaussianBlur(frame_gray, (3, 3), 0)
    # Dilate to ensure lines without gaps
    frame_gray = cv2.morphologyEx(
        frame_gray, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8)
    )
    return frame_gray


def convert_to_polar(ellipse_center, point):
    vector = np.array(
        [
            point[0] - ellipse_center[0],
            point[1] - ellipse_center[1],
        ]
    )
    radius = np.linalg.norm(vector)
    # Return signed angle in radians
    angle = math.atan2(vector[1], vector[0])

    # Convert angle to degrees
    angle_degrees = math.degrees(angle)
    # Ensure angle is in the range [0, 360)
    if angle_degrees < 0:
        angle_degrees += 360

    return radius, angle_degrees


def get_point_color(frame, frame_point, marker_seq):
    assert (
        len(set(marker_seq).difference(set(["B", "W", "Y", "M", "C"]))) == 0
    ), "Marker sequence is not supported"

    # HSV color space is more suitable for object detection
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # frame_point are encoded as (x, y) but frame uses (y, x)
    frame_value = np.array(frame_hsv[round(frame_point[1]), round(frame_point[0])])

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
