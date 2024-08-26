import os, sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)
from utils.geometric_utils import (
    fit_line,
    find_line_equation,
    find_line_line_intersection,
    random_points_on_line_segment,
)
from utils.general_utils import get_resized_frame
from utils.scanner_utils import get_world_points_from_cm
from plate_marker import Ellipse
import cv2
import numpy as np


def find_laser_line_backmarker(rectangle: np.ndarray, frame: np.ndarray):

    height, width, _ = frame.shape
    rectangle_mask = np.zeros((height, width), dtype=np.uint8)

    # Smooth frame with filter robust to outliers, making it useful for removing impulse noise or salt-and-pepper noise/reflections
    frame = cv2.medianBlur(frame.copy(), 3)
    # Extract red channel from the frame, not interested in the other colors
    _, laser_mask = cv2.threshold(frame[:, :, 2], 200, 255, cv2.THRESH_BINARY)

    # Fill the polygon (rectangle) on the mask
    cv2.fillPoly(rectangle_mask, [rectangle], (255, 255, 255))
    # Perform bitwise AND between the threshold mask and the rectangle mask
    points_inside_mask = cv2.bitwise_and(laser_mask, rectangle_mask)
    # Extract points (non-zero coordinates) from the resulting mask

    filtered_idx = np.where(points_inside_mask > 0)[::-1]
    points_inside_rectangle = np.column_stack(filtered_idx)
    line_a, line_b, line_c = fit_line(points_inside_rectangle)

    return line_a, line_b, line_c


def find_n_laser_point_backmarker(
    rectangle: np.ndarray, frame: np.ndarray, n_points: int = 10
) -> np.ndarray:
    v1, v2, v3, v4 = rectangle[:, 0, :]
    line_a, line_b, line_c = find_laser_line_backmarker(rectangle, frame)
    upper_line = find_line_equation(v1[0], v1[1], v2[0], v2[1])
    lower_line = find_line_equation(v3[0], v3[1], v4[0], v4[1])
    first_point = find_line_line_intersection((line_a, line_b, line_c), upper_line)
    second_point = find_line_line_intersection((line_a, line_b, line_c), lower_line)

    points = random_points_on_line_segment(first_point, second_point, n_points)
    points = np.round(points).astype(np.int32)
    return points


def find_n_laser_point_platemarker(
    ellipse: Ellipse, frame: np.ndarray, n_points: int = 10
):
    center_x, center_y = round(ellipse[0][0]), round(ellipse[0][1])
    # Crop the frame around the ellipse center to reduce the search area
    # Finding point don't require high precision
    cropped_frame = frame[
        center_y + 100 : center_y + 200, center_x - 150 : center_x + 150, :
    ]

    # HSV color space is more suitable for object detection
    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
    # Smooth frame with filter robust to outliers, making it useful for removing impulse noise or salt-and-pepper noise/reflections
    cropped_frame = cv2.medianBlur(cropped_frame, 3)

    # Red can have two distinct hue ranges in the HSV space (0-10 and 160-180)
    # Thresholds here are fixed to take into account some color variations
    mask1 = cv2.inRange(cropped_frame, np.array((0, 55, 230)), np.array((20, 255, 255)))
    mask2 = cv2.inRange(
        cropped_frame, np.array((150, 55, 230)), np.array((180, 255, 255))
    )
    laser_mask = cv2.bitwise_or(mask1, mask2)

    # Pick the non-zero values in the mask
    points = cv2.findNonZero(laser_mask)
    if points is None:
        raise Exception("Point not detected")

    sampled_idx = np.random.choice(points.shape[0], n_points, replace=False)
    laser_points = points[sampled_idx, 0, :]
    # Convert to non-cropped coordinates
    laser_points[:, 0] = laser_points[:, 0] + center_x - 150
    laser_points[:, 1] = laser_points[:, 1] + center_y + 100
    laser_points = np.round(laser_points).astype(np.int32)
    return laser_points


def find_all_laser_points_obj(ellipse: Ellipse, frame: np.ndarray) -> np.ndarray:
    height, width, _ = frame.shape
    ellipse_mask = np.zeros((height, width), dtype=np.uint8)
    # HSV color space is more suitable for object detection
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Smooth frame with filter robust to outliers, making it useful for removing impulse noise or salt-and-pepper noise/reflections
    frame_hsv = cv2.medianBlur(frame_hsv, 3)

    # Red can have two distinct hue ranges in the HSV space (0-10 and 160-180)
    # Thresholds here are fixed to take into account some color variations
    mask1 = cv2.inRange(frame_hsv, np.array((0, 70, 225)), np.array((20, 255, 255)))
    mask2 = cv2.inRange(frame_hsv, np.array((155, 70, 230)), np.array((180, 255, 255)))
    laser_mask = cv2.bitwise_or(mask1, mask2)

    center_x, center_y = round(ellipse[0][0]), round(ellipse[0][1])
    half_axis_1, half_axis_2 = round(ellipse[1][0] / 2), round(ellipse[1][1] / 2)
    angle = round(ellipse[2])
    poly = cv2.ellipse2Poly(
        (center_x, center_y), (half_axis_1, half_axis_2), angle, 0, 360, delta=5
    )
    poly = np.asarray(poly)
    cv2.fillPoly(ellipse_mask, [poly], (255, 255, 255))
    points_inside_mask = cv2.bitwise_and(laser_mask, ellipse_mask)
    points_inside_mask = cv2.morphologyEx(
        points_inside_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    filtered_idx = np.where(points_inside_mask > 0)[::-1]
    points_inside_ellipse = np.column_stack(filtered_idx)
    points_inside_ellipse = np.round(points_inside_ellipse).astype(np.int32)
    return points_inside_ellipse
