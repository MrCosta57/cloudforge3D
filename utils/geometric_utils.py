import math
import cv2
import numpy as np
from typing import Tuple


def fit_line(points: np.ndarray) -> Tuple[float, float, float]:
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    # Extract line parameters from fitLine output
    vx, vy, x0, y0 = line[0], line[1], line[2], line[3]

    # Convert to the form ax + by + c = 0
    # The direction vector (vx, vy) is perpendicular to the normal vector (a, b)
    a = vy
    b = -vx
    c = -(a * x0 + b * y0)
    return a, b, c


def find_line_equation(
    x1: float, y1: float, x2: float, y2: float
) -> Tuple[float, float, float]:
    """
    Find the equation of the line passing through two points
    """
    if x2 - x1 == 0:
        a = 1
        b = 0
        c = -x1
    else:
        m = (y2 - y1) / (x2 - x1)
        a = -m
        b = 1
        c = m * x1 - y1
    return a, b, c


def random_points_on_line_segment(
    start: Tuple[float, float], end: Tuple[float, float], n: int
) -> np.ndarray:
    start_vec = np.array(start).squeeze()
    end_vec = np.array(end).squeeze()
    t = np.random.random(n).reshape(-1, 1)
    points = start_vec + t * (end_vec - start_vec)
    return points


def find_plane_equation(point1, point2, point3):
    """
    Find the equation of the point passing through three points
    """
    vector1 = np.array(point2) - np.array(point1)
    vector2 = np.array(point3) - np.array(point1)
    normal_vector = np.cross(vector1, vector2)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    k = -np.sum(point1 * normal_vector)
    return np.array([normal_vector[0], normal_vector[1], normal_vector[2], k])


def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Calculate the mean of the points, i.e., the centroid
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    # Compute the SVD
    _, _, vh = np.linalg.svd(centered_points)
    # The normal of the plane is the last row of vh
    normal = vh[-1]
    return centroid, normal


def find_plane_equation_from_normal(point: np.ndarray, normal: np.ndarray):
    """
    Find the equation of the plane passing through a point with a given normal
    """
    k = -np.sum(point * normal)
    return np.array([normal[0], normal[1], normal[2], k])


def plane2camera(
    point: np.ndarray, normal: np.ndarray, r: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    rot_mat = cv2.Rodrigues(r)[0]
    t = t.reshape(3, 1)

    new_point = point.reshape(3, 1)
    new_point = rot_mat @ new_point + t
    new_normal = normal.reshape(3, 1)
    new_normal = rot_mat @ new_normal
    new_normal = new_normal.squeeze()
    new_point = new_point.squeeze()
    return new_point, new_normal


def find_line_line_intersection(
    line1: Tuple[float, float, float], line2: Tuple[float, float, float]
) -> Tuple[float, float]:
    """
    Find intersection between two lines
    """
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(det) > 1e-6:
        x = (b1 * c2 - b2 * c1) / det
        y = (c1 * a2 - c2 * a1) / det
        return (x, y)
    else:
        raise Exception("No intersection found")


def find_plane_line_intersection(
    plane: Tuple[np.ndarray, np.ndarray], point1: np.ndarray, point2: np.ndarray
) -> np.ndarray:
    assert len(plane) == 2
    """
    Find intersection between a plane and the line passing through two points
    """
    plane_eq = find_plane_equation_from_normal(plane[0], plane[1])
    # [n, 3, 1]
    line_direction = point2 - point1
    # [1, 3, 1]
    plane_normal = np.array(plane_eq[:3]).reshape(1, 3, 1)
    k = plane_eq[3]
    # [n, 1, 1]
    denominator = np.sum(plane_normal * line_direction, axis=1, keepdims=True)
    if (np.abs(denominator) > 1e-6).all():
        # [1, 3, 1]
        p_co = plane_normal * (-k / np.sum(plane_normal**2))
        # [n, 3, 1]
        w = point1 - p_co
        # [1, 3, 1] * [n, 3, 1] -> [n, 3, 1] -> [n, 1, 1]
        fac = -np.sum(plane_normal * w, axis=1, keepdims=True) / denominator
        # [n, 3, 1] + [n, 1, 1] * [n, 3, 1] -> [n, 3, 1]
        intersections = point1 + fac * line_direction
        return intersections
    else:
        raise Exception("No intersection found")


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
