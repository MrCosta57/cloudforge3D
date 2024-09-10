import math
import cv2
import numpy as np
from typing import Tuple


def fit_line(points: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit a line to a set of points using the least squares method

    :param points: Points to fit the line to
    :return: Line parameters (a, b, c)
    """
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

    :param x1: x-coordinate of the first point
    :param y1: y-coordinate of the first point
    :param x2: x-coordinate of the second point
    :param y2: y-coordinate of the second point
    :return: Line parameters (a, b, c)
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
    """
    Generate random points on a line segment

    :param start: Start point of the line segment
    :param end: End point of the line segment
    :param n: Number of points to generate
    :return: Random points on the line segment
    """
    start_vec = np.array(start).squeeze()
    end_vec = np.array(end).squeeze()
    t = np.random.random(n).reshape(-1, 1)
    points = start_vec + t * (end_vec - start_vec)
    return points


def find_plane_equation(point1, point2, point3):
    """
    Find the equation of the point passing through three points

    :param point1: First point
    :param point2: Second point
    :param point3: Third point
    :return: Plane parameters (a, b, c, d)
    """
    vector1 = np.array(point2) - np.array(point1)
    vector2 = np.array(point3) - np.array(point1)
    normal_vector = np.cross(vector1, vector2)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    k = -np.sum(point1 * normal_vector)
    return np.array([normal_vector[0], normal_vector[1], normal_vector[2], k])


def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a plane to a set of points using the SVD method

    :param points: Points to fit the plane to
    :return: Centroid and normal of the fitted plane
    """

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

    :param point: Point on the plane
    :param normal: Normal vector of the plane
    :return: Plane parameters (a, b, c, d)
    """
    k = -np.sum(point * normal)
    return np.array([normal[0], normal[1], normal[2], k])


def find_line_line_intersection(
    line1: Tuple[float, float, float], line2: Tuple[float, float, float]
) -> Tuple[float, float]:
    """
    Find intersection between two lines

    :param line1: Line parameters (a1, b1, c1)
    :param line2: Line parameters (a2, b2, c2)
    :return: Intersection point (x, y)
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
    """
    Find intersection between a plane and the line passing through two points

    :param plane: Plane parameters (normal, point)
    :param point1: First point on the line
    :param point2: Second point on the line
    :return: Intersection point (x, y, z)
    """
    assert len(plane) == 2
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


def convert_to_polar(ellipse_center: Tuple[int, int], point: Tuple[float, float]):
    """
    Convert a point from Cartesian to polar coordinates

    :param ellipse_center: Center of the ellipse
    :param point: Point in Cartesian coordinates
    :return: Polar coordinates (radius, angle)
    """
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
