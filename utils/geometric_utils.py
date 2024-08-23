import math
import cv2
import numpy as np


def fit_line(points: np.ndarray):
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    # Extract line parameters from fitLine output
    vx, vy, x0, y0 = line[0], line[1], line[2], line[3]

    # Convert to the form ax + by + c = 0
    # The direction vector (vx, vy) is perpendicular to the normal vector (a, b)
    a = vy
    b = -vx
    c = -(a * x0 + b * y0)
    return a, b, c


def find_line_equation(x1, y1, x2, y2):
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


def find_plane_equation(point1, point2, point3):
    """
    Find the equation of the point passing through three points
    """
    vector1 = np.array(point2) - np.array(point1)
    vector2 = np.array(point3) - np.array(point1)
    normal_vector = np.cross(vector1, vector2)

    k = -np.sum(point1 * normal_vector)
    return np.array([normal_vector[0], normal_vector[1], normal_vector[2], k])


def find_plane_equation_from_normal(point, normal):
    """
    Find the equation of the plane passing through a point with a given normal
    """
    k = -np.sum(point * normal)
    return np.array([normal[0], normal[1], normal[2], k])


def plane2camera(point, normal, r, t):
    rot_mat = cv2.Rodrigues(r)[0]
    t = t.reshape(1, 3)

    # TODO check if this is correct
    new_point = np.array(point)
    new_point = new_point + t
    new_normal = np.array(normal)
    new_normal = new_normal @ rot_mat
    return new_point, new_normal


def find_line_line_intersection(line1, line2):
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


def find_plane_line_intersection(plane, point1, point2):
    """
    Find intersection between a plane and the line passing through two points
    """
    if len(plane) == 2:
        plane = find_plane_equation_from_normal(point=plane[0], normal=plane[1])

    point1 = np.array(point1)
    point2 = np.array(point2)

    direction = point2 - point1
    plane_norm = np.array([plane[0], plane[1], plane[2]])
    product = plane_norm @ direction
    if abs(product) > 1e-6:
        p_co = plane_norm * (-plane[3] / (plane_norm @ plane_norm))

        w = point1 - p_co
        fac = -(plane_norm @ w) / product
        return point1 + (direction * fac)
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
