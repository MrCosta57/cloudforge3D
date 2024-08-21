import os, sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)
from typing import List, Tuple
import cv2
import numpy as np
from utils.scanner_utils import get_world_points_from_cm


def fit_marker_rectangle(
    contours, min_area: int = 100000, debug=False, palette_frame=None
):
    """
    Fit a rectangle to the given contours.
    :param contours: List of contours to search for the rectangle
    :param min_area: Minimum area for the rectangle
    :return: Inner rectangle found from the marker contours
    """
    rectangles = []
    for contour in contours:
        # Approximate the contour to a polygon
        # arcLength: Calculates a contour perimeter or a curve length
        epsilon = 0.01 * cv2.arcLength(contour, closed=True)
        poly = cv2.approxPolyDP(contour, epsilon, closed=True)
        area = cv2.contourArea(poly)
        # If polygon has 4 vertices and area not too small, it's the rectangle marker
        if len(poly) == 4 and area > min_area:
            rectangles.append(poly)

    if len(rectangles) == 0:
        raise Exception("No rectangles found")

    rectangles.sort(key=cv2.contourArea)
    inner_rectangle = rectangles[0]

    if debug and palette_frame is not None:
        for i in range(4):
            cv2.line(
                palette_frame,
                tuple(inner_rectangle[i][0]),
                tuple(inner_rectangle[(i + 1) % 4][0]),
                (50, 205, 50),
                3,
            )
            cv2.drawMarker(
                palette_frame,
                tuple(inner_rectangle[i][0]),
                (50, 205, 70),
                markerSize=15,
                markerType=cv2.MARKER_TILTED_CROSS,
                thickness=3,
            )

    return inner_rectangle


def compute_back_marker_extrinsic(
    rectangle: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    real_marker_size: Tuple[float, float],
    debug=False,
    palette_frame=None,
):
    a, b, c, d = rectangle
    # (width, height)
    marker_size = (
        get_world_points_from_cm(real_marker_size[0]),
        get_world_points_from_cm(real_marker_size[1]),
    )
    obj_points = np.array(
        [
            [0, 0, 0],
            [marker_size[0], 0, 0],
            [marker_size[0], marker_size[1], 0],
            [0, marker_size[1], 0],
        ],
        dtype=np.float32,
    )
    img_points = np.array(
        [
            a[0],
            b[0],
            c[0],
            d[0],
        ],
        dtype=np.float32,
    )

    _, r, t = cv2.solvePnP(
        obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE
    )

    if debug and palette_frame is not None:
        proj_vertexes = cv2.projectPoints(
            np.concatenate([obj_points, [[0, 0, 100]]], axis=0),
            r,
            t,
            camera_matrix,
            dist_coeffs,
            img_points,
        )[0]
        proj_vertexes = np.round(proj_vertexes).astype(np.int32)
        for i in range(len(proj_vertexes) - 1):
            cv2.drawMarker(
                palette_frame,
                tuple(proj_vertexes[i][0]),
                (255, 255, 255) if i == 0 else (0, 255, 0),
                markerSize=15,
                markerType=cv2.MARKER_STAR if i == 0 else cv2.MARKER_TILTED_CROSS,
                thickness=3,
            )
        # Axis lines
        cv2.arrowedLine(
            palette_frame,
            tuple(proj_vertexes[0][0]),
            tuple(proj_vertexes[1][0]),
            (255, 0, 0),
            3,
            tipLength=0.05,
        )
        cv2.putText(
            palette_frame,
            "X",
            tuple(proj_vertexes[1][0] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        cv2.arrowedLine(
            palette_frame,
            tuple(proj_vertexes[0][0]),
            tuple(proj_vertexes[3][0]),
            (0, 255, 0),
            3,
            tipLength=0.05,
        )
        cv2.putText(
            palette_frame,
            "Y",
            tuple(proj_vertexes[3][0] + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.arrowedLine(
            palette_frame,
            tuple(proj_vertexes[0][0]),
            tuple(proj_vertexes[4][0]),
            (0, 0, 255),
            3,
            tipLength=0.05,
        )
        cv2.putText(
            palette_frame,
            "Z",
            tuple(proj_vertexes[4][0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    return r, t
