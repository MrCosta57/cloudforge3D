import os, sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)
import math
import cv2
import random
import numpy as np
from utils.scanner_utils import (
    convert_to_polar,
    get_point_color,
    get_marker_seq_start,
    get_world_points_from_cm,
)


def find_plate_marker_cand_dot_centers(contours, frame_w: int, frame_h: int):
    """
    Find the candidate centers of the plate marker dots. There might be some noise points
    :param contours: List of contours to search for the dots
    :param frame_w: Frame width
    :param frame_h: Frame height
    :return: List of candidate dot centers
    """
    centers = []
    for contour in contours:
        if len(contour) < 5:
            # Ellipse fitting requires at least 5 points
            continue

        ellipse = cv2.fitEllipse(contour)
        center_x, center_y = round(ellipse[0][0]), round(ellipse[0][1])
        axis_1, axis_2 = ellipse[1]
        angle = ellipse[2]

        # Marker dots are:
        # - In the lower half of the frame
        # - Not too close to the edges
        # - Not too small or too big
        if (
            10 < center_x < frame_w - 10
            and frame_h / 2 < center_y < frame_h
            and 20 < axis_1 < 60
            and 20 < axis_2 < 60
        ):
            # Check if the center is not too close to any other center
            predicate = any([math.dist(c, [center_x, center_y]) < 30 for c in centers])
            if not predicate:
                centers.append((center_x, center_y))

    return centers


def fit_marker_ellipse(points, num_round: int = 50, dist_thresh: int = 5):
    """
    Fit an ellipse to the given points using RANSAC.
    :param points: List of points to fit the ellipse
    :param num_round: Number of RANSAC rounds
    :param dist_thresh: Distance threshold for inliers
    :return: Best ellipse found
    """
    if len(points) < 10:
        raise Exception("Too few points for ellipse fitting")

    candidates = []
    for _ in range(num_round):
        # Randomly sample 5 points (minimum required for ellipse fitting)
        sampled = np.array(random.sample(points, 5))
        candidate = cv2.fitEllipse(sampled)

        center_x, center_y = round(candidate[0][0]), round(candidate[0][1])
        half_axis_1 = round(candidate[1][0] / 2)
        half_axis_2 = round(candidate[1][1] / 2)
        angle = round(candidate[2])

        # Sanity check
        if math.isnan(half_axis_1) or math.isnan(half_axis_2):
            continue

        # Approximate ellipse to polygon for having the contour
        # Delta: angle between the subsequent polyline vertices. It defines the approximation accuracy.
        poly = cv2.ellipse2Poly(
            (center_x, center_y),
            (half_axis_1, half_axis_2),
            angle,
            0,
            360,
            delta=5,
        )
        poly = np.asarray(poly)

        # The point is an inlier if its distance from the polygon is not too large
        inliers = [
            p
            for p in points
            if abs(cv2.pointPolygonTest(poly, p, measureDist=True)) < dist_thresh
        ]
        candidates.append([candidate, len(inliers)])

    # Extract the candidate with max votes
    best_ellipse = max(candidates, key=lambda item: item[1])[0]
    return best_ellipse


def compute_plate_marker_extrinsic(
    ellipse, dot_centers, camera_matrix, dist_coeffs, marker_info, frame, palette_frame
):
    seq_string, seq_len, seq_vocabulary_len, min_pattern_len, marker_radius_cm = (
        marker_info
    )

    ellipse_center = (round(ellipse[0][0]), round(ellipse[0][1]))
    ellipse_half_axes = (round(ellipse[1][0] / 2), round(ellipse[1][1] / 2))
    ellipse_angle = round(ellipse[2])

    # Approximate ellipse to polygon for having the contour
    poly = cv2.ellipse2Poly(
        ellipse_center,
        ellipse_half_axes,
        ellipse_angle,
        0,
        360,
        delta=10,
    )
    poly = np.asarray(poly)

    # Filter out the dot centers that are not close to the ellipse contour
    dot_centers = [
        p
        for p in dot_centers
        if abs(cv2.pointPolygonTest(poly, p, measureDist=True)) < 5
    ]
    if len(dot_centers) < 10:
        raise Exception("Too few dot centers for plate marker detection")

    # Create a list containing the point angle in polar coordinates, the point color and the original position
    # If color is not found, value is None
    dot_tuple = [
        (
            convert_to_polar(ellipse_center, p)[1],
            get_point_color(frame, p, seq_string),
            p,
        )
        for p in dot_centers
    ]
    # Sort the tuples by angle in descending order so the first element is with 0 degrees
    dot_tuple.sort(key=lambda x: x[0], reverse=True)
    dot_tuple_len = len(dot_tuple)

    patterns = []
    for i in range(dot_tuple_len):
        pattern = ""
        # Build the pattern for identifing the circle id by taking the color of the dot in the next min_pattern_len positions
        for j in range(min_pattern_len):
            curr_t = dot_tuple[(i + j) % dot_tuple_len]
            next_t = dot_tuple[(i + j + 1) % dot_tuple_len]
            # Make the pattern invalid when:
            # - The angle difference is more than 30 degrees, probably there's a gap among the dot sequence
            # - The color is None
            if (curr_t[1] is None) or (abs(curr_t[0] - next_t[0] + 360) % 360 > 30):
                pattern = ""
                break
            pattern += curr_t[1]
        patterns.append(pattern)

    # Find the indexes of the dots
    dot_indexes = [get_marker_seq_start(seq_string, pattern) for pattern in patterns]
    # Some dot ids might be missing (value -1), find their indexes using the other dots
    for i in range(dot_tuple_len):
        if dot_indexes[i] >= 0:
            # If the dot id is valid, update the indexes next n dots by overwriting the value
            # This should fill the gaps in the dot indexes
            for j in range(1, min_pattern_len + 1):
                dot_indexes[(i + j) % dot_tuple_len] = (dot_indexes[i] + j) % seq_len

    marker_angle = 360 // seq_len
    marker_radius_rw = get_world_points_from_cm(marker_radius_cm)
    obj_points = np.array(
        [
            [
                marker_radius_rw * math.cos(math.radians(marker_angle * idx)),
                marker_radius_rw * math.sin(math.radians(marker_angle * idx)),
                0,
            ]
            for idx in dot_indexes
        ],
        dtype=np.float32,
    )
    img_points = np.array(
        [dot_tuple[i][2] for i in range(dot_tuple_len)], dtype=np.float32
    )

    _, r, t = cv2.solvePnP(
        obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE
    )

    proj_dot_centers = cv2.projectPoints(
        obj_points, r, t, camera_matrix, dist_coeffs, img_points
    )[0]
    proj_dot_centers = np.round(proj_dot_centers).astype(np.int32)

    cv2.drawMarker(
        palette_frame,
        ellipse_center,
        (0, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=15,
        thickness=2,
    )

    for i in range(dot_tuple_len):
        cv2.drawMarker(
            palette_frame,
            (proj_dot_centers[i][0][0], proj_dot_centers[i][0][1]),
            (0, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=15,
            thickness=2,
        )
        color_dict = {
            "B": (0, 0, 0),
            "W": (255, 255, 255),
            "Y": (0, 255, 255),
            "M": (255, 0, 255),
            "C": (255, 255, 0),
        }
        # Red color for the dots with unknown color
        tmp_color = (
            (0, 0, 255) if dot_tuple[i][1] is None else color_dict[dot_tuple[i][1]]  # type: ignore
        )
        cv2.putText(
            palette_frame,
            str(dot_indexes[i]),
            (dot_tuple[i][2][0] - 10, dot_tuple[i][2][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            tmp_color,
            2,
        )
    cv2.ellipse(
        palette_frame,
        ellipse,
        (255, 0, 0),
        2,
    )

    return r, t
