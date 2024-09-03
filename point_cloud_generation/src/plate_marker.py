import os, sys
from typing import List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)
import math
import cv2
import random
import numpy as np
from utils.scanner_utils import (
    get_point_color,
    get_marker_seq_start,
    get_world_points_from_cm,
)
from cv2.typing import MatLike
from utils.geometric_utils import convert_to_polar
from termcolor import colored

Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]
ListCenter = List[Tuple[int, int]]


def find_plate_marker_cand_dot_centers(
    contours,
    frame_w: int,
    frame_h: int,
    debug: bool = False,
    palette_frame=None,
) -> ListCenter:
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
        axis_1, axis_2 = round(ellipse[1][0]), round(ellipse[1][1])
        angle = round(ellipse[2])

        # Marker dots are:
        # - In the lower half of the frame
        # - Not too close to the edges
        # - Not too small or too big
        if (
            10 < center_x < frame_w - 10
            and frame_h / 2 < center_y < frame_h
            and 15 < axis_1 < 65
            and 15 < axis_2 < 65
        ):
            # Check if the center is not too close to any other center
            predicate = any([math.dist(c, [center_x, center_y]) < 30 for c in centers])
            if not predicate:
                centers.append((center_x, center_y))

                if debug and palette_frame is not None:
                    cv2.ellipse(
                        palette_frame,
                        ellipse,
                        (34, 139, 34),
                        2,
                    )
                    cv2.drawMarker(
                        palette_frame,
                        (center_x, center_y),
                        (34, 139, 54),
                        markerSize=15,
                        markerType=cv2.MARKER_TILTED_CROSS,
                        thickness=2,
                    )

    return centers


def fit_marker_ellipse(
    points: ListCenter,
    num_round: int = 50,
    dist_thresh: int = 5,
    debug: bool = False,
    palette_frame=None,
) -> Ellipse:
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

    if debug and palette_frame is not None:
        cv2.ellipse(
            palette_frame,
            best_ellipse,
            (50, 205, 50),
            3,
        )
        cv2.drawMarker(
            palette_frame,
            (round(best_ellipse[0][0]), round(best_ellipse[0][1])),
            (50, 205, 70),
            markerSize=15,
            markerType=cv2.MARKER_TILTED_CROSS,
            thickness=3,
        )
    return best_ellipse


def compute_plate_marker_extrinsic(
    ellipse: Ellipse,
    dot_centers: ListCenter,
    camera_matrix: np.ndarray,
    marker_info: Tuple[str, int, float],
    frame: MatLike,
    debug: bool = False,
    palette_frame: MatLike | None = None,
    print_fn=print,
    angle_gap_threshold=35,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the extrinsic parameters of the plate marker using the ellipse and the dot centers
    :param ellipse: Ellipse of the plate marker
    :param dot_centers: List of dot centers
    :param camera_matrix: Camera matrix (intrinsic parameters)
    :param marker_info: Tuple containing the sequence string, the minimum pattern length and the marker radius
    :param frame: Frame to search for the dot colors
    :param debug: Debug flag
    :param palette_frame: Frame to draw the debug information on
    :param print_fn: Print function
    :param angle_gap_threshold: Angle gap threshold for the pattern identification in the marker sequences
    :return: Rotation and translation vectors
    """
    seq_string, min_pattern_len, marker_radius_cm = marker_info
    seq_len = len(seq_string)
    # Duplicate the sequence to handle the case when
    # the pattern is split between the last and the first element
    seq_string = seq_string * 2

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
        if abs(cv2.pointPolygonTest(poly, p, measureDist=True)) < 20
    ]
    if len(dot_centers) < 10:
        raise Exception("Too few dot centers for plate marker detection")

    if debug:
        if len(dot_centers) != seq_len:
            print_fn(
                colored(
                    f"Dot centers for plate extrinsic computation are {len(dot_centers)}",
                    "light_yellow",
                )
            )

    # Create a list containing the point angle in polar coordinates, the point color and the original position
    # If color is not found, value is None
    dot_tuple = [
        (
            convert_to_polar(ellipse_center, p)[1],
            get_point_color(frame, p),
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
        # Build the pattern for identifying the circle id by taking the color of the dot
        # in the next `min_pattern_len` positions
        for j in range(min_pattern_len):
            curr_t = dot_tuple[(i + j) % dot_tuple_len]
            next_t = dot_tuple[(i + j + 1) % dot_tuple_len]
            # Make the pattern invalid when:
            # - The angle difference is more than 30 degrees, probably there's a gap among the dot sequence
            # - The color is not found (None)
            if (curr_t[1] is None) or (
                abs(curr_t[0] - next_t[0] + 360) % 360 > angle_gap_threshold
            ):
                pattern = ""
                break
            pattern += curr_t[1]
        patterns.append(pattern)

    # Find the indexes of the dots, if the pattern is not found, the index is -1
    dot_indexes = [get_marker_seq_start(seq_string, pattern) for pattern in patterns]

    # Fill the missing indexes by inferring their values from the known ones
    # Duplicate the length of the dot list to fill indexes in a circular manner
    for i in range((dot_tuple_len - 1) * 2):
        curr_t = dot_tuple[i % dot_tuple_len]
        next_t = dot_tuple[(i + 1) % dot_tuple_len]
        curr_idx = dot_indexes[i % dot_tuple_len]
        next_idx = dot_indexes[(i + 1) % dot_tuple_len]

        # Infer the index of the next dot if:
        # - The current dot has a known index
        # - The next dot has an unknown index
        # - The angle difference between the current and the next dot is less than a threshold
        if (
            curr_idx != -1
            and next_idx == -1
            and abs(curr_t[0] - next_t[0] + 360) % 360 <= angle_gap_threshold
        ):
            dot_indexes[(i + 1) % dot_tuple_len] = (curr_idx + 1) % seq_len

    marker_angle = 360 // seq_len
    marker_radius_rw = get_world_points_from_cm(marker_radius_cm)

    # Filter out the dots with invalid indexes (solvePnP requires at least 4 points)
    # and build the object and image points
    obj_points = np.array(
        [
            [
                marker_radius_rw * math.cos(math.radians(marker_angle * idx)),
                marker_radius_rw * math.sin(math.radians(marker_angle * idx)),
                0,
            ]
            for idx in dot_indexes
            if idx != -1
        ],
        dtype=np.float32,
    )
    img_points = np.array(
        [t[2] for t, idx in zip(dot_tuple, dot_indexes) if idx != -1], dtype=np.float32
    )
    if len(obj_points) < 4 or len(img_points) < 4:
        raise Exception("Too few points for plate marker pose estimation")

    # Solve the 3D-2D point correspondences
    _, r, t = cv2.solvePnP(
        obj_points, img_points, camera_matrix, np.empty(0), flags=cv2.SOLVEPNP_IPPE
    )

    if debug and palette_frame is not None:
        proj_dot_centers = cv2.projectPoints(
            obj_points, r, t, camera_matrix, np.empty(0), img_points
        )[0]
        proj_dot_centers = np.round(proj_dot_centers).astype(np.int32)
        for p in proj_dot_centers:
            cv2.drawMarker(
                palette_frame,
                (p[0][0], p[0][1]),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=15,
                thickness=2,
            )
        debug_color_dict = {
            "B": (0, 0, 0),
            "W": (255, 255, 255),
            "Y": (0, 255, 255),
            "M": (255, 0, 255),
            "C": (255, 255, 0),
        }
        for i in range(dot_tuple_len):
            # Red color for the dots with unknown color
            tmp_color = (
                (0, 0, 255)
                if dot_tuple[i][1] is None
                else debug_color_dict[dot_tuple[i][1]]  # type: ignore
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
        proj_axis_pts = cv2.projectPoints(
            np.array(
                [
                    [0, 0, 0],
                    [marker_radius_rw, 0, 0],
                    [0, marker_radius_rw, 0],
                    [0, 0, marker_radius_rw],
                ],
                dtype=np.float32,
            ),
            r,
            t,
            camera_matrix,
            np.empty(0),
            img_points,
        )[0]
        proj_axis_pts = np.round(proj_axis_pts).astype(np.int32)
        cv2.arrowedLine(
            palette_frame,
            tuple(proj_axis_pts[0][0]),
            tuple(proj_axis_pts[1][0]),
            (255, 0, 0),
            3,
            tipLength=0.05,
        )
        cv2.putText(
            palette_frame,
            "X",
            tuple(proj_axis_pts[1][0] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        cv2.arrowedLine(
            palette_frame,
            tuple(proj_axis_pts[0][0]),
            tuple(proj_axis_pts[2][0]),
            (0, 255, 0),
            3,
            tipLength=0.05,
        )
        cv2.putText(
            palette_frame,
            "Y",
            tuple(proj_axis_pts[2][0] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.arrowedLine(
            palette_frame,
            tuple(proj_axis_pts[0][0]),
            tuple(proj_axis_pts[3][0]),
            (0, 0, 255),
            3,
            tipLength=0.05,
        )
        cv2.putText(
            palette_frame,
            "Z",
            tuple(proj_axis_pts[3][0] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.drawMarker(
            palette_frame,
            tuple(proj_axis_pts[0][0]),
            (255, 255, 255),
            markerSize=15,
            markerType=cv2.MARKER_STAR,
            thickness=2,
        )

    return r, t
