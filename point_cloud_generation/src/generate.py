import os, sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)
import argparse, json
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime
from termcolor import colored
from utils.general_utils import *
from laser import *
from utils.scanner_utils import *
from utils.geometric_utils import *
from back_marker import *
from plate_marker import *


def main(args: argparse.Namespace):
    print("***Point cloud generation script***")
    debug = args.debug
    print(colored("Debug mode is on", "yellow") if debug else "")
    plate_marker_info = args.plate_marker_info
    back_marker_size = args.back_marker_size
    window_scaling_factor = args.window_scaling_factor
    camera_params_path = os.path.join(args.camera_params_dir, args.camera_params_name)
    video_path = os.path.join(args.video_dir, args.video_name)
    # Date string for the output file in compact format
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir, args.video_name.split(".")[0] + "_" + date_str + ".txt"
    )

    cap = cv2.VideoCapture(video_path)
    # Sanity checks
    if not cap.isOpened():
        print(colored("Error opening video file.", "red"))
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert frame_height > frame_width, "Video frame is not in portrait mode"
    camera_matrix = None
    dist_coeffs = None
    with open(camera_params_path, "r") as file:
        camera_params = json.load(file)
        camera_matrix = np.array(camera_params["camera_matrix"])
        dist_coeffs = np.array(camera_params["distortion_coefficients"])
        error = camera_params["total_error"]
        print(
            colored(
                f"Camera parameters loaded successfully",
                "green",
            )
        )
        print(colored(f"Calibration error: {error:.4f}", "dark_grey"))

    inv_camera_matrix = np.linalg.inv(camera_matrix)

    for _ in tqdm(range(total_frames), desc="Processing Video", unit="frame"):
        ret, original_frame = cap.read()
        if not ret:
            break

        # Get the undistorted frame
        # It will be used for all the processing and it will be not modified
        original_frame = get_undistorted_frame(
            frame=original_frame,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )

        proj_frame = original_frame.copy()
        laser_frame = original_frame.copy()
        black_obj_frame = find_black_objects(frame=original_frame)

        # Compute the initial contours to allow the fitting of some shapes using the edges of the black objects
        # Note that both markers and dots inside plate marker have black contours
        # Find contours of the black objects
        black_contours, _ = cv2.findContours(
            black_obj_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        if debug:
            black_obj_frame = cv2.cvtColor(black_obj_frame, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(black_obj_frame, black_contours, -1, (0, 0, 255), 2)

        # Find inner rectangle of the back marker in image coordinate system
        rectangle = fit_marker_rectangle(
            contours=black_contours, debug=debug, palette_frame=black_obj_frame
        )
        # Find the centers of the dots on the plate marker in image coordinate system
        dot_centers = find_plate_marker_cand_dot_centers(
            contours=black_contours,
            frame_w=frame_width,
            frame_h=frame_height,
            debug=debug,
            palette_frame=black_obj_frame,
        )
        # Find the ellipse of the plate marker in image coordinate system based on the dot centers
        ellipse = fit_marker_ellipse(
            points=dot_centers, debug=debug, palette_frame=black_obj_frame
        )

        # Once all the information of markers are found, extrinsic parameters can be computed
        # Find the extrinsic parameters of the back marker
        r_back, t_back = compute_back_marker_extrinsic(
            rectangle=rectangle,
            camera_matrix=camera_matrix,
            real_marker_size=back_marker_size,
            debug=debug,
            palette_frame=proj_frame,
        )
        # Find the extrinsic parameters of the plate marker
        r_plate, t_plate = compute_plate_marker_extrinsic(
            ellipse=ellipse,
            dot_centers=dot_centers,
            camera_matrix=camera_matrix,
            marker_info=plate_marker_info,
            frame=original_frame,
            debug=debug,
            palette_frame=proj_frame,
            print_fn=tqdm.write,
        )

        # Matrix to convert from camera reference system to plate marker reference system
        camera2plate_mtx = camera2marker(r_plate, t_plate)

        # Find some laser points on the back marker and plate marker
        # This is needed to fit the laser plane
        points_back = find_n_laser_point_backmarker(
            rectangle=rectangle,
            frame=original_frame,
            n_points=20,
        )
        points_plate = find_n_laser_point_platemarker(
            ellipse=ellipse,
            frame=original_frame,
            n_points=20,
        )
        # Find all the laser points on the plate marker
        # Points will be used to generate the point cloud of the object
        all_laser_points_plate = find_all_laser_points_obj(
            ellipse=ellipse,
            frame=original_frame,
        )

        if debug:
            for p in points_back:
                cv2.drawMarker(
                    laser_frame,
                    tuple(p),
                    (255, 0, 0),
                    markerSize=15,
                    markerType=cv2.MARKER_CROSS,
                    thickness=2,
                )
            for p in points_plate:
                cv2.drawMarker(
                    laser_frame,
                    tuple(p),
                    (255, 0, 0),
                    markerSize=15,
                    markerType=cv2.MARKER_CROSS,
                    thickness=2,
                )
            for p in all_laser_points_plate:
                cv2.drawMarker(
                    laser_frame,
                    tuple(p),
                    (0, 255, 255),
                    markerSize=2,
                    markerType=cv2.MARKER_STAR,
                    thickness=1,
                )

        # All the laser points found before are in image coordinate system (2D)
        # 1) Add a dimension to the points to make them 3D
        # 2) Use K^(-1) to map them from image frame to camera frame
        # 3) As a result the 3D z coordinate have normalized depth (z=1)
        # [1, 3, 3] @ [n, 3, 1]
        points_back_cam = np.expand_dims(inv_camera_matrix, axis=0) @ np.expand_dims(
            np.column_stack([points_back, np.ones(len(points_back))]), axis=2
        )
        points_plate_cam = np.expand_dims(inv_camera_matrix, axis=0) @ np.expand_dims(
            np.column_stack([points_plate, np.ones(len(points_plate))]), axis=2
        )

        # It's possible to define a plane in the markers reference system arbitrarily, as they are coplanar
        # The simplest plane is the one defined by the normal vector [0, 0, 1] and the point [0, 0, 0]
        # The plane can be converted to the camera reference system using the extrinsic parameters
        # [4, ]
        plane_back_cam = plane_marker2plane_camera(
            np.array([0, 0, 0]), np.array([0, 0, 1]), r_back, t_back
        )
        plane_plate_cam = plane_marker2plane_camera(
            np.array([0, 0, 0]), np.array([0, 0, 1]), r_plate, t_plate
        )

        # Use the lines passing trough the camera center and the laser points that
        # intersect the plane in camera reference system.
        # The intersection points are the 3D laser points on the respective plane in camera reference system
        # [n, 3, 1]
        points_back_laser = find_plane_line_intersection(
            plane_back_cam, np.array([0, 0, 0]).reshape(1, 3, 1), points_back_cam
        )
        points_back_laser = points_back_laser.squeeze(axis=-1)

        points_plate_laser = find_plane_line_intersection(
            plane_plate_cam, np.array([0, 0, 0]).reshape(1, 3, 1), points_plate_cam
        )
        points_plate_laser = points_plate_laser.squeeze(axis=-1)

        # Fit the laser plane in camera reference system using the points on the markers' planes
        laser_plane = fit_plane(
            np.concatenate([points_back_laser, points_plate_laser], axis=0)
        )

        # Map also the laser points on the plate marker (object's points)
        # from image to camera reference system
        # [1, 3, 3] @ [n, 3, 1]
        pl_cam = np.expand_dims(inv_camera_matrix, axis=0) @ np.expand_dims(
            np.column_stack(
                [all_laser_points_plate, np.ones(len(all_laser_points_plate))]
            ),
            axis=2,
        )
        # Find the intersection between the laser plane and the lines passing through
        # the camera center and the laser points
        # [n, 3, 1]
        pl_intersection = find_plane_line_intersection(
            laser_plane, np.array([0, 0, 0]).reshape(1, 3, 1), pl_cam
        )
        pl_intersection = pl_intersection.squeeze(axis=-1)

        # Convert the intersection points from camera reference system to plate marker reference system
        # to allow the correct point cloud generation
        # [1, 4, 4] @ [n, 4, 1]
        pl_w = np.expand_dims(camera2plate_mtx, axis=0) @ np.expand_dims(
            np.column_stack([pl_intersection, np.ones(len(pl_intersection))]),
            axis=2,
        )
        pl_w = pl_w.squeeze(axis=-1)
        # Extract 3d points from the homogeneous coordinates
        pl_w = pl_w[:, :3]

        with open(output_path, "a") as file:
            # Write all the object points to the output file
            for p in pl_w:
                file.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

                if debug:
                    p_proj = cv2.projectPoints(
                        p, r_plate, t_plate, camera_matrix, np.empty(0)
                    )[0]
                    p_proj = np.round(p_proj).astype(np.int32).squeeze()
                    cv2.drawMarker(
                        proj_frame,
                        tuple(p_proj),
                        (226, 43, 138),
                        markerSize=2,
                        markerType=cv2.MARKER_STAR,
                        thickness=1,
                    )
        if debug:
            black_obj_frame_resized = get_resized_frame(
                black_obj_frame,
                width=frame_width,
                height=frame_height,
                scaling_factor=window_scaling_factor,
            )
            laser_frame_resized = get_resized_frame(
                laser_frame,
                width=frame_width,
                height=frame_height,
                scaling_factor=window_scaling_factor,
            )
            proj_frame_resized = get_resized_frame(
                proj_frame,
                width=frame_width,
                height=frame_height,
                scaling_factor=window_scaling_factor,
            )
            cv2.imshow("Shapes frame", black_obj_frame_resized)
            cv2.imshow("Projection frame", proj_frame_resized)
            cv2.imshow("Laser frame", laser_frame_resized)
        else:
            original_frame_resized = get_resized_frame(
                original_frame,
                width=frame_width,
                height=frame_height,
                scaling_factor=window_scaling_factor,
            )
            cv2.imshow("Original frame", original_frame_resized)

        key = cv2.waitKey(1)
        if key == 32:  # space bar key
            tqdm.write(colored("Paused. Press any key to continue", "yellow"))
            cv2.waitKey(0)
        elif key & 0xFF == ord("q"):  # q key
            tqdm.write(colored("User interrupted the process", "red"))
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()
    print(colored("Video processing completed!", "green"))
    print(colored(f"Point cloud saved to {output_path}", "green"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that perform point cloud generation from a provided video"
    )
    parser.add_argument(
        "--debug",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable debug mode (print additional information)",
    )
    parser.add_argument(
        "--window_scaling_factor",
        type=float,
        default=0.4,
        help="Scaling factor for the window size",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="point_cloud_generation/data",
        help="Directory containing the video file",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="ball.mov",
        help="Name of the video file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="point_cloud_generation/output",
        help="Directory to save the output file",
    )
    parser.add_argument(
        "--camera_params_dir",
        type=str,
        default="camera_calibration/output",
        help="Directory containing the camera parameters file",
    )
    parser.add_argument(
        "--camera_params_name",
        type=str,
        default="camera_params.json",
        help="Name of the camera parameters file",
    )
    parser.add_argument(
        "--plate_marker_info",
        type=str,
        nargs=3,
        default=("YWMBMMCCCYWBMYWBYWBC", 4, 7.5),
        help="Information about the object marker (seq_string, min_pattern_len, marker_radius_cm)",
    )
    parser.add_argument(
        "--back_marker_size",
        type=float,
        nargs=2,
        default=(13.0, 23.0),
        help="Size of the back marker (width_cm, height_cm)",
    )

    args = parser.parse_args()
    args.plate_marker_info = tuple(args.plate_marker_info)
    args.plate_marker_info = (
        str(args.plate_marker_info[0]),
        int(args.plate_marker_info[1]),
        float(args.plate_marker_info[2]),
    )
    args.back_marker_size = tuple(args.back_marker_size)

    assert (
        len(args.plate_marker_info[0]) > 0
        and args.plate_marker_info[1] > 0
        and args.plate_marker_info[2] > 0
    )
    assert (
        args.back_marker_size[0] > 0 and args.back_marker_size[1] > 0
    ), "Invalid back marker size"

    seed_everything()
    main(args)
