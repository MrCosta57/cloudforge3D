import cv2
import numpy as np


def find_laser_trace(frame: np.ndarray) -> np.ndarray:
    # HSV color space is more suitable for object detection
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Smooth frame with filter robust to outliers, making it useful for removing impulse noise or salt-and-pepper noise/reflections
    frame_hsv = cv2.medianBlur(frame_hsv, 3)
    # Red can have two distinct hue ranges in the HSV space (0-10 and 160-180)
    # Thresholds here are fixed to take into account some color variations
    mask1 = cv2.inRange(frame_hsv, np.array((0, 70, 230)), np.array((15, 255, 255)))
    mask2 = cv2.inRange(frame_hsv, np.array((155, 70, 230)), np.array((180, 255, 255)))
    red_mask = cv2.bitwise_or(mask1, mask2)
    return red_mask
