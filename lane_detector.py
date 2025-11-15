from __future__ import annotations

import cv2
import numpy as np

SID_TEXT = "SID: 523K0073"


def _canny_edges(image: np.ndarray) -> np.ndarray:
    """Convert frame to grayscale, denoise, then run Canny."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)


def _region_of_interest(image: np.ndarray) -> np.ndarray:
    """Mask everything outside a triangular region covering the lanes."""
    height, width = image.shape[:2]
    polygon = np.array(
        [
            [
                (int(0.1 * width), height),
                (int(0.9 * width), height),
                (int(0.55 * width), int(0.6 * height)),
                (int(0.45 * width), int(0.6 * height)),
            ]
        ]
    )
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)


def _make_coordinates(image: np.ndarray, slope_intercept: tuple[float, float]) -> np.ndarray:
    slope, intercept = slope_intercept
    height = image.shape[0]
    y1 = height
    y2 = int(height * 0.62)
    if slope == 0:
        slope = 0.01
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def _average_slope_intercept(image: np.ndarray, lines: np.ndarray | None) -> np.ndarray:
    """Average out line segments detected via Hough to stabilize the result."""
    if lines is None:
        return np.array([])

    left_fits = []
    right_fits = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if slope < -0.4:
            left_fits.append((slope, intercept))
        elif slope > 0.4:
            right_fits.append((slope, intercept))

    averaged_lines = []
    if left_fits:
        left_fit_average = np.mean(left_fits, axis=0)
        averaged_lines.append(_make_coordinates(image, left_fit_average))
    if right_fits:
        right_fit_average = np.mean(right_fits, axis=0)
        averaged_lines.append(_make_coordinates(image, right_fit_average))

    return np.array(averaged_lines)


def _draw_lines(image: np.ndarray, lines: np.ndarray) -> np.ndarray:
    line_image = np.zeros_like(image)
    if lines is None or len(lines) == 0:
        return line_image
    for x1, y1, x2, y2 in lines:
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image


def add_sid_text(image: np.ndarray) -> np.ndarray:
    result = image.copy()
    cv2.putText(
        result,
        SID_TEXT,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Detect and draw lane lines on a single frame.

    Returns a new frame with the detected lane overlay and SID text.
    """
    lane_image = np.copy(frame)
    edges = _canny_edges(lane_image)
    cropped = _region_of_interest(edges)
    lines = cv2.HoughLinesP(
        cropped,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=120,
    )
    averaged_lines = _average_slope_intercept(lane_image, lines)
    line_image = _draw_lines(lane_image, averaged_lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    return add_sid_text(combo_image)
