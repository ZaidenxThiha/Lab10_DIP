from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from lane_detector import process_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise 1: Detect lanes in a driving video and render annotated output."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Lab10_test2.mp4"),
        help="Path to the input video",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("image_01_01.mp4"),
        help="Path to the output video (MP4).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show frames while processing. Press 'q' to quit early.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Cannot find input video: {args.input}")

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {args.output}")

    frame_counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = process_frame(frame)
            writer.write(processed)
            frame_counter += 1
            if args.display:
                cv2.imshow("Lane Detection", processed)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    if frame_counter == 0:
        raise RuntimeError("No frames were processed from the input video.")
    print(f"Processed {frame_counter} frames. Output saved to {args.output}")


if __name__ == "__main__":
    main()
