from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from lane_detector import process_frame

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise 2: Apply lane detection to each image in a folder."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("frames"),
        help="Directory that holds the input frames.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("image_02_outputs"),
        help="Directory to store the annotated outputs.",
    )
    return parser.parse_args()


def iter_images(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")
    files = [
        path
        for path in sorted(directory.iterdir())
        if path.suffix.lower() in ALLOWED_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(
            f"No image files with extensions {sorted(ALLOWED_EXTENSIONS)} in {directory}"
        )
    return files


def process_images(input_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_count = 0
    for index, image_path in enumerate(iter_images(input_dir), start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable file: {image_path}")
            continue
        annotated = process_frame(image)
        output_path = output_dir / f"image_02_{index:02d}.png"
        cv2.imwrite(str(output_path), annotated)
        processed_count += 1
    return processed_count


def main() -> None:
    args = parse_args()
    count = process_images(args.input_dir, args.output_dir)
    if count == 0:
        raise RuntimeError("No images were processed.")
    print(f"Processed {count} images. Outputs saved under {args.output_dir}")


if __name__ == "__main__":
    main()
