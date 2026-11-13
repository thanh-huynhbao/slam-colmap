import argparse
from pathlib import Path

import cv2

from slam_colmap.config import load_config, resolve_project_path


def preprocess_insv(
    input_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Preprocess an INSV video for COLMAP/SLAM.

    Args:
        input_path: Path to the input INSV video.
        output_dir: Optional workspace directory override.
            If omitted, use paths.processed_data_dir from params.yaml.

    Returns:
        Path to the directory containing the extracted images.
    """
    config = load_config()
    paths_config = config["paths"]
    preprocess_config = config["preprocessing"]

    # Resolve input path.
    input_path = Path(input_path).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file video: {input_path}"
        )

    # Read preprocessing parameters from params.yaml.
    width = int(preprocess_config["image_width"])
    height = int(preprocess_config["image_height"])
    image_format = str(
        preprocess_config.get("image_format", "jpg")
    ).lower().lstrip(".")
    jpeg_quality = int(
        preprocess_config.get("jpeg_quality", 100)
    )

    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")

    if image_format != "jpg":
        raise ValueError(
            "This implementation currently supports JPEG output only."
        )

    if not 0 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 0 and 100.")

    # Example:
    # input:  data/raw-data/VID_20261112_150605_00_001.insv
    # output: data/VID_20261112_150605_00_001/frontlens/
    video_name = input_path.stem

    if output_dir is None:
        processed_root = resolve_project_path(
            paths_config["processed_data_dir"]
        )
        workspace_dir = processed_root / video_name
    else:
        # When specified, output_dir represents the video workspace.
        workspace_dir = Path(output_dir).expanduser().resolve()

    image_dir_name = paths_config.get("image_dir", "frontlens")
    image_dir = workspace_dir / image_dir_name

    image_dir.mkdir(parents=True, exist_ok=True)

    # Open video.
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"OpenCV không thể mở file: {input_path}. "
            "Có thể cần chuyển INSV sang MP4 trước."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0
        print(
            "Cảnh báo: Không đọc được FPS từ video. "
            "Tạm sử dụng FPS = 30."
        )

    frame_idx = 0
    saved_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Relative timestamp estimated from frame index and FPS.
            timestamp_ns = round(
                frame_idx / fps * 1_000_000_000
            )

            # Crop the LEFT half of the frame.
            frame_height, frame_width = frame.shape[:2]

            if frame_width < 2:
                raise RuntimeError(
                    f"Frame không đủ rộng để crop: "
                    f"{frame_width} pixels."
                )

            left_frame = frame[:, frame_width // 2 : frame_width]

            # Resize to configured resolution.
            resized = cv2.resize(
                left_frame,
                (width, height),
                interpolation=cv2.INTER_AREA,
            )

            # Use timestamp as image filename.
            filename = f"{timestamp_ns:012d}.jpg"
            image_path = image_dir / filename

            success = cv2.imwrite(
                str(image_path),
                resized,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )

            if not success:
                raise RuntimeError(
                    f"Không thể ghi ảnh: {image_path}"
                )

            frame_idx += 1
            saved_count += 1

    finally:
        cap.release()

    if saved_count == 0:
        raise RuntimeError(
            f"Không trích xuất được frame nào từ video: {input_path}"
        )

    print(f"Video input : {input_path}")
    print(f"Workspace   : {workspace_dir}")
    print(f"Image folder: {image_dir}")
    print(f"Resolution  : {width} x {height}")
    print(f"FPS         : {fps}")
    print(f"Saved frames: {saved_count}")

    return image_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess INSV video for COLMAP/SLAM"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input INSV video",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional video workspace directory. "
            "Defaults to the configured processed_data_dir."
        ),
    )

    args = parser.parse_args()

    preprocess_insv(
        input_path=args.input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()