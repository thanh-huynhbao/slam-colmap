import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from slam_colmap.config import load_config, resolve_project_path


def quaternion_to_rotation_matrix(
    q: np.ndarray,
) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise ValueError("Invalid zero quaternion.")

    w, x, y, z = q / norm

    return np.array([
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
        ],
        [
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
        ],
        [
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
    ])


def rotation_matrix_to_quaternion(
    rotation: np.ndarray,
) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion (w, x, y, z)."""
    m = rotation
    trace = np.trace(m)

    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        ])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(
            max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        )
        q = np.array([
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        ])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(
            max(0.0, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        )
        q = np.array([
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        ])
    else:
        s = 2.0 * np.sqrt(
            max(0.0, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        )
        q = np.array([
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        ])

    q /= np.linalg.norm(q)

    # Use a consistent sign convention.
    if q[0] < 0:
        q = -q

    return q


def find_reconstruction_model(reconstruction_dir: Path) -> Path:
    """Find the reconstruction model with the highest numeric directory index."""
    if not reconstruction_dir.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục reconstruction: {reconstruction_dir}"
        )

    def has_model_files(directory: Path) -> bool:
        return (
            (directory / "images.bin").is_file()
            or (directory / "images.txt").is_file()
        )

    # Tìm các model nằm trong thư mục được đánh số, ví dụ 0, 1, 2, ...
    numbered_models = []

    for child in reconstruction_dir.iterdir():
        if child.is_dir() and child.name.isdigit() and has_model_files(child):
            numbered_models.append(child)

    # Chọn thư mục có chỉ số lớn nhất.
    if numbered_models:
        selected_model = max(numbered_models, key=lambda path: int(path.name))
        print(f"Selected reconstruction model: {selected_model}")
        return selected_model

    # Nếu model nằm trực tiếp trong thư mục reconstruction.
    if has_model_files(reconstruction_dir):
        print(f"Selected reconstruction model: {reconstruction_dir}")
        return reconstruction_dir

    # Fallback: tìm model trong các thư mục con khác.
    candidates = sorted(
        {
            *reconstruction_dir.rglob("images.bin"),
            *reconstruction_dir.rglob("images.txt"),
        }
    )

    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy images.bin hoặc images.txt trong "
            f"{reconstruction_dir}. Hãy chạy reconstruction trước."
        )

    # Gom các file thuộc cùng model thành các thư mục duy nhất.
    model_dirs = sorted(
        {candidate.parent for candidate in candidates},
        key=lambda path: (
            int(path.name) if path.name.isdigit() else -1,
            str(path),
        ),
    )

    selected_model = model_dirs[-1]
    print(f"Selected reconstruction model: {selected_model}")
    return selected_model


def ensure_text_model(
    model_dir: Path,
    colmap_executable: str,
) -> Path:
    """Convert a binary COLMAP model to TXT when necessary."""
    if (model_dir / "images.txt").is_file():
        return model_dir

    if not (model_dir / "images.bin").is_file():
        raise FileNotFoundError(
            f"Không tìm thấy images.bin trong {model_dir}"
        )

    # The converted files only need to exist during parsing.
    # Caller should use the temporary directory as a context manager.
    raise RuntimeError(
        "Binary model conversion requires a temporary output directory. "
        "Use convert_model_to_text() instead."
    )


def convert_model_to_text(
    model_dir: Path,
    colmap_executable: str,
    output_dir: Path,
) -> Path:
    """Convert a COLMAP-compatible reconstruction model to TXT format."""
    output_dir = output_dir.expanduser().resolve()

    # COLMAP yêu cầu thư mục output phải tồn tại trước khi chuyển đổi.
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        colmap_executable,
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(output_dir),
        "--output_type",
        "TXT",
    ]

    print("Converting reconstruction model to TXT:")
    print(" ".join(command))

    subprocess.run(command, check=True)

    images_txt = output_dir / "images.txt"
    if not images_txt.is_file():
        raise FileNotFoundError(
            f"COLMAP conversion completed, but images.txt was not found: "
            f"{images_txt}"
        )

    return output_dir


def read_images_txt(images_txt: Path) -> list[dict]:
    """
    Read camera poses from COLMAP images.txt.

    COLMAP image records:
    IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME

    These quaternions and translations describe world-to-camera poses.
    """
    records = []

    with images_txt.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        fields = line.split()

        if len(fields) < 10:
            i += 1
            continue

        try:
            image_id = int(fields[0])
            q_w2c = np.array(
                [float(value) for value in fields[1:5]],
                dtype=float,
            )
            t_w2c = np.array(
                [float(value) for value in fields[5:8]],
                dtype=float,
            )
            camera_id = int(fields[8])
        except (ValueError, IndexError):
            i += 1
            continue

        image_name = " ".join(fields[9:])

        # Convert world-to-camera pose into camera-to-world pose.
        rotation_w2c = quaternion_to_rotation_matrix(q_w2c)
        rotation_c2w = rotation_w2c.T
        camera_center = -rotation_c2w @ t_w2c

        records.append({
            "image_id": image_id,
            "camera_id": camera_id,
            "image_name": image_name,
            "position": camera_center,
            "rotation": rotation_c2w,
        })

        # COLMAP images.txt alternates image records and 2D-point records.
        i += 2

    if not records:
        raise RuntimeError(
            f"Không đọc được camera poses từ {images_txt}"
        )

    # Sort lexically by image filename.
    # Timestamp filenames such as 000000000001.jpg preserve time order.
    records.sort(key=lambda record: record["image_name"])

    return records


def normalize_trajectory(
    records: list[dict],
    zero_initialize: bool = True,
) -> list[dict]:
    """
    Express all poses relative to the first camera pose.

    T_relative_i = inverse(T_world_camera_0) @ T_world_camera_i

    The first pose becomes identity when zero_initialize is True.
    """
    if not records:
        raise ValueError("Trajectory is empty.")

    first_position = records[0]["position"]
    first_rotation = records[0]["rotation"]

    normalized = []

    for record in records:
        position = record["position"]
        rotation = record["rotation"]

        if zero_initialize:
            relative_position = (
                first_rotation.T @ (position - first_position)
            )
            relative_rotation = first_rotation.T @ rotation
        else:
            relative_position = position.copy()
            relative_rotation = rotation.copy()

        quaternion = rotation_matrix_to_quaternion(
            relative_rotation
        )

        normalized.append({
            "image_id": record["image_id"],
            "image_name": record["image_name"],
            "timestamp": Path(record["image_name"]).stem,
            "tx": float(relative_position[0]),
            "ty": float(relative_position[1]),
            "tz": float(relative_position[2]),
            "qw": float(quaternion[0]),
            "qx": float(quaternion[1]),
            "qy": float(quaternion[2]),
            "qz": float(quaternion[3]),
        })

    return normalized


def save_trajectory(
    trajectory: list[dict],
    output_path: Path,
    output_format: str,
) -> None:
    """Save trajectory to CSV or TXT."""
    output_format = output_format.lower()

    if output_format not in {"csv", "txt"}:
        raise ValueError(
            "output_format chỉ hỗ trợ 'csv' hoặc 'txt'."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "image_name",
        "image_id",
        "tx",
        "ty",
        "tz",
        "qw",
        "qx",
        "qy",
        "qz",
    ]

    if output_format == "csv":
        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(trajectory)

    else:
        with output_path.open("w", encoding="utf-8") as file:
            file.write("# Camera trajectory relative to first pose\n")
            file.write("# Columns: " + " ".join(fieldnames) + "\n")

            for record in trajectory:
                values = [
                    str(record["timestamp"]),
                    str(record["image_name"]),
                    str(record["image_id"]),
                    f"{record['tx']:.12f}",
                    f"{record['ty']:.12f}",
                    f"{record['tz']:.12f}",
                    f"{record['qw']:.12f}",
                    f"{record['qx']:.12f}",
                    f"{record['qy']:.12f}",
                    f"{record['qz']:.12f}",
                ]
                file.write(" ".join(values) + "\n")

    print(f"Đã lưu trajectory: {output_path}")


def postprocess_trajectory(
    workspace: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Extract, normalize, and export a camera trajectory."""
    workspace = Path(workspace).expanduser().resolve()

    if not workspace.is_dir():
        raise FileNotFoundError(
            f"Workspace không tồn tại: {workspace}"
        )

    config = load_config()
    paths_config = config["paths"]
    post_config = config["postprocessing"]

    source = str(post_config.get("source", "glomap")).lower()
    output_format = str(
        post_config.get("output_format", "csv")
    ).lower()
    zero_initialize = bool(
        post_config.get("zero_initialize", True)
    )

    if source == "colmap":
        reconstruction_dir = (
            workspace / paths_config["colmap_output_dir"]
        )
    elif source == "glomap":
        reconstruction_dir = (
            workspace / paths_config["glomap_output_dir"]
        )
    else:
        raise ValueError(
            f"Source không hợp lệ: {source}. "
            "Chỉ chấp nhận 'colmap' hoặc 'glomap'."
        )

    model_dir = find_reconstruction_model(reconstruction_dir)

    colmap_executable = config["slam"].get(
        "colmap_executable", "colmap"
    )

    with tempfile.TemporaryDirectory(
        prefix="slam_colmap_model_"
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)

        text_model_dir = convert_model_to_text(
            model_dir=model_dir,
            colmap_executable=colmap_executable,
            output_dir=temporary_path / "text_model",
        )

        records = read_images_txt(
            text_model_dir / "images.txt"
        )

    trajectory = normalize_trajectory(
        records,
        zero_initialize=zero_initialize,
    )

    if output_path is None:
        output_root = resolve_project_path(
            paths_config.get(
                "trajectory_output_dir",
                "data/trajectory",
            )
        )
        output_path = (
            output_root
            / f"{workspace.name}.{output_format}"
        )
    else:
        output_path = Path(output_path).expanduser().resolve()

        # If the user provides a directory, generate a default filename.
        if output_path.is_dir() or not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = (
                output_path
                / f"{workspace.name}_trajectory.{output_format}"
            )

    # Ensure the extension matches the selected output format.
    output_path = output_path.with_suffix(f".{output_format}")

    save_trajectory(
        trajectory=trajectory,
        output_path=output_path,
        output_format=output_format,
    )

    print(f"Nguồn reconstruction: {source}")
    print(f"Số camera poses: {len(trajectory)}")
    print(f"Zero initialize: {zero_initialize}")

    if zero_initialize:
        first_pose = trajectory[0]
        print(
            "Pose đầu tiên sau chuẩn hóa: "
            f"t=({first_pose['tx']:.6f}, "
            f"{first_pose['ty']:.6f}, "
            f"{first_pose['tz']:.6f}), "
            f"q=({first_pose['qw']:.6f}, "
            f"{first_pose['qx']:.6f}, "
            f"{first_pose['qy']:.6f}, "
            f"{first_pose['qz']:.6f})"
        )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export COLMAP/GLOMAP camera trajectory"
    )

    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Video workspace containing COLMAP/GLOMAP results",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output file path or directory. "
            "Defaults to paths.trajectory_output_dir in params.yaml."
        ),
    )

    args = parser.parse_args()

    postprocess_trajectory(
        workspace=args.workspace,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()