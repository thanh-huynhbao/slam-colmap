import argparse
import subprocess
from pathlib import Path

from slam_colmap.config import get_workspace_paths, load_config


def run_command(command: list[str]) -> None:
    """Execute a command and raise an error if it fails."""
    print("\nRunning command:")
    print(" ".join(command), "\n")

    subprocess.run(command, check=True)


def get_slam_config() -> dict:
    """Load SLAM configuration from params.yaml."""
    config = load_config()
    return config["slam"]


def format_camera_params(params: list | str | None) -> str | None:
    """Convert camera parameters to a comma-separated string."""
    if params is None:
        return None

    if isinstance(params, str):
        return params

    return ",".join(str(value) for value in params)


def validate_workspace(workspace: Path) -> dict[str, Path]:
    """Resolve and validate paths for a video workspace."""
    if not workspace.is_dir():
        raise FileNotFoundError(
            f"Workspace không tồn tại: {workspace}"
        )

    paths = get_workspace_paths(workspace)

    if not paths["image_path"].is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh: {paths['image_path']}. "
            "Hãy chạy preprocessing trước."
        )

    image_files = [
        path
        for path in paths["image_path"].iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    ]

    if not image_files:
        raise RuntimeError(
            f"Không tìm thấy ảnh trong: {paths['image_path']}"
        )

    return paths


def extract_features(
    workspace: Path,
) -> None:
    """Extract SIFT features using configuration from params.yaml."""
    paths = validate_workspace(workspace)
    config = get_slam_config()

    database_path = paths["database_path"]
    image_path = paths["image_path"]

    database_path.parent.mkdir(parents=True, exist_ok=True)

    camera_params = format_camera_params(
        config.get("camera_params")
    )

    command = [
        config.get("colmap_executable", "colmap"),
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--ImageReader.camera_model",
        str(config.get("camera_model", "OPENCV_FISHEYE")),
        "--ImageReader.single_camera",
        str(int(config.get("single_camera", True))),
        "--FeatureExtraction.use_gpu",
        str(int(config.get("use_gpu", True))),
        "--SiftExtraction.max_num_features",
        str(int(config.get("max_num_features", 2000))),
    ]

    if camera_params is not None:
        command.extend([
            "--ImageReader.camera_params",
            camera_params,
        ])

    run_command(command)


def sequential_match(
    workspace: Path,
) -> None:
    """Match features between sequential images."""
    paths = validate_workspace(workspace)
    config = get_slam_config()

    database_path = paths["database_path"]

    if not database_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy database: {database_path}. "
            "Hãy chạy extract trước."
        )

    overlap = int(
        config.get("sequential_matching_overlap", 5)
    )

    if overlap < 1:
        raise ValueError(
            "sequential_matching_overlap phải >= 1."
        )

    command = [
        config.get("colmap_executable", "colmap"),
        "sequential_matcher",
        "--database_path",
        str(database_path),
        "--SequentialMatching.overlap",
        str(overlap),
    ]

    run_command(command)


def run_mapper(
    workspace: Path,
) -> None:
    """Run COLMAP or GLOMAP mapper using params.yaml."""
    paths = validate_workspace(workspace)
    config = get_slam_config()

    database_path = paths["database_path"]
    image_path = paths["image_path"]

    if not database_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy database: {database_path}. "
            "Hãy chạy extract và match trước."
        )

    mapper = str(config.get("mapper", "glomap")).lower()

    if mapper == "colmap":
        executable = config.get("colmap_executable", "colmap")
        output_path = paths["colmap_output_path"]
        subcommand = "mapper"

    elif mapper == "glomap":
        executable = config.get("glomap_executable", "glomap")
        output_path = paths["glomap_output_path"]
        subcommand = "mapper"

    else:
        raise ValueError(
            f"Mapper không được hỗ trợ: {mapper}. "
            "Chỉ chấp nhận 'colmap' hoặc 'glomap'."
        )

    output_path.mkdir(parents=True, exist_ok=True)

    command = [
        executable,
        subcommand,
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(output_path),
    ]

    run_command(command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "COLMAP/GLOMAP pipeline for a preprocessed video workspace"
        )
    )

    subparsers = parser.add_subparsers(
        dest="action",
        required=True,
    )

    # Shared workspace argument.
    def add_workspace_argument(subparser):
        subparser.add_argument(
            "--workspace",
            type=Path,
            required=True,
            help=(
                "Video workspace containing frontlens/, "
                "database.db and reconstruction outputs"
            ),
        )

    # 1. Feature extraction.
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract SIFT features",
    )
    add_workspace_argument(extract_parser)

    # 2. Sequential matching.
    match_parser = subparsers.add_parser(
        "match",
        help="Match sequential images",
    )
    add_workspace_argument(match_parser)

    # 3. Mapping.
    mapper_parser = subparsers.add_parser(
        "mapper",
        help="Run COLMAP or GLOMAP reconstruction",
    )
    add_workspace_argument(mapper_parser)

    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()

    if args.action == "extract":
        extract_features(workspace)

    elif args.action == "match":
        sequential_match(workspace)

    elif args.action == "mapper":
        run_mapper(workspace)


if __name__ == "__main__":
    main()