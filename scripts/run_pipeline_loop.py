#!/usr/bin/env python3
"""Run preprocessing -> feature extraction -> matching -> mapping -> trajectory export
for every .insv file in an input directory.

Run from the project root:
    uv run python scripts/run_pipeline_loop.py --input-dir data/raw-data
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import traceback
from pathlib import Path
from typing import Callable

# Ensure this script can import the project's src-layout package when run from
# the repository root without requiring an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from slam_colmap.config import load_config, resolve_project_path  # noqa: E402


def load_function(module_names: tuple[str, ...], function_name: str,
                  filename_patterns: tuple[str, ...]) -> Callable:
    """Import a function from common module names, with filename fallback."""
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=[function_name])
            fn = getattr(module, function_name, None)
            if callable(fn):
                return fn
        except (ImportError, ModuleNotFoundError):
            pass

    roots = [PROJECT_ROOT / "src" / "slam_colmap", PROJECT_ROOT / "scripts", PROJECT_ROOT]
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in filename_patterns:
            for path in root.glob(pattern):
                spec = importlib.util.spec_from_file_location(
                    f"_pipeline_dynamic_{function_name}", path
                )
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    continue
                fn = getattr(module, function_name, None)
                if callable(fn):
                    return fn
    raise ImportError(f"Không tìm thấy hàm {function_name} trong các module/script của project.")


def load_postprocess_function() -> Callable:
    """Load postprocess_trajectory from common module/file naming variants."""
    return load_function(
        ("slam_colmap.post_processing", "slam_colmap.postprocessing", "slam_colmap.post_process"),
        "postprocess_trajectory",
        ("*post*process*.py",),
    )


def run_one(video_path: Path, processed_root: Path, trajectory_root: Path,
            postprocess: Callable, force: bool = False) -> Path:
    workspace = processed_root / video_path.stem
    output_format = str(load_config().get("postprocessing", {}).get("output_format", "csv")).lower()
    trajectory_path = trajectory_root / f"{video_path.stem}_trajectory.{output_format}"

    print("\n" + "=" * 88)
    print(f"VIDEO: {video_path.name}")
    print(f"WORKSPACE: {workspace}")
    print(f"TRAJECTORY OUTPUT: {trajectory_path}")
    print("=" * 88)

    if force and workspace.exists():
        print(f"[FORCE] Xóa workspace cũ: {workspace}")
        shutil.rmtree(workspace)
    if force and trajectory_path.exists():
        trajectory_path.unlink()

    # 1. Decode/crop/resize the INSV into workspace/frontlens.
    preprocess_insv(video_path, output_dir=workspace)

    # Avoid accidentally using an old database/reconstruction when re-running.
    # --force is recommended when reprocessing the same input video.
    if not force:
        database_name = load_config()["paths"].get("database_name", "database.db")
        stale_db = workspace / database_name
        if stale_db.exists():
            print(
                f"[WARN] Đã có {stale_db}. COLMAP có thể tái sử dụng database cũ; "
                "dùng --force nếu muốn chạy sạch từ đầu."
            )

    # 2. COLMAP/GLOMAP pipeline.
    extract_features(workspace)
    sequential_match(workspace)
    run_mapper(workspace)

    # 3. Export normalized trajectory to a per-video output file.
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    result = postprocess(workspace, output_path=trajectory_path)
    result = Path(result).resolve()
    if not result.is_file():
        raise FileNotFoundError(f"Post-processing không tạo được trajectory: {result}")
    print(f"[OK] Hoàn tất {video_path.name} -> {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full SLAM pipeline for each .insv file in a directory."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Folder containing .insv files. Defaults to paths.raw_data_dir in params.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional trajectory output directory. Defaults to paths.trajectory_output_dir.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Optional root for per-video workspaces. Defaults to paths.processed_data_dir.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete each target workspace and matching trajectory before processing it again.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the next video if one video fails.",
    )
    args = parser.parse_args()

    config = load_config()
    paths = config["paths"]

    input_dir = (
        args.input_dir.expanduser().resolve()
        if args.input_dir
        else resolve_project_path(paths.get("raw_data_dir", "data/raw-data"))
    )
    processed_root = (
        args.processed_dir.expanduser().resolve()
        if args.processed_dir
        else resolve_project_path(paths.get("processed_data_dir", "data"))
    )
    trajectory_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else resolve_project_path(paths.get("trajectory_output_dir", "data/trajectory"))
    )

    if not input_dir.is_dir():
        parser.error(f"Input directory không tồn tại: {input_dir}")

    videos = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".insv"
    )
    if not videos:
        print(f"Không tìm thấy file .insv nào trong: {input_dir}")
        return 1

    print(f"Input folder     : {input_dir}")
    print(f"Processed folder : {processed_root}")
    print(f"Trajectory folder: {trajectory_root}")
    print(f"Số video         : {len(videos)}")

    preprocess = load_function(
        ("slam_colmap.preprocessing", "slam_colmap.pre_processing"),
        "preprocess_insv",
        ("*pre*process*.py",),
    )
    extract = load_function(("slam_colmap.slam",), "extract_features", ("slam.py",))
    match = load_function(("slam_colmap.slam",), "sequential_match", ("slam.py",))
    mapper = load_function(("slam_colmap.slam",), "run_mapper", ("slam.py",))
    postprocess = load_postprocess_function()
    # Bind the dynamically discovered functions used by run_one.
    global preprocess_insv, extract_features, sequential_match, run_mapper
    preprocess_insv, extract_features, sequential_match, run_mapper = preprocess, extract, match, mapper

    successes: list[tuple[Path, Path]] = []
    failures: list[tuple[Path, str]] = []

    for video in videos:
        try:
            result = run_one(
                video, processed_root, trajectory_root, postprocess, force=args.force
            )
            successes.append((video, result))
        except Exception as exc:
            failures.append((video, str(exc)))
            print(f"[ERROR] {video.name}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                traceback.print_exc()
                print("Dừng pipeline. Dùng --continue-on-error để tiếp tục video tiếp theo.")
                break

    print("\n" + "=" * 88)
    print(f"KẾT QUẢ: {len(successes)} thành công / {len(failures)} lỗi")
    for video, result in successes:
        print(f"[OK]   {video.name} -> {result}")
    for video, error in failures:
        print(f"[FAIL] {video.name}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
