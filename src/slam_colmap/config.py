from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "params.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load configuration from params.yaml."""

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid configuration: expected a YAML mapping.")

    return config


def resolve_project_path(path: str) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""

    resolved_path = Path(path).expanduser()

    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    return resolved_path.resolve()


def get_workspace_paths(workspace: Path) -> dict[str, Path]:
    """Resolve image, database, and reconstruction paths for a video."""

    workspace = workspace.expanduser().resolve()
    config = load_config()
    paths = config["paths"]

    return {
        "workspace": workspace,
        "image_path": workspace / paths["image_dir"],
        "database_path": workspace / paths["database_name"],
        "colmap_output_path": workspace / paths["colmap_output_dir"],
        "glomap_output_path": workspace / paths["glomap_output_dir"],
    }