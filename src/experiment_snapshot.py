"""Save the source, settings, and environment associated with an experiment."""

import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile


DINO_ENVIRONMENT = (
    "DINOV3_REPO",
    "DINOV3_VITB16_WEIGHTS",
    "DINOV3_VITL16_WEIGHTS",
    "DINOTXT_WEIGHTS",
    "DINOTXT_BPE",
)


def capture_snapshot(project_root, snapshot, config_bytes):
    """Archive the exact config read by the runner and the current local source."""
    (snapshot / "config.yaml").write_bytes(config_bytes)
    for name in (".python-version", "pyproject.toml", "uv.lock"):
        copyfile(project_root / name, snapshot / name)
    for directory in ("src", "scripts"):
        for source in (project_root / directory).rglob("*.py"):
            destination = snapshot / source.relative_to(project_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            copyfile(source, destination)

    environment = {}
    for name in DINO_ENVIRONMENT:
        value = os.environ.get(name)
        if value:
            # Signed checkpoint URLs can contain credentials.
            environment[name] = "<HTTPS URL omitted>" if value.startswith("https://") else value
    return {
        "working_directory": str(Path.cwd()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {name: version(name) for name in ("numpy", "scikit-learn", "torch")},
        "dino_environment": environment,
    }


def finish_snapshot(snapshot, result, runtime, device, artifact_path):
    manifest = {
        **result,
        "runtime": {**runtime, "device": device},
        "artifact_path_relative_to_results": artifact_path.as_posix(),
    }
    (snapshot / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
