"""Build a deterministic, data-free source archive for kernel-env."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tomllib
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "README.md",
    "RL_API.md",
    "CHANGELOG.md",
    "LICENSE",
    "pyproject.toml",
    "task_catalog.json",
    "STANDARD-PLAN.md",
)
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "build",
    "episodes",
    "dist",
}


def collect_source_files(project_root: Path = PROJECT_ROOT) -> list[tuple[str, Path]]:
    """Collect only files covered by the source-release allowlist."""
    files: dict[str, Path] = {}

    def add(path: Path) -> None:
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(project_root).as_posix()
            files[relative] = path

    for name in ROOT_FILES:
        add(project_root / name)
    kernel_env = project_root / "kernel_env"
    if kernel_env.exists():
        for path in kernel_env.rglob("*.py"):
            if _is_source_file_for_root(path, project_root):
                add(path)
    for folder in ("starters", "tests", "examples"):
        base = project_root / folder
        if base.exists():
            for path in base.glob("*.py"):
                if _is_source_file_for_root(path, project_root):
                    add(path)
    suite = project_root / "suite"
    if suite.exists():
        for path in suite.glob("*.py"):
            if _is_source_file_for_root(path, project_root):
                add(path)
        add(suite / "LICENSE")
    return [(name, files[name]) for name in sorted(files)]


def _is_source_file_for_root(path: Path, project_root: Path) -> bool:
    if not path.is_file() or path.is_symlink() or path.suffix != ".py":
        return False
    parts = path.relative_to(project_root).parts
    return not any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in parts)


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    return info


def build_source_release(output: Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Write a reproducible source archive, refusing to replace an existing file."""
    output = output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)

    with (project_root / "pyproject.toml").open("rb") as config_file:
        version = tomllib.load(config_file)["project"]["version"]
    source_files = collect_source_files(project_root)
    archive_files = [
        (f"kernel-env/{relative}", path.read_bytes()) for relative, path in source_files
    ]
    manifest = {
        "version": version,
        "files": [
            {"path": name, "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in archive_files
        ],
    }
    archive_files.append(
        (
            "kernel-env/SOURCE-MANIFEST.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    )

    # Exclusive creation guarantees an existing release is never overwritten.
    with output.open("xb") as archive_file:
        with ZipFile(archive_file, mode="w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for name, content in sorted(archive_files):
                archive.writestr(_zip_info(name), content, compress_type=ZIP_DEFLATED, compresslevel=9)
    return output


def main() -> int:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as config_file:
        version = tomllib.load(config_file)["project"]["version"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dist" / f"kernel-env-source-{version}.zip",
        help="archive path (default: dist/kernel-env-source-VERSION.zip)",
    )
    args = parser.parse_args()
    try:
        archive = build_source_release(args.output)
    except FileExistsError:
        parser.error(f"refusing to overwrite existing archive: {args.output}")
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
