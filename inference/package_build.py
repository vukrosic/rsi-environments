"""Small PEP 517 backend for a curated, namespaced inference wheel.

The source experiment deliberately keeps its scripts flat for reproducibility.
The wheel places private copies below ``inference_env._runtime`` and rewrites
only their local imports, so installing this package never exports generic
``scorer`` or ``assets`` modules into the interpreter's top-level namespace.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
from pathlib import Path
import re
import tarfile
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parent
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]
PACKAGE_NAME = PROJECT["name"]
VERSION = PROJECT["version"]
DIST_INFO = f"vuk_rsi_inference-{VERSION}.dist-info"
RUNTIME_MODULES = (
    "assets.py", "attempt.py", "cache_probes.py", "controller.py", "interface.py",
    "reference.py", "replay.py", "rl_environment.py", "scorer.py",
    "shared_contract.py", "solution.py", "supervisor.py", "worker.py",
)
RUNTIME_DATA = (
    "assets-manifest.json", "model.json", "requirements-lock.txt",
    "requirements.txt", "tasks.json", "TASK.md", "DESIGN.md", "RL-CONTRACT.md",
)
PACKAGE_INIT = """\"\"\"Private runtime for the packaged RSI inference environment.\"\"\"\n"""


def _metadata() -> str:
    project = PROJECT
    lines = [
        "Metadata-Version: 2.3",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        f"Summary: {project['description']}",
        f"Requires-Python: {project['requires-python']}",
    ]
    for dependency in project.get("dependencies", []):
        lines.append(f"Requires-Dist: {dependency}")
    for classifier in project.get("classifiers", []):
        lines.append(f"Classifier: {classifier}")
    lines.extend(["", "This package supports macOS with CPU or Apple MPS inference."])
    return "\n".join(lines)


def _entry_points() -> str:
    return "[console_scripts]\nrsi-inference = inference_env.cli:main\n"


def _wheel_metadata(directory: Path) -> None:
    dist = directory / DIST_INFO
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(_metadata(), encoding="utf-8")
    (dist / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: vuk-rsi-inference-package-build\n"
        "Root-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8",
    )
    (dist / "entry_points.txt").write_text(_entry_points(), encoding="utf-8")


def _rewrite_imports(source: str) -> str:
    """Make checked-in flat-script imports private-package aware.

    A conditional keeps each evaluator snapshot runnable as a plain script
    after scorer.py copies it into a run directory.
    """
    replacements = {
        "assets": (".assets", "assets"),
        "cache_probes": (".cache_probes", "cache_probes"),
        "scorer": (".scorer", "scorer"),
        "shared_contract": (".shared_contract", "shared_contract"),
        "supervisor": (".supervisor", "supervisor"),
        "shared.resources": (".shared.resources", "shared.resources"),
        "shared.records": (".shared.records", "shared.records"),
    }
    output: list[str] = []
    pattern = re.compile(r"^(\s*)from (assets|cache_probes|scorer|shared_contract|supervisor|shared\.(?:resources|records)) import (.+)$")
    for line in source.splitlines(keepends=True):
        match = pattern.match(line.rstrip("\r\n"))
        if not match:
            output.append(line)
            continue
        indent, name, symbols = match.groups()
        package_import, script_import = replacements[name]
        newline = "\n" if line.endswith("\n") else ""
        output.append(f"{indent}if __package__:\n")
        output.append(f"{indent}    from {package_import} import {symbols}\n")
        output.append(f"{indent}else:\n")
        output.append(f"{indent}    from {script_import} import {symbols}{newline}")
    return "".join(output)


def _stage(destination: Path) -> list[Path]:
    package = destination / "inference_env"
    runtime = package / "_runtime"
    shared = runtime / "shared"
    shared.mkdir(parents=True)
    (package / "__init__.py").write_text(
        '"""Versioned RSI inference environment.\"\"\"\n'
        "from importlib.metadata import PackageNotFoundError, version\n"
        "try:\n    __version__ = version('vuk-rsi-inference')\n"
        f"except PackageNotFoundError:  # source-tree imports\n    __version__ = '{VERSION}'\n"
        "__all__ = ['CodingAgentEnvironment', 'InferenceEnvironment', '__version__']\n"
        "def __getattr__(name):\n"
        "    if name == 'CodingAgentEnvironment':\n"
        "        from ._runtime.rl_environment import CodingAgentEnvironment\n"
        "        return CodingAgentEnvironment\n"
        "    if name == 'InferenceEnvironment':\n"
        "        from ._runtime.interface import InferenceEnvironment\n"
        "        return InferenceEnvironment\n"
        "    raise AttributeError(name)\n",
        encoding="utf-8",
    )
    (package / "cli.py").write_text((ROOT / "inference_env" / "cli.py").read_text("utf-8"), encoding="utf-8")
    (package / "_runtime" / "__init__.py").write_text(PACKAGE_INIT, encoding="utf-8")
    included: list[Path] = [package / "__init__.py", package / "cli.py", runtime / "__init__.py"]
    for name in RUNTIME_MODULES:
        source = ROOT / name
        if not source.is_file():
            raise FileNotFoundError(f"required inference runtime module is missing: {source}")
        target = runtime / name
        target.write_text(_rewrite_imports(source.read_text("utf-8")), encoding="utf-8")
        included.append(target)
    for name in RUNTIME_DATA:
        source = ROOT / name
        if not source.is_file():
            raise FileNotFoundError(f"required inference runtime data is missing: {source}")
        target = runtime / name
        target.write_bytes(source.read_bytes())
        included.append(target)
    for name in ("__init__.py", "records.py", "resources.py"):
        source = ROOT / "shared" / name
        if not source.is_file():
            raise FileNotFoundError(f"required MIT shared helper is missing: {source}")
        target = shared / name
        target.write_bytes(source.read_bytes())
        included.append(target)
    return included


def _record(path: str, content: bytes) -> tuple[str, str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
    return path, f"sha256={digest}", str(len(content))


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    del config_settings, metadata_directory
    output = Path(wheel_directory)
    output.mkdir(parents=True, exist_ok=True)
    wheel_name = f"vuk_rsi_inference-{VERSION}-py3-none-any.whl"
    with tempfile.TemporaryDirectory(prefix="rsi-wheel-stage-") as temporary:
        stage = Path(temporary)
        files = _stage(stage)
        _wheel_metadata(stage)
        dist = stage / DIST_INFO
        files.extend(dist / name for name in ("METADATA", "WHEEL", "entry_points.txt"))
        record_path = f"{DIST_INFO}/RECORD"
        rows = []
        for path in sorted(files, key=lambda item: item.relative_to(stage).as_posix()):
            arcname = path.relative_to(stage).as_posix()
            rows.append(_record(arcname, path.read_bytes()))
        rows.append((record_path, "", ""))
        record = io.StringIO(newline="")
        csv.writer(record, lineterminator="\n").writerows(rows)
        (dist / "RECORD").write_text(record.getvalue(), encoding="utf-8")
        from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
        with ZipFile(output / wheel_name, "w", compression=ZIP_DEFLATED) as archive:
            for path in sorted(files + [dist / "RECORD"], key=lambda item: item.relative_to(stage).as_posix()):
                arcname = path.relative_to(stage).as_posix()
                info = ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
    return wheel_name


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    del config_settings
    directory = Path(metadata_directory) / DIST_INFO
    _wheel_metadata(directory.parent)
    return DIST_INFO


def get_requires_for_build_wheel(config_settings=None):
    del config_settings
    return []


def build_sdist(sdist_directory, config_settings=None):
    del config_settings
    output = Path(sdist_directory)
    output.mkdir(parents=True, exist_ok=True)
    name = f"vuk-rsi-inference-{VERSION}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="rsi-sdist-stage-") as temporary:
        staged = Path(temporary) / f"vuk-rsi-inference-{VERSION}"
        staged.mkdir()
        for filename in ("pyproject.toml", "package_build.py", "MANIFEST.in", "PACKAGING.md", "README.md"):
            path = ROOT / filename
            if path.is_file():
                (staged / filename).write_bytes(path.read_bytes())
        for filename in RUNTIME_MODULES + RUNTIME_DATA:
            (staged / filename).write_bytes((ROOT / filename).read_bytes())
        (staged / "inference_env").mkdir()
        for filename in ("__init__.py", "cli.py"):
            source = ROOT / "inference_env" / filename
            (staged / "inference_env" / filename).write_bytes(source.read_bytes())
        (staged / "shared").mkdir()
        for filename in ("__init__.py", "records.py", "resources.py"):
            source = ROOT / "shared" / filename
            (staged / "shared" / filename).write_bytes(source.read_bytes())
        with tarfile.open(output / name, "w:gz") as archive:
            archive.add(staged, arcname=staged.name)
    return name


def get_requires_for_build_sdist(config_settings=None):
    del config_settings
    return []
