"""Verify a packaged training environment using an explicitly supplied Python.

This is a same-machine source-package reproduction check, not a fresh-machine
install or an outer-policy training run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile


FORBIDDEN_SUFFIXES = {".npy", ".pt", ".pth", ".log", ".jsonl", ".csv", ".bin"}
FORBIDDEN_PARTS = {"data", "datasets", "private", "agent-logs", "agent_logs", "traces"}
REQUIRED_FILES = {
    "training-env/prepare.py", "training-env/interaction.py", "training-env/starter.py",
    "training-env/data.py", "training-env/evaluate.py", "training-env/worker.py",
    "training-env/grader.py", "training-env/checks.py", "training-env/protocol.py",
    "training-env/requirements.txt", "suite/__init__.py", "suite/records.py",
    "suite/resources.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    files = []
    names = set()
    for member in members:
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts or not name.parts:
            raise ValueError(f"unsafe archive path: {member.name!r}")
        if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
            raise ValueError(f"unsupported archive entry: {member.name!r}")
        if member.isfile():
            relative = "/".join(name.parts[1:]) if name.parts and name.parts[0] == "rsi" else "/".join(name.parts)
            lower_parts = {part.lower() for part in name.parts}
            if Path(name.name).suffix.lower() in FORBIDDEN_SUFFIXES or lower_parts & FORBIDDEN_PARTS:
                raise ValueError(f"archive contains prohibited data or logs: {member.name!r}")
            if relative in names:
                raise ValueError(f"duplicate archive member: {member.name!r}")
            names.add(relative)
            files.append(member)
    if not REQUIRED_FILES.issubset(names):
        raise ValueError(f"archive is missing required source files: {sorted(REQUIRED_FILES - names)}")
    return files


def extract_safely(archive: tarfile.TarFile, members: list[tarfile.TarInfo], root: Path) -> list[Path]:
    written = []
    for member in members:
        parts = PurePosixPath(member.name).parts
        relative = Path(*parts[1:]) if parts and parts[0] == "rsi" else Path(*parts)
        target = (root / relative).resolve()
        if root.resolve() not in target.parents:
            raise ValueError(f"archive path escapes extraction root: {member.name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"cannot read archive member: {member.name!r}")
        with source, target.open("xb") as output:
            output.write(source.read())
        written.append(target)
    return written


def run(python: Path, cwd: Path, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(python), *args], cwd=cwd, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=150)


def require_ok(result: subprocess.CompletedProcess[str], label: str) -> str:
    if result.returncode:
        raise RuntimeError(f"{label} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path, help="interpreter in the fresh environment")
    parser.add_argument("--output", required=True, type=Path, help="new, empty destination for extraction and receipt")
    args = parser.parse_args()
    archive_path, output = (p.expanduser().resolve() for p in (args.archive, args.output))
    # Keep the venv interpreter spelling: resolving its symlink bypasses venv discovery.
    python = Path(os.path.abspath(args.python.expanduser()))
    if not archive_path.is_file() or not python.is_file():
        parser.error("--archive and --python must name existing files")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    source_root = output / "extracted"
    source_root.mkdir()

    with tarfile.open(archive_path, "r:*") as package:
        members = inspect_members(package)
        extracted = extract_safely(package, members, source_root)
    env_dir, suite_parent = source_root / "training-env", source_root
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = str(suite_parent)
    run_env.pop("PYTHONHOME", None)
    run_env["PYTHONNOUSERSITE"] = "1"

    versions = require_ok(run(python, env_dir, ["-c", "import json,sys,numpy,torch,tokenizers; print(json.dumps({'python':sys.version.split()[0],'prefix':sys.prefix,'base_prefix':sys.base_prefix,'numpy':numpy.__version__,'torch':torch.__version__,'tokenizers':tokenizers.__version__},sort_keys=True))"], run_env), "dependency version check")
    dependencies = json.loads(versions.strip().splitlines()[-1])
    smoke_dir = output / "smoke-assets"
    prepared = require_ok(run(python, env_dir, ["prepare.py", "--synthetic", "--output", str(smoke_dir)], run_env), "synthetic preparation")
    prep_info = json.loads(prepared.strip().splitlines()[-1])
    if prep_info.get("kind") != "synthetic":
        raise RuntimeError("synthetic preparation did not produce a synthetic manifest")

    standard_dir = output / "standard-synthetic-rejection"
    standard_dir.mkdir()
    standard_run = run(python, env_dir, ["evaluate.py", "--data", str(smoke_dir), "--output", str(standard_dir / "receipt"), "--profile", "standard", "--device", "cpu"], run_env)
    standard_record_path = standard_dir / "receipt" / "record.json"
    if not standard_record_path.is_file():
        raise RuntimeError(f"standard synthetic rejection produced no normalized record: {standard_run.stderr}")
    standard_record = json.loads(standard_record_path.read_text())
    if standard_record.get("status") != "blocked" or "synthetic" not in standard_record.get("metrics", {}).get("reason", "").lower():
        raise RuntimeError(f"standard profile did not reject synthetic data as expected: {standard_record}")

    episode_dir = output / "terminal-episode"
    action_code = (
        "import json,sys; from pathlib import Path; from interaction import TrainingEnv; "
        "env=TrainingEnv(sys.argv[1],sys.argv[2],profile='smoke',device='cpu'); "
        "print(json.dumps(env.step({'code':Path(sys.argv[3]).read_text(),'final':True}),allow_nan=False))"
    )
    episode = require_ok(run(python, env_dir, ["-c", action_code, str(episode_dir / "session"),
                          str(smoke_dir), str(env_dir / "starter.py")], run_env), "terminal CPU code-action episode")
    episode_result = json.loads(episode.strip().splitlines()[-1])
    record = episode_result.get("record", {})
    if not episode_result.get("terminated") or episode_result.get("truncated") or record.get("status") != "ok":
        raise RuntimeError(f"terminal code-action episode did not complete successfully: {episode_result}")
    if record.get("profile") != "smoke" or record.get("provenance", {}).get("data_kind") != "synthetic":
        raise RuntimeError("terminal normalized record does not identify the synthetic smoke run")
    if record.get("schema_version") != 1 or record.get("task") != "tinystories-next-token":
        raise RuntimeError("terminal normalized record has unexpected schema or task")

    source_hashes = {str(path.relative_to(source_root)): sha256(path) for path in sorted(extracted)}
    receipt = {
        "status": "ok", "archive": str(archive_path), "archive_sha256": sha256(archive_path),
        "python": str(python), "dependencies": dependencies, "extracted_source_sha256": source_hashes,
        "synthetic_preparation": prep_info, "standard_synthetic_rejection": standard_record,
        "terminal_episode_record": record, "terminal_episode_trainable": episode_result.get("info", {}).get("trainable"),
        "outer_policy_trained": False, "device_used": "cpu", "scope": "same-Mac source-package reproduction; no fresh-machine install",
        "raw_data_or_private_logs_in_archive": False,
    }
    receipt_path = output / "reproduction-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "receipt": str(receipt_path), "archive_sha256": receipt["archive_sha256"],
                      "terminal_status": record["status"], "terminal_loss": record["metrics"].get("held_out_loss"),
                      "terminal_reward": record["reward"], "standard_synthetic_status": standard_record["status"],
                      "outer_policy_trained": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
