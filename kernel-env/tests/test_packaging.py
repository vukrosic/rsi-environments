import hashlib
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "build_source_release.py"
SPEC = importlib.util.spec_from_file_location("build_source_release", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def make_source_tree(root: Path) -> dict[str, bytes]:
    files = {
        "README.md": b"# Fixture\n",
        "RL_API.md": b"API\n",
        "CHANGELOG.md": b"Changes\n",
        "LICENSE": b"Project license\n",
        "pyproject.toml": b'[project]\nname = "fixture"\nversion = "4.5.6"\n',
        "task_catalog.json": b"{}\n",
        "kernel_env/__init__.py": b"PACKAGE = True\n",
        "kernel_env/submodule/task.py": b"TASK = 1\n",
        "suite/__init__.py": b"SUITE = True\n",
        "suite/LICENSE": b"Suite license\n",
        "starters/softmax.py": b"def starter(): return None\n",
        "tests/test_one.py": b"def test_one(): pass\n",
        "examples/demo.py": b"print('demo')\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    excluded = {
        "evidence/private.py": b"secret episode snapshot",
        "runs/private.py": b"secret run",
        "checkpoints/weights.py": b"model weights",
        "dist/old.py": b"old distribution",
        "build/generated.py": b"generated build output",
        ".venv/lib/private.py": b"environment file",
        "kernel_env/__pycache__/cached.py": b"cache file",
        "kernel_env/episodes/snapshot.py": b"episode source snapshot",
        "tests/fixtures/secret.py": b"nested test fixture",
        "examples/generated/snapshot.py": b"nested example output",
        "suite/snapshots/generated.py": b"nested suite output",
        "outside.txt": b"not an allowed extension",
    }
    for relative, content in excluded.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return files


def test_source_release_contains_only_allowlisted_files_and_hashes(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_files = make_source_tree(source_root)
    archive_path = tmp_path / "release.zip"

    builder.build_source_release(archive_path, source_root)

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        expected = {f"kernel-env/{name}" for name in source_files}
        expected.add("kernel-env/SOURCE-MANIFEST.json")
        assert names == expected
        for relative, original in source_files.items():
            assert archive.read(f"kernel-env/{relative}") == original

        manifest = json.loads(archive.read("kernel-env/SOURCE-MANIFEST.json"))
        assert manifest["version"] == "4.5.6"
        assert manifest["files"] == [
            {
                "path": f"kernel-env/{relative}",
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for relative, content in sorted(source_files.items())
        ]


def test_source_release_is_reproducible_and_refuses_overwrite(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    make_source_tree(source_root)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    builder.build_source_release(first, source_root)
    builder.build_source_release(second, source_root)
    first_bytes = first.read_bytes()
    assert second.read_bytes() == first_bytes

    try:
        builder.build_source_release(first, source_root)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing source archive was overwritten")
    assert first.read_bytes() == first_bytes
