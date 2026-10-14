from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unittest
from pathlib import Path
from zipfile import ZipFile

import package_build


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ("TASK.md", "DESIGN.md", "RL-CONTRACT.md")


class PackagingTests(unittest.TestCase):
    def test_wheel_contains_versioned_complete_contract_documents(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]
        version = project["version"]
        self.assertEqual(package_build.VERSION, version)

        with tempfile.TemporaryDirectory(prefix="rsi-contract-wheel-") as temporary:
            temp = Path(temporary)
            wheel_dir = temp / "wheel"
            wheel_dir.mkdir()
            name = package_build.build_wheel(wheel_dir)
            self.assertEqual(name, f"vuk_rsi_inference-{version}-py3-none-any.whl")

            unpacked = temp / "unpacked"
            unpacked.mkdir()
            with ZipFile(wheel_dir / name) as wheel:
                names = set(wheel.namelist())
                self.assertIn(f"vuk_rsi_inference-{version}.dist-info/METADATA", names)
                metadata = wheel.read(f"vuk_rsi_inference-{version}.dist-info/METADATA").decode()
                self.assertIn(f"Version: {version}", metadata)
                for filename in CONTRACTS:
                    self.assertIn(f"inference_env/_runtime/{filename}", names)
                self.assertFalse(any(name.endswith((".pt", ".safetensors", ".jsonl", ".log")) for name in names))
                self.assertFalse(any(part in {"episodes", "evidence", ".cache", ".venv"}
                                     for name in names for part in name.split("/")))
                wheel.extractall(unpacked)

            probe = r'''
import hashlib, json, pathlib
from inference_env import CodingAgentEnvironment
import inference_env
import inference_env._runtime.worker as packaged_worker

required = {"TASK.md", "DESIGN.md", "RL-CONTRACT.md"}
episode_root = pathlib.Path("episode-root").resolve()
observation = CodingAgentEnvironment(episode_root, device="cpu").reset(split="train", seed=17)
documents = observation["contract_documents"]
assert set(documents) == required, set(documents)
runtime = pathlib.Path(inference_env.__file__).resolve().parent / "_runtime"
resolved = {}
for filename, item in documents.items():
    assert set(item) == {"content", "sha256"}
    content = item["content"]
    bundled = (runtime / filename).read_text(encoding="utf-8")
    assert content == bundled
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert item["sha256"] == digest
    resolved[filename] = {"bytes": len(content.encode("utf-8")), "sha256": digest}

task = documents["TASK.md"]["content"].lower()
forbidden_rules = ("reading scorer inputs", "baseline outputs", "timing artifacts",
                   "modifying clocks", "untimed setup violates")
assert all(rule in task for rule in forbidden_rules), forbidden_rules
packaged_worker._dependencies()
import inference_env._runtime.assets as packaged_assets
import inference_env._runtime.cache_probes as packaged_probes
assert pathlib.Path(packaged_assets.__file__).resolve().parent == runtime
assert pathlib.Path(packaged_probes.__file__).resolve().parent == runtime
print(json.dumps({"contract": observation["evaluation_contract"], "documents": resolved,
                  "version": inference_env.__version__,
                  "prohibitions_in_task_text": list(forbidden_rules),
                  "installed_worker_bootstrap": {"worker": packaged_worker.__file__,
                    "assets": packaged_assets.__file__, "cache_probes": packaged_probes.__file__}}))
'''
            env = os.environ.copy()
            env["PYTHONPATH"] = str(unpacked)
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            completed = subprocess.run(
                [sys.executable, "-c", probe], cwd=temp, env=env, text=True,
                capture_output=True, check=False, timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["version"], version)
            self.assertEqual(result["contract"], "rsi.inference.smollm2-135m.v2")
            self.assertEqual(set(result["documents"]), set(CONTRACTS))

            standalone_probe = f'''
import json, pathlib, sys
sys.path.insert(0, {str(unpacked / 'inference_env' / '_runtime')!r})
import worker
assert worker.__package__ in (None, "")
worker._dependencies()
import assets, cache_probes
runtime = pathlib.Path(worker.__file__).resolve().parent
assert pathlib.Path(assets.__file__).resolve().parent == runtime
assert pathlib.Path(cache_probes.__file__).resolve().parent == runtime
print(json.dumps({{"worker": worker.__file__, "assets": assets.__file__,
                  "cache_probes": cache_probes.__file__}}))
'''
            standalone = subprocess.run(
                [sys.executable, "-c", standalone_probe], cwd=temp, env=env,
                text=True, capture_output=True, check=False, timeout=30,
            )
            self.assertEqual(standalone.returncode, 0, standalone.stderr)
            bootstrap = json.loads(standalone.stdout)
            self.assertEqual(Path(bootstrap["worker"]).resolve().parent,
                             (unpacked / "inference_env" / "_runtime").resolve())

    def test_sdist_contains_all_contract_documents(self):
        with tempfile.TemporaryDirectory(prefix="rsi-contract-sdist-") as temporary:
            name = package_build.build_sdist(temporary)
            archive = Path(temporary) / name
            with tarfile.open(archive, "r:gz") as source:
                names = set(source.getnames())
                prefix = f"vuk-rsi-inference-{package_build.VERSION}/"
                self.assertIn(prefix + "README.md", names)
                bundled_readme = source.extractfile(prefix + "README.md").read()
            self.assertEqual(bundled_readme, (ROOT / "README.md").read_bytes())
            prefix = f"vuk-rsi-inference-{package_build.VERSION}/"
            for filename in CONTRACTS:
                self.assertIn(prefix + filename, names)


if __name__ == "__main__":
    unittest.main()
