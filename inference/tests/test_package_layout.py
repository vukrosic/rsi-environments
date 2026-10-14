from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import package_build


class PackageLayoutTests(unittest.TestCase):
    def test_wheel_is_versioned_namespaced_and_weight_free(self):
        with tempfile.TemporaryDirectory(prefix="rsi-package-test-") as temporary:
            wheel_dir = Path(temporary)
            name = package_build.build_wheel(wheel_dir)
            self.assertEqual(name, "vuk_rsi_inference-0.2.0-py3-none-any.whl")
            with ZipFile(wheel_dir / name) as wheel:
                names = set(wheel.namelist())
                for module in (item for item in names if item.startswith("inference_env/") and item.endswith(".py")):
                    compile(wheel.read(module), module, "exec")
            self.assertIn("inference_env/__init__.py", names)
            self.assertIn("inference_env/_runtime/scorer.py", names)
            self.assertIn("inference_env/_runtime/controller.py", names)
            self.assertIn("inference_env/_runtime/supervisor.py", names)
            self.assertIn("inference_env/_runtime/cache_probes.py", names)
            self.assertIn("inference_env/_runtime/shared/records.py", names)
            self.assertIn("vuk_rsi_inference-0.2.0.dist-info/METADATA", names)
            self.assertFalse(any(name.startswith("scorer.py") for name in names))
            forbidden = ("episodes/", "evidence/", ".venv/", "assets/", ".cache/")
            self.assertFalse(any(any(part in name for part in forbidden) for name in names))
            self.assertFalse(any(name.endswith((".pt", ".safetensors", ".log", ".jsonl")) for name in names))
            self.assertLess(sum(len(name) for name in names), 100_000)

    def test_package_imports_do_not_export_generic_module_names(self):
        rewritten = package_build._rewrite_imports(
            "from assets import HERE\nfrom cache_probes import run\n"
            "from scorer import evaluate\nfrom supervisor import run\n"
        )
        self.assertIn("if __package__:", rewritten)
        self.assertIn("from .assets import HERE", rewritten)
        self.assertIn("from .cache_probes import run", rewritten)
        self.assertIn("from .scorer import evaluate", rewritten)
        self.assertIn("from .supervisor import run", rewritten)
        self.assertIn("else:", rewritten)

    def test_wheel_bytes_are_reproducible(self):
        with tempfile.TemporaryDirectory(prefix="rsi-package-repro-") as temporary:
            first_dir = Path(temporary) / "first"
            second_dir = Path(temporary) / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            name = package_build.build_wheel(first_dir)
            self.assertEqual(name, package_build.build_wheel(second_dir))
            self.assertEqual((first_dir / name).read_bytes(), (second_dir / name).read_bytes())
if __name__ == "__main__":
    unittest.main()
