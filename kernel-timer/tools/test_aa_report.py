"""Test the pair3 receipt report.
  python -m unittest tools/test_aa_report.py
"""
import json, subprocess, sys, tempfile, unittest
from pathlib import Path
from tools import aa_report


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        pairs = []
        for i, (incumbent, candidate, clock) in enumerate(zip(
                (10, 20, 30, 40), (9, 22, 39, 60), (800, 900, 1000, 1000))):
            pairs.append({"order": ["incumbent", "candidate"] if i % 2 == 0 else ["candidate", "incumbent"],
                          "incumbent_ms": incumbent, "candidate_ms": candidate, "anchor_ms": 2 * incumbent,
                          "clock_sm_mhz": clock, "clock_read": "nvml_under_load" if i else "fallback"})
        self.ok = self.root / "ok.json"
        self.invalid = self.root / "invalid.json"
        self.ok.write_text(json.dumps({"schema_version": "kernel-koth-paired-v1", "status": "ok",
                                       "interaction": {"pairs": pairs}, "validation": {"pairs": pairs}}))
        self.invalid.write_text(json.dumps({"schema_version": "kernel-koth-paired-v1", "status": "invalid",
                                            "failure": {"blame": "candidate", "stage": "correctness",
                                                        "reason": "wrong answer"}}))

    def test_statistics(self):
        report = aa_report.summarize(self.ok)
        for name in ("interaction", "validation"):
            phase = report[name]
            self.assertEqual(phase["n"], 4)
            self.assertAlmostEqual(phase["candidate_incumbent_median"], 1.2)
            self.assertAlmostEqual(phase["order_medians"]["incumbent-first"], 1.1)
            self.assertAlmostEqual(phase["order_medians"]["candidate-first"], 1.3)
            self.assertEqual(phase["anchor_incumbent_median"], 2)
            self.assertAlmostEqual(phase["anchor_candidate_median"], (20 / 11 + 20 / 13) / 2)
            self.assertEqual(phase["clock_sm_mhz"], {"min": 800, "median": 950, "max": 1000})
            self.assertEqual(phase["clock_read"], ["fallback", "nvml_under_load"])
            self.assertEqual(phase["low_clock_count"], 1)
            self.assertEqual(phase["incumbent_mean_ms"], {"low": 10, "kept": 30})
        self.assertTrue(report["copy_risk"])
        self.assertAlmostEqual(report["gate_distance"], 0.18)
        self.assertIn("COPY-RISK; 18.00% from the gate", aa_report.format_report(report))
        self.assertFalse(aa_report.summarize(self.ok, gate=0.3)["copy_risk"])

    def test_invalid(self):
        report = aa_report.summarize(self.invalid)
        self.assertEqual(aa_report.format_report(report),
                         f"{self.invalid}: status=invalid blame=candidate stage=correctness reason=wrong answer")

    def test_cli(self):
        command = [sys.executable, str(Path(aa_report.__file__))]
        result = subprocess.run(command + [str(self.root / "*.json"), "--json"],
                                check=True, capture_output=True, text=True)
        reports = json.loads(result.stdout)
        self.assertEqual([r["status"] for r in reports], ["invalid", "ok"])
        self.assertEqual(reports[1]["validation"]["low_clock_count"], 1)
        result = subprocess.run(command + [str(self.invalid), str(self.ok), "--gate", "0.3"],
                                check=True, capture_output=True, text=True)
        self.assertIn("status=invalid blame=candidate stage=correctness reason=wrong answer", result.stdout)
        self.assertIn("10.00% from the gate", result.stdout)
        self.assertNotIn("COPY-RISK", result.stdout)


if __name__ == "__main__":
    unittest.main()
