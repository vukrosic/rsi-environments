"""Remote-only stdlib fixtures for public preparation/native-result boundaries."""
import copy
import math
import unittest
from unittest.mock import patch

import public_cuda_example as example
import verify_public_cuda_episode as public_audit


class PublicResultBoundaries(unittest.TestCase):
    def test_optional_operator_uuid_pin_is_enforced(self):
        spec = example.read(example.SPEC)
        versions = spec["runtime"]
        row = "GPU-pinned, 00000000:02:00.0, NVIDIA RTX 4090, 595.91.07, 24564"
        with patch.object(example.platform,"system",return_value="Linux"), \
             patch.object(example.platform,"python_version",return_value=versions["python"]), \
             patch.object(example.importlib.metadata,"version",side_effect=lambda key:versions[key]), \
             patch.object(example,"query",side_effect=[row, "", row]):
            value = example.physical({"RSI_PUBLIC_CUDA_EXPECTED_UUID":"GPU-pinned"},spec)
            self.assertEqual(value["gpu_identity"]["uuid"],"GPU-pinned")
            with self.assertRaises(ValueError):
                example.physical({"RSI_PUBLIC_CUDA_EXPECTED_UUID":"GPU-wrong"},spec)

    def test_auditor_binds_actual_device_name_without_brand_gate(self):
        inv = {"available":True,"device_count":1,"devices":[{"name":"NVIDIA RTX 4090"}]}
        public_audit.verify_admitted_device(inv,"NVIDIA RTX 4090")
        with self.assertRaises(ValueError):
            public_audit.verify_admitted_device(inv,"NVIDIA RTX A2000")
        inv["device_count"] = 2
        with self.assertRaises(ValueError):
            public_audit.verify_admitted_device(inv,"NVIDIA RTX 4090")

    def test_single_other_nvidia_device_can_attempt_same_protocol(self):
        actual = example.single_device("GPU-other, 00000000:02:00.0, NVIDIA RTX 4090, 595.91.07, 24564")
        self.assertEqual(actual["name"],"NVIDIA RTX 4090")
        self.assertEqual(actual["memory_mib"],24564)
        self.assertEqual(actual["uuid"],"GPU-other")
        for invalid in ("", "GPU-a, pci, name, driver, 100\nGPU-b, pci, name, driver, 100",
                        "GPU-a, pci, name, driver, 0", "GPU-a, pci, name, driver, N/A"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                example.single_device(invalid)

    def test_v1_or_unverified_input_proof_does_not_admit_training(self):
        valid = {"status":"ok", "supervisor":{"status":"ok","cleanup_errors":[],"discovery_errors":[]},
                 "child":{"status":"prepared","accepted92_equivalent":False,"raw_identity_sha256":"a" * 64,
                    "raw_input_provenance":{"version":"raw-input-boundary-v2","cache_reverified_after_preparation":True,
                    "raw_identity_sha256":"a" * 64,"preparer_input_sha256":{"train_text":"b" * 64,"validation_text":"c" * 64}}}}
        self.assertEqual(example.prepared_gate(valid,"raw-input-boundary-v2"),valid["child"])
        for mutation in ("v1", "changed-identity", "failed-cleanup"):
            value = copy.deepcopy(valid)
            if mutation == "v1":
                value["child"]["raw_input_provenance"]["version"] = "v1"
            elif mutation == "changed-identity":
                value["child"]["raw_input_provenance"]["raw_identity_sha256"] = "0" * 64
            else:
                value["supervisor"]["cleanup_errors"] = ["simulated"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                example.prepared_gate(value,"raw-input-boundary-v2")

    def test_non_ok_native_record_cannot_pass_even_with_finite_reward(self):
        spec = example.read(example.SPEC)
        record = self.record(spec)
        self.assertEqual(example.native_gate(record,spec,"d" * 64)["native_reward"],math.exp(-3))
        for status in ("blocked","error","invalid"):
            value = copy.deepcopy(record)
            value["status"] = status
            with self.subTest(status=status), self.assertRaises(ValueError):
                example.native_gate(value,spec,"d" * 64)

    def test_fabricated_reward_or_wrong_corpus_is_rejected(self):
        spec = example.read(example.SPEC)
        for mutation in ("reward", "data", "source", "calls"):
            record = self.record(spec)
            if mutation == "reward":
                record["reward"] = 0.9
            elif mutation == "data":
                record["provenance"]["data_manifest_sha256"] = "e" * 64
            elif mutation == "source":
                record["provenance"]["source_sha256"] = "f" * 64
            else:
                record["metrics"]["steps"] = 499
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                example.native_gate(record,spec,"d" * 64)

    @staticmethod
    def record(spec):
        return {"status":"ok","profile":"standard","reward":math.exp(-3),
                "metrics":{"held_out_loss":3.0,"protocol":spec["budget"],"split":"dev",
                           "steps":500,"training_tokens":4096000,"eval_tokens":491520},
                "provenance":{"source_sha256":spec["candidate_sha256"],"data_manifest_sha256":"d" * 64,
                              "device":"cuda","data_kind":"real"}}


if __name__ == "__main__":
    unittest.main()
