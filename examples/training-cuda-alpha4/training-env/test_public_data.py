"""Focused remote-only stdlib fixtures for the new raw acquisition boundary.

No model/NumPy/tokenizers imports and no real network requests.
"""
import copy
import hashlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_public_data as prep


class Response(io.BytesIO):
    def __init__(self, body=b"abc", status=206, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers if headers is not None else {
            "Content-Range": "bytes 0-2/100", "Content-Length": "3"}
        self.reads = 0

    def geturl(self):
        return "https://cdn.example.invalid/asset?temporary-secret=not-for-logs"

    def read(self, n=-1):
        self.reads += 1
        return super().read(n)


def tiny_preparation_boundary(root):
    """Remote stdlib-only raw-text fixture; no tokenizer/arrays/network."""
    profile = copy.deepcopy(json.loads(prep.PROFILE.read_text()))
    upstream = profile["upstream"]
    bodies = {"train": b"abc", "validation": b"defg"}
    for split, body in bodies.items():
        upstream[split].update(full_bytes=100 if split == "train" else len(body),
                              selected_bytes=len(body), full_sha256=hashlib.sha256(body).hexdigest())
    cache = Path(root) / prep.identity(upstream)
    cache.mkdir()
    files = {"publisher-README.md": b"---\nlicense: cdla-sharing-1.0\n---\n"}
    native = {"source": {}}
    for split, body in bodies.items():
        spec = upstream[split]
        files[split + ".lfs"] = ("version https://git-lfs.github.com/spec/v1\n"
            + "oid sha256:" + spec["full_sha256"] + "\nsize " + str(spec["full_bytes"]) + "\n").encode()
        files[spec["cache_filename"]] = body
        native_key = "train_text" if split == "train" else "validation_text"
        native["source"][native_key] = {"path": str((cache / spec["cache_filename"]).resolve()),
                                       "sha256": hashlib.sha256(body).hexdigest()}
    for name, body in files.items():
        (cache / name).write_bytes(body)
    raw = {"upstream": upstream, "train_full_blob_contents_verified": False,
           "files": {name: {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
                     for name, body in files.items()}}
    lock = {"schema": "training-raw-lock-v1", "identity": raw, "identity_sha256": prep.identity(raw)}
    prep.save(cache / "raw-lock.json", lock)
    return profile, cache, lock, native


class AcquisitionBoundary(unittest.TestCase):
    def fetch(self, response, destination, **kwargs):
        class Opener:
            def open(self, *args, **kw):
                return response
        with patch.object(prep.urllib.request, "build_opener", return_value=Opener()):
            return prep.fetch("https://example.invalid/source", destination,
                maximum=3, exact=3, deadline=time.monotonic() + 5, **kwargs)

    def test_range_ignored_rejected_before_body_or_output(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "raw"
            response = Response(status=200)
            with self.assertRaises(ValueError):
                self.fetch(response, destination, content_range="bytes 0-2/100")
            self.assertEqual(response.reads, 0)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_headers_and_transport_encoding_fail_before_body(self):
        cases = [
            {"Content-Range": "bytes 1-3/100", "Content-Length": "3"},
            {"Content-Range": "bytes 0-2/100", "Content-Length": "100"},
            {"Content-Range": "bytes 0-2/100", "Content-Encoding": "gzip"},
        ]
        for headers in cases:
            with self.subTest(headers=headers), tempfile.TemporaryDirectory() as root:
                response = Response(headers=headers)
                with self.assertRaises(ValueError):
                    self.fetch(response, Path(root) / "raw", content_range="bytes 0-2/100")
                self.assertEqual(response.reads, 0)

    def test_short_body_and_bad_hash_preserve_partial_without_ready_file(self):
        for body, expected in [(b"ab", None), (b"abc", "0" * 64)]:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as root:
                destination = Path(root) / "raw"
                with self.assertRaises(ValueError):
                    self.fetch(Response(body=body), destination, expected_sha=expected,
                               content_range="bytes 0-2/100")
                self.assertFalse(destination.exists())
                self.assertTrue((Path(root) / "raw.part").exists())

    def test_success_records_hash_without_signed_url(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "raw"
            result = self.fetch(Response(), destination, content_range="bytes 0-2/100")
            self.assertEqual(destination.read_bytes(), b"abc")
            self.assertEqual(result["sha256"], hashlib.sha256(b"abc").hexdigest())
            self.assertNotIn("temporary-secret", json.dumps(result))
            self.assertEqual(result["final_host"], "cdn.example.invalid")

    def test_cache_reuse_and_tampering_and_offline_miss(self):
        profile = copy.deepcopy(json.loads(prep.PROFILE.read_text()))
        upstream = profile["upstream"]
        for split, body in (("train", b"abc"), ("validation", b"defg")):
            upstream[split].update(full_bytes=len(body), selected_bytes=len(body),
                                   full_sha256=hashlib.sha256(body).hexdigest())
        with tempfile.TemporaryDirectory() as root:
            cache_root = Path(root)
            cache = cache_root / prep.identity(upstream)
            with self.assertRaises(ValueError):
                prep.acquire(profile, cache_root, offline=True)
            cache.mkdir()
            with self.assertRaises(ValueError):
                prep.acquire(profile, cache_root, offline=False)
            files = {"publisher-README.md": b"---\nlicense: cdla-sharing-1.0\n---\n"}
            for split, body in (("train", b"abc"), ("validation", b"defg")):
                spec = upstream[split]
                files[split + ".lfs"] = ("version https://git-lfs.github.com/spec/v1\n"
                    + "oid sha256:" + spec["full_sha256"] + "\nsize " + str(len(body)) + "\n").encode()
                files[spec["cache_filename"]] = body
            for name, body in files.items():
                (cache / name).write_bytes(body)
            raw = {"upstream": upstream, "train_full_blob_contents_verified": False,
                   "files": {name: {"bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest()} for name, body in files.items()}}
            prep.save(cache / "raw-lock.json", {"schema": "training-raw-lock-v1",
                "identity": raw, "identity_sha256": prep.identity(raw)})
            with patch.object(prep, "fetch", side_effect=AssertionError("cache must not download")):
                actual, _, reused = prep.acquire(profile, cache_root, offline=True)
                self.assertEqual(actual, cache)
                self.assertTrue(reused)
                (cache / upstream["train"]["cache_filename"]).write_bytes(b"xyz")
                with self.assertRaises(ValueError):
                    prep.acquire(profile, cache_root, offline=True)

    def test_plain_http_redirect_is_rejected(self):
        with self.assertRaises(ValueError):
            prep.HTTPSRedirects().redirect_request(None, None, 302, "", {}, "http://example.invalid/raw")

    def test_input_binding_rechecks_cache_after_preparation(self):
        with tempfile.TemporaryDirectory() as root:
            profile, cache, _, manifest = tiny_preparation_boundary(root)
            # acquire returns a verified lock; native preparation is simulated
            # as returning matching original provenance while raw bytes change.
            _, original, reused = prep.acquire(profile, Path(root), offline=True)
            self.assertTrue(reused)
            (cache / profile["upstream"]["train"]["cache_filename"]).write_bytes(b"xyz")
            with self.assertRaisesRegex(ValueError, "raw cache contents changed"):
                prep.verify_prepared_inputs(cache, profile, original, manifest)

    def test_input_binding_rejects_preparer_hash_disagreement(self):
        for field in ("train_text", "validation_text"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                profile, cache, original, manifest = tiny_preparation_boundary(root)
                manifest["source"][field]["sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "preparer input provenance differs"):
                    prep.verify_prepared_inputs(cache, profile, original, manifest)

    def test_input_binding_rejects_replacement_cache_lock(self):
        with tempfile.TemporaryDirectory() as root:
            profile, cache, original, manifest = tiny_preparation_boundary(root)
            replacement = copy.deepcopy(original)
            filename = profile["upstream"]["train"]["cache_filename"]
            (cache / filename).write_bytes(b"xyz")
            replacement["identity"]["files"][filename]["sha256"] = hashlib.sha256(b"xyz").hexdigest()
            replacement["identity_sha256"] = prep.identity(replacement["identity"])
            (cache / "raw-lock.json").write_text(json.dumps(replacement))
            # Replacement is self-consistent but cannot replace the original
            # identity captured before preparation.
            self.assertEqual(prep.verify_cache(cache, profile["upstream"])["identity_sha256"],
                             replacement["identity_sha256"])
            with self.assertRaisesRegex(ValueError, "raw lock identity changed"):
                prep.verify_prepared_inputs(cache, profile, original, manifest)

    def test_input_binding_accepts_matching_native_hashes_and_stable_cache(self):
        with tempfile.TemporaryDirectory() as root:
            profile, cache, original, manifest = tiny_preparation_boundary(root)
            proof = prep.verify_prepared_inputs(cache, profile, original, manifest)
            self.assertEqual(proof["raw_identity_sha256"], original["identity_sha256"])
            self.assertTrue(proof["cache_reverified_after_preparation"])
            self.assertEqual(proof["preparer_input_sha256"], {
                key: manifest["source"][key]["sha256"] for key in ("train_text", "validation_text")})


if __name__ == "__main__":
    unittest.main()
