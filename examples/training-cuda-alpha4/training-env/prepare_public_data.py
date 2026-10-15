"""Bounded official-text acquisition; reuse prepare.py for a NEW data identity.

The public entrypoint uses only the standard library. NumPy/tokenizers load only
inside its supervised CPU child after source, runtime and raw-file checks.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE = HERE / "raw-data-profile.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def phase(output, name):
    with (output / "phases.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"phase": name, "utc": datetime.now(timezone.utc).isoformat()}) + "\n")


def checked_profile():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    if profile["schema"] != "training-public-raw-profile-v1" or profile["accepted92_equivalent"] is not False:
        raise ValueError("unsupported public data profile")
    if set(profile["source_sha256"]) != {"prepare.py", "data.py", "process_control.py"}:
        raise ValueError("public data dependency inventory mismatch")
    for name, expected in profile["source_sha256"].items():
        if name not in {"prepare.py", "data.py", "process_control.py"} or digest(HERE / name) != expected:
            raise ValueError("public data dependency source mismatch")
    return profile


class HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ValueError("non-HTTPS dataset redirect rejected")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def response_check(response, *, exact, maximum, content_range=None):
    # Check BEFORE reading the body: a Range-ignored 2.2GB response is rejected.
    if urllib.parse.urlsplit(response.geturl()).scheme != "https":
        raise ValueError("non-HTTPS dataset response rejected")
    if response.status != (206 if content_range else 200):
        raise ValueError("unexpected dataset HTTP status")
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ValueError("encoded dataset response rejected")
    if response.headers.get("Content-Range") != content_range:
        raise ValueError("dataset Content-Range mismatch")
    length = response.headers.get("Content-Length")
    if length is not None:
        length = int(length)
        if length < 0 or length > maximum or (exact is not None and length != exact):
            raise ValueError("dataset Content-Length mismatch")


def fetch(url, destination, *, maximum, deadline, exact=None, expected_sha=None,
          content_range=None):
    headers = {"Accept-Encoding": "identity", "User-Agent": "training-public-data-v1"}
    if content_range:
        headers["Range"] = "bytes=0-" + str(exact - 1)
    request = urllib.request.Request(url, headers=headers)
    partial = destination.with_name(destination.name + ".part")
    if destination.exists() or partial.exists():
        raise ValueError("raw destination exists; preserve it and choose another cache")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("raw acquisition deadline reached")
    try:
        opener = urllib.request.build_opener(HTTPSRedirects())
        with opener.open(request, timeout=min(10, remaining)) as response:
            response_check(response, exact=exact, maximum=maximum, content_range=content_range)
            size = 0
            h = hashlib.sha256()
            with partial.open("xb") as stream:
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("raw acquisition deadline reached")
                    block = response.read(min(1024 * 1024, maximum - size + 1))
                    if not block:
                        break
                    size += len(block)
                    if size > maximum:
                        raise ValueError("raw response exceeded byte limit")
                    stream.write(block)
                    h.update(block)
            if not size or (exact is not None and size != exact):
                raise ValueError("raw response byte count mismatch")
            if expected_sha is not None and h.hexdigest() != expected_sha:
                raise ValueError("raw response SHA-256 mismatch")
            receipt = {"bytes": size, "sha256": h.hexdigest(), "status": response.status,
                       "final_host": urllib.parse.urlsplit(response.geturl()).hostname,
                       "content_range": response.headers.get("Content-Range")}
    except urllib.error.HTTPError as exc:
        # Do not persist signed redirect URLs or exception strings containing them.
        raise ValueError(f"official dataset HTTP status {exc.code}") from None
    except urllib.error.URLError:
        raise ValueError("official dataset transport failed; no retry performed") from None
    partial.rename(destination)
    return receipt


def raw_url(upstream, route, filename):
    return ("https://huggingface.co/datasets/" + upstream["repository"] + "/" + route
            + "/" + upstream["revision"] + "/" + filename)


def verify_cache(cache, upstream):
    lock = json.loads((cache / "raw-lock.json").read_text(encoding="utf-8"))
    raw = lock["identity"]
    if (lock.get("schema") != "training-raw-lock-v1" or raw["upstream"] != upstream
            or raw.get("train_full_blob_contents_verified") is not False
            or lock["identity_sha256"] != identity(raw)):
        raise ValueError("raw cache definition/identity mismatch")
    expected_names = {"publisher-README.md", "train.lfs", "validation.lfs",
                      upstream["train"]["cache_filename"], upstream["validation"]["cache_filename"]}
    if set(raw["files"]) != expected_names:
        raise ValueError("raw cache file inventory mismatch")
    for name, record in raw["files"].items():
        path = cache / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("raw cache contents changed")
    check_raw_metadata(cache, upstream)
    return lock


def check_raw_metadata(cache, upstream):
    card = (cache / "publisher-README.md").read_text(encoding="utf-8")
    if not re.search(r"(?m)^license:\s*cdla-sharing-1\.0\s*$", card):
        raise ValueError("publisher license metadata differs from pinned profile")
    for split in ("train", "validation"):
        spec = upstream[split]
        pointer = (cache / (split + ".lfs")).read_text(encoding="ascii")
        lines = pointer.splitlines()
        expected = ["version https://git-lfs.github.com/spec/v1",
                    "oid sha256:" + spec["full_sha256"], "size " + str(spec["full_bytes"])]
        if lines != expected:
            raise ValueError("pinned revision LFS pointer differs from profile")
        path = cache / spec["cache_filename"]
        if path.stat().st_size != spec["selected_bytes"]:
            raise ValueError("selected raw bytes mismatch")
        if split == "validation" and digest(path) != spec["full_sha256"]:
            raise ValueError("full validation contents mismatch")


def verify_prepared_inputs(cache, profile, lock, manifest):
    """Bind native input provenance and the post-use cache to the original lock.

    Before/after checks are cooperative filesystem integrity controls. They do
    not prove that a hostile writer never changed and restored bytes mid-read.
    """
    upstream = profile["upstream"]
    original = lock["identity"]
    if lock["identity_sha256"] != identity(original) or original["upstream"] != upstream:
        raise ValueError("original raw identity differs from preparation profile")
    consumed = {}
    for split, native_key in (("train", "train_text"), ("validation", "validation_text")):
        filename = upstream[split]["cache_filename"]
        expected = original["files"][filename]["sha256"]
        native = manifest.get("source", {}).get(native_key, {})
        if native.get("sha256") != expected or native.get("path") != str((cache / filename).resolve()):
            raise ValueError("preparer input provenance differs from original raw lock")
        consumed[native_key] = expected
    after = verify_cache(cache, upstream)  # text, card, both pointers and lock
    if after["identity_sha256"] != lock["identity_sha256"] or after["identity"] != original:
        raise ValueError("raw lock identity changed across preparation")
    return {"version": "raw-input-boundary-v2", "raw_identity_sha256": lock["identity_sha256"],
            "preparer_input_sha256": consumed, "cache_reverified_after_preparation": True}


def acquire(profile, cache_root, offline):
    upstream = profile["upstream"]
    cache = cache_root / identity(upstream)
    if cache.exists():
        if not (cache / "raw-lock.json").is_file():
            raise ValueError("incomplete raw cache preserved; use a new cache root")
        return cache, verify_cache(cache, upstream), True
    if offline:
        raise ValueError("offline mode requires a complete verified raw cache")
    cache.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + profile["limits"]["network_seconds"]
    receipts = {}
    receipts["publisher-README.md"] = fetch(raw_url(upstream, "resolve", "README.md"),
        cache / "publisher-README.md", maximum=32768, deadline=deadline)
    if not re.search(r"(?m)^license:\s*cdla-sharing-1\.0\s*$",
                     (cache / "publisher-README.md").read_text(encoding="utf-8")):
        raise ValueError("publisher license metadata differs from pinned profile")
    for split in ("train", "validation"):
        spec = upstream[split]
        receipts[split + ".lfs"] = fetch(raw_url(upstream, "raw", spec["upstream_filename"]),
            cache / (split + ".lfs"), maximum=1024, deadline=deadline)
        # Check the pointer before acquiring its selected contents.
        lines = (cache / (split + ".lfs")).read_text(encoding="ascii").splitlines()
        if lines != ["version https://git-lfs.github.com/spec/v1",
                     "oid sha256:" + spec["full_sha256"], "size " + str(spec["full_bytes"])]:
            raise ValueError("pinned revision LFS pointer differs from profile")
        selected = spec["selected_bytes"]
        receipts[spec["cache_filename"]] = fetch(raw_url(upstream, "resolve", spec["upstream_filename"]),
            cache / spec["cache_filename"], maximum=selected, exact=selected, deadline=deadline,
            expected_sha=spec["full_sha256"] if split == "validation" else None,
            content_range=f"bytes 0-{selected - 1}/{spec['full_bytes']}" if split == "train" else None)
    check_raw_metadata(cache, upstream)
    raw = {"upstream": upstream,
           "files": {name: {"bytes": value["bytes"], "sha256": value["sha256"]}
                     for name, value in receipts.items()},
           "train_full_blob_contents_verified": False,
           "train_prefix_hash": "measured on first acquisition; not a publisher full-blob integrity check"}
    lock = {"schema": "training-raw-lock-v1", "identity": raw, "identity_sha256": identity(raw),
            "acquired_at": datetime.now(timezone.utc).isoformat(), "http": receipts}
    save(cache / "raw-lock.json", lock)
    return cache, lock, False


def child(args, profile):
    profile_sha = digest(PROFILE)
    phase(args.output, "runtime-metadata")
    runtime = {"python": platform.python_version()}
    if not args.acquire_only:
        runtime.update({name: importlib.metadata.version(name) for name in ("numpy", "tokenizers")})
        if runtime != profile["runtime"]:
            raise ValueError("preparation runtime differs from pinned observed lane")
    if shutil.disk_usage(args.output).free < 1024 ** 3:
        raise ValueError("preparation requires at least 1 GiB free disk before acquisition")
    phase(args.output, "raw-acquisition-or-cache-verification")
    cache, lock, reused = acquire(profile, args.cache, args.offline)
    result = {"schema": "training-public-data-result-v1", "status": "raw-ready",
              "accepted92_equivalent": False, "gpu_used": False, "model_updates": 0,
              "agent_trainable": False, "effective_reward": None,
              "raw_identity_sha256": lock["identity_sha256"], "cache_reused": reused,
              "profile_sha256": profile_sha, "runtime": runtime,
              "runtime_origins": {"python_executable": sys.executable,
                                  "python_prefix": sys.prefix, "python_base_prefix": sys.base_prefix},
              "byte_reproduction_verified": False}
    workload = None
    if not args.acquire_only:
        phase(args.output, "existing-text-preparer")
        # Reuse the existing tokenizer/split pipeline, unchanged. No torch import.
        from prepare import prepare_from_text
        from data import load_data
        import numpy
        import tokenizers
        if (numpy.__version__ != runtime["numpy"] or tokenizers.__version__ != runtime["tokenizers"]):
            raise ValueError("imported preparation dependency differs from distribution metadata")
        result["runtime_origins"].update(numpy_file=str(Path(numpy.__file__).resolve()),
                                         tokenizers_file=str(Path(tokenizers.__file__).resolve()))
        data_root = args.output / "data"
        manifest = prepare_from_text(cache / profile["upstream"]["train"]["cache_filename"],
                                     cache / profile["upstream"]["validation"]["cache_filename"], data_root)
        phase(args.output, "raw-provenance-after-preparation")
        provenance = verify_prepared_inputs(cache, profile, lock, manifest)
        phase(args.output, "native-standard-semantic-validation")
        _, _, _, metadata = load_data(data_root, profile="standard", use_cache=False)
        portable = {"family": profile["workload_family"], "profile_sha256": profile_sha,
                    "raw_identity_sha256": lock["identity_sha256"], "runtime": runtime,
                    "source_sha256": profile["source_sha256"], "asset_sha256": manifest["sha256"],
                    "raw_input_provenance": provenance,
                    "counts": manifest["counts"], "story_counts": manifest["story_counts"],
                    "tokenizer": profile["tokenizer"], "accepted92_equivalent": False}
        workload_id = profile["workload_family"] + ":" + identity(portable)
        result.update(status="prepared", workload_id=workload_id,
                      native_manifest_sha256=digest(data_root / "manifest.json"),
                      semantic_validation=metadata["validation"],
                      paired_dev_window_capacity=manifest["counts"]["dev"] >= 986880)
        workload = {"schema": "training-new-workload-v1",
             "workload_id": workload_id, "identity": portable,
             "native_manifest_sha256": result["native_manifest_sha256"],
             "native_manifest_paths": "private machine paths retained; excluded from portable identity"}
        (data_root / "publisher-README.md").write_bytes((cache / "publisher-README.md").read_bytes())
        (data_root / "NOTICE.md").write_text(
            "Source: https://huggingface.co/datasets/roneneldan/TinyStories\n"
            + "Revision: " + profile["upstream"]["revision"] + "\n"
            + "License: CDLA-Sharing-1.0 https://cdla.dev/sharing-1-0/\n"
            + "Modified: selected training prefix; fit BPE on training stories only; tokenized and split validation.\n"
            + "New workload, not accepted92. Publisher card retained; consult license before redistribution.\n",
            encoding="utf-8")
    checked_profile()
    if digest(PROFILE) != profile_sha:
        raise ValueError("public data profile changed during workflow")
    phase(args.output, "raw-provenance-before-controls")
    if workload is not None:
        # Recheck after semantic validation and card copying as well, before
        # saving a successful workload. A mismatch preserves partial output.
        result["raw_input_provenance"] = verify_prepared_inputs(cache, profile, lock, manifest)
        if digest(data_root / "publisher-README.md") != lock["identity"]["files"]["publisher-README.md"]["sha256"]:
            raise ValueError("preserved publisher card differs from original raw lock")
    else:
        after = verify_cache(cache, profile["upstream"])
        if after["identity_sha256"] != lock["identity_sha256"] or after["identity"] != lock["identity"]:
            raise ValueError("raw lock identity changed across acquisition-only workflow")
    phase(args.output, "save-controls")
    if workload is not None:
        save(args.output / "workload.json", workload)
    save(args.output / "raw-lock.json", lock)
    save(args.output / "child-result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="fresh job directory; dataset goes in data/")
    parser.add_argument("--python", required=True, type=Path, help="existing intended CPU interpreter")
    parser.add_argument("--overlay", action="append", type=Path, default=[])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--acquire-only", action="store_true")
    parser.add_argument("--work-seconds", type=int, default=None,
                        help="child work watchdog, positive and at most the profile's 300 seconds; cleanup is separate")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.cache = args.cache.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.python = args.python.expanduser().absolute()  # preserve venv interpreter symlinks
    args.overlay = [path.expanduser().resolve() for path in args.overlay]
    profile = checked_profile()
    if args.work_seconds is None:
        args.work_seconds = profile["limits"]["workflow_seconds"]
    if not 1 <= args.work_seconds <= profile["limits"]["workflow_seconds"]:
        parser.error("--work-seconds must be positive and at most the pinned profile limit")
    if args._child:
        try:
            child(args, profile)
        except Exception as exc:
            # Diagnostics carry error class, never arbitrary URL/token-bearing text.
            save(args.output / "child-error.json", {"status": "blocked", "error_type": type(exc).__name__,
                 "effective_reward": None, "candidate_failure": False})
            return 2
        return 0
    for path in (args.cache, args.output):
        if path == HERE or HERE.is_relative_to(path) or path.is_relative_to(HERE):
            raise ValueError("raw/cache/job paths must be outside the source directory")
    if (args.cache == args.output or args.cache.is_relative_to(args.output)
            or args.output.is_relative_to(args.cache)):
        raise ValueError("raw cache and job output must be disjoint")
    if not args.python.is_file() or any(not path.is_dir() for path in args.overlay):
        raise ValueError("interpreter or explicit overlay missing")
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
               RAYON_NUM_THREADS="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               CUDA_VISIBLE_DEVICES="")
    if args.overlay:
        env["PYTHONPATH"] = os.pathsep.join(map(str, args.overlay))
    command = [str(args.python), "-s", str(Path(__file__).resolve()), "--_child", "--python", str(args.python),
               "--cache", str(args.cache), "--output", str(args.output), "--work-seconds", str(args.work_seconds)]
    if args.offline:
        command.append("--offline")
    if args.acquire_only:
        command.append("--acquire-only")
    from process_control import run_command
    result = {"schema": "training-public-data-supervisor-v1", "status": "blocked",
              "effective_reward": None, "agent_trainable": False, "candidate_failure": False,
              "gpu_used": False, "model_updates": 0, "profile_sha256": digest(PROFILE),
              "child_work_timeout_seconds": args.work_seconds,
              "whole_parent_wall_limit_enforced": False,
              "cleanup": "existing pinned process_control separate bounded tail",
              "launcher_runtime": {"python_executable": sys.executable, "python": platform.python_version()}}
    try:
        completed = run_command(command, cwd=args.output, env=env,
                                timeout=args.work_seconds, max_output_bytes=200000)
        result.update(returncode=completed.returncode, supervisor=completed.supervisor_metadata)
        (args.output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (args.output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode == 0:
            result.update(status="ok", child=json.loads((args.output / "child-result.json").read_text()))
    except Exception as exc:
        result.update(error_type=type(exc).__name__, supervisor=getattr(exc, "supervisor_metadata", None))
    save(args.output / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
