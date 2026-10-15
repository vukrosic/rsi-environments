"""One command: public TinyStories text -> fixed CUDA training -> native reward.

Orchestration only; reuse the existing preparer, calibration, evaluator, grader,
saved-state audit and reviewed process supervisor. No outer LLM optimizer.
"""
import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SPEC = HERE / "public_cuda_example-spec.json"


def read(path):
    path = Path(path)
    if path.stat().st_size > 1000000:
        raise ValueError("control exceeds 1 MB")
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def dependencies():
    # Import only the existing stdlib launcher helpers, after checking their
    # bytes without executing native/candidate modules.
    import hashlib
    spec = read(SPEC)
    for name, expected in spec["source_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid source dependency")
        path = HERE / relative
        if not path.is_file() or path.stat().st_size > 262144 or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("missing/changed public example dependency: " + name)
    import recipe_pair
    if Path(recipe_pair.__file__).resolve() != HERE / "recipe_pair.py":
        raise ValueError("launcher helper imported from another source root")
    return spec, recipe_pair


def query(argv, env):
    value = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=5, check=True)
    if len(value.stdout) + len(value.stderr) > 16000:
        raise ValueError("GPU inventory output exceeds bound")
    return value.stdout.strip()


def physical(env, spec):
    runtime = {"python": platform.python_version(),
               **{key: importlib.metadata.version(key) for key in ("torch", "numpy", "tokenizers")}}
    expected = {key: spec["runtime"][key] for key in runtime}
    if platform.system() != "Linux" or runtime != expected:
        raise ValueError("observed Linux Python/dependency lane required")
    gpu = query(["nvidia-smi", "--query-gpu=uuid,pci.bus_id,name,driver_version,memory.total", "--format=csv,noheader,nounits"], env)
    device = single_device(gpu)
    expected_uuid = env.get("RSI_PUBLIC_CUDA_EXPECTED_UUID")
    if expected_uuid and device["uuid"] != expected_uuid:
        raise ValueError("actual NVIDIA UUID differs from the operator's device pin")
    apps = query(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], env)
    if apps:
        raise ValueError("GPU compute applications present; preserve unrelated jobs")
    return {"gpu_identity": device, "compute_apps": [], "runtime": runtime,
            "python_executable": sys.executable,
            "memory_scope": "inventory only; no hard RSS/VRAM quota or continuous-idle proof"}


def single_device(text):
    rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError("exactly one NVIDIA device inventory row required")
    uuid,pci,name,driver,memory = (field.strip() for field in rows[0])
    if (not uuid.startswith("GPU-") or not all((pci,name,driver))
            or not memory.isdigit() or int(memory) <= 0):
        raise ValueError("incomplete NVIDIA UUID/PCI/name/driver/memory identity")
    return {"uuid":uuid,"pci_bus_id":pci,"name":name,"driver":driver,"memory_mib":int(memory)}


def prepared_gate(value, expected_version):
    child = value.get("child", {})
    proof = child.get("raw_input_provenance", {})
    if (value.get("status") != "ok" or child.get("status") != "prepared"
            or child.get("accepted92_equivalent") is not False
            or proof.get("version") != expected_version
            or proof.get("cache_reverified_after_preparation") is not True
            or proof.get("raw_identity_sha256") != child.get("raw_identity_sha256")
            or set(proof.get("preparer_input_sha256", {})) != {"train_text", "validation_text"}):
        raise ValueError("fresh data lacks successful v2 raw-input proof")
    supervision = value.get("supervisor", {})
    if supervision.get("status") != "ok" or supervision.get("cleanup_errors") or supervision.get("discovery_errors"):
        raise ValueError("raw preparation supervision did not pass")
    return child


def native_gate(record, spec, manifest_sha):
    m, p = record.get("metrics", {}), record.get("provenance", {})
    ce = m.get("held_out_loss")
    reward = record.get("reward")
    if (record.get("status") != "ok" or record.get("profile") != "standard"
            or m.get("protocol") != spec["budget"] or m.get("split") != "dev"
            or (m.get("steps"), m.get("training_tokens"), m.get("eval_tokens")) != (500,4096000,491520)
            or p.get("source_sha256") != spec["candidate_sha256"]
            or p.get("data_manifest_sha256") != manifest_sha
            or p.get("device") != "cuda" or p.get("data_kind") != "real"
            or type(ce) not in (int,float) or not math.isfinite(ce) or ce < 0
            or type(reward) not in (int,float) or not math.isfinite(reward)
            or not math.isclose(reward, math.exp(-ce), rel_tol=1e-12)):
        raise ValueError("native standard result/source/data/loss/reward gate failed")
    return {"loss": ce, "native_reward": reward}


def workflow(output, deadline):
    spec, helpers = dependencies()
    job = read(output / "job.json")
    if helpers.sha(SPEC) != job["spec_sha256"]:
        raise ValueError("example specification changed")
    if helpers.sha(Path(__file__)) != job["entrypoint_sha256"]:
        raise ValueError("example entry point changed")
    original_job_sha = helpers.sha(output / "job.json")
    env = helpers.child_environment(job["overlays"], job["canonical_lock"])
    if job.get("expected_gpu_uuid"):
        env["RSI_PUBLIC_CUDA_EXPECTED_UUID"] = job["expected_gpu_uuid"]
    env.update(OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    run = helpers.supervisor(HERE / "process_control.py")
    logs = output / "supervisors"
    logs.mkdir()
    result = {"status": "blocked", "effective_reward": None, "reward_consumed": False,
              "llm_weights_updated": False, "learning_gain": None, "session_accepted": False,
              "accepted92_equivalent": False, "model_execution": False, "phases": []}

    def phase(name, argv, cap):
        dependencies()
        if helpers.sha(SPEC) != job["spec_sha256"] or helpers.sha(output / "job.json") != original_job_sha:
            raise ValueError("example job/spec changed before phase")
        remaining = deadline - time.monotonic() - spec["phase_caps_seconds"]["cleanup_reserve"]
        if remaining <= 0:
            raise TimeoutError("whole example deadline exhausted")
        receipt = {"argv": argv, "timeout_seconds": min(cap,remaining)}
        try:
            completed = run(argv, cwd=output, env=env, timeout=receipt["timeout_seconds"], max_output_bytes=200000)
            receipt.update(returncode=completed.returncode, supervisor=completed.supervisor_metadata)
            (logs / (name + ".stdout.log")).write_text(completed.stdout)
            (logs / (name + ".stderr.log")).write_text(completed.stderr)
            if (completed.returncode != 0 or completed.supervisor_metadata["status"] != "ok"
                    or completed.supervisor_metadata["cleanup_errors"] or completed.supervisor_metadata["discovery_errors"]):
                raise ValueError("phase process/supervision failed: " + name)
        except BaseException as exc:
            receipt.update(error_type=type(exc).__name__,
                           supervisor=getattr(exc,"supervisor_metadata",receipt.get("supervisor")))
            raise
        finally:
            save(logs / (name + ".json"), receipt)
            result["phases"].append(name)

    before = None
    try:
        before = physical(env,spec)
        save(output / "physical-before.json", before)
        if job["preflight_only"]:
            result.update(status="preflight_only", model_execution=False,
                          data_preparation=False, native_reward=None)
        else:
            py = job["python"]
            prepare = [py,"-B",str(HERE / "prepare_public_data.py"),"--python",py,
                       "--cache",job["raw_cache"],"--output",str(output / "preparation"),"--work-seconds","300"]
            for overlay in job["overlays"]:
                prepare += ["--overlay",overlay]
            phase("raw-preparation",prepare,300)
            raw = prepared_gate(read(output / "preparation/result.json"),spec["raw_input_proof_version"])
            data = output / "preparation/data"
            manifest_sha = helpers.sha(data / "manifest.json")
            workload = read(output / "preparation/workload.json")
            if (not workload["workload_id"].startswith(spec["new_workload_family"] + ":")
                    or workload["native_manifest_sha256"] != manifest_sha):
                raise ValueError("new workload identity/native manifest binding failed")
            native = HERE / "recipe_pair_source/training-env"
            calibration = output / "calibration.json"
            phase("calibration",[py,"-B",str(native / "calibrate.py"),"--device","cuda","--idle-confirmed","--output",str(calibration)],90)
            calibrated = physical(env,spec)
            if calibrated["gpu_identity"] != before["gpu_identity"]:
                raise ValueError("physical GPU identity drift")
            save(output / "physical-calibrated.json",calibrated)
            calibration_sha = helpers.sha(calibration)
            episode = output / "episode"
            result["model_execution"] = None  # API failure may leave partial work
            phase("native-episode",[py,"-B",str(native / "evaluate.py"),"--candidate",str(HERE / spec["candidate"]),
                  "--data",str(data),"--output",str(episode),"--profile","standard","--device","cuda","--split","dev",
                  "--seed",str(spec["seed"]),"--calibration",str(calibration),"--state-evidence"],600)
            record = read(episode / "record.json")
            measured = native_gate(record,spec,manifest_sha)
            result["model_execution"] = True
            cal = record["provenance"].get("calibration",{})
            if (cal.get("receipt_sha256") != calibration_sha or not 0.9 <= cal.get("median_ratio",-1) <= 1.1
                    or not 0 <= cal.get("relative_spread",-1) <= 0.1):
                raise ValueError("fresh calibration/source binding failed")
            if record["provenance"].get("lease",{}).get("lock_path") != job["canonical_lock"]:
                raise ValueError("native common lease mismatch")
            packet = {"budget":spec["budget"],"candidate_sha256":spec["candidate_sha256"],
                      "physical_cuda_device_name":before["gpu_identity"]["name"],
                      "data_manifest_sha256":manifest_sha,"target_runtime":spec["runtime"],
                      "evaluator_sha256":{name:spec["source_sha256"]["recipe_pair_source/training-env/" + name]
                           for name in ("evaluate.py","worker.py","grader.py","protocol.py","checks.py","data.py","tensor_hashes.py")},
                      "suite_sha256":{name:spec["source_sha256"]["recipe_pair_source/suite/" + name] for name in ("records.py","resources.py")}}
            save(output / "audit-packet.json",packet)
            phase("saved-state-audit",[py,"-B",str(HERE / "verify_public_cuda_episode.py"),"--run",str(episode),
                  "--packet",str(output / "audit-packet.json"),"--training-root",str(native),"--output",str(output / "saved-audit.json")],30)
            audit = read(output / "saved-audit.json")
            if audit.get("saved_checkpoint_consistency") is not True or audit.get("actual_tensor_metadata_consistent") is not True:
                raise ValueError("independent saved-state audit failed")
            policies = []
            for role in ("worker","grader"):
                proof = read(episode / (role + "-operator.json"))
                inv = proof["cuda_inventory"]
                policy = (inv.get("matmul_allow_tf32"),inv.get("cudnn_allow_tf32"))
                if proof.get("torch_cuda_version") != spec["runtime"]["torch_cuda"] or any(type(v) is not bool for v in policy):
                    raise ValueError("actual CUDA build/TF32 metadata missing")
                policies.append(policy)
            if policies[0] != policies[1] or helpers.sha(calibration) != calibration_sha:
                raise ValueError("worker/grader policy or calibration drift")
            result.update(status="completed_pending_independent_review",**measured,
                          workload_id=workload["workload_id"], raw_identity_sha256=raw["raw_identity_sha256"],
                          native_manifest_sha256=manifest_sha, native_record_sha256=helpers.sha(episode / "record.json"),
                          initial_tensor_sha256=record["metrics"]["initial_tensor_sha256"],
                          final_tensor_sha256=record["metrics"]["final_tensor_sha256"],
                          raw_input_provenance=raw["raw_input_provenance"], model_execution=True,
                          final_npy_loss_graded=False, saved_audit_sha256=helpers.sha(output / "saved-audit.json"))
    except BaseException as exc:
        result.update(status="blocked",reason=type(exc).__name__ + ": " + str(exc))
    finally:
        try:
            dependencies()
            if (helpers.sha(SPEC) != job["spec_sha256"] or helpers.sha(Path(__file__)) != job["entrypoint_sha256"]
                    or helpers.sha(output / "job.json") != original_job_sha):
                raise ValueError("example source/job/spec changed")
            after = physical(env,spec)
            if before and after["gpu_identity"] != before["gpu_identity"]:
                raise ValueError("final GPU identity drift")
            # Probe only after native commands release their own lease.
            import importlib.util
            path = HERE / "recipe_pair_source/suite/resources.py"
            loader = importlib.util.spec_from_file_location("_public_example_lease",path)
            module = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(module)
            with module.accelerator_lock(device="cuda",timeout_seconds=0) as lease:
                after["common_lease_probe"] = lease
            save(output / "physical-cleanup.json",after)
            result["cleanup_control"] = "passed"
        except Exception as exc:
            result.update(status="blocked",cleanup_control="blocked",cleanup_reason=type(exc).__name__ + ": " + str(exc))
        save(output / "workflow.json",result)
    return 0 if result["status"] in ("preflight_only","completed_pending_independent_review") else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python",type=Path)
    parser.add_argument("--overlay",type=Path,action="append",default=[])
    parser.add_argument("--raw-cache",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--idle-confirmed",action="store_true")
    parser.add_argument("--preflight-only",action="store_true")
    parser.add_argument("--expected-gpu-uuid",help="optional operator pin; otherwise record the actual single NVIDIA device")
    parser.add_argument("--_execute",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--_deadline",type=float,help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if args._execute:
        if args._deadline is None or not math.isfinite(args._deadline):
            parser.error("internal finite deadline required")
        return workflow(output,args._deadline)
    if args.python is None or args.raw_cache is None or (not args.preflight_only and not args.idle_confirmed):
        parser.error("explicit --python/--raw-cache and full-run --idle-confirmed required")
    started = time.monotonic()
    spec,helpers = dependencies()
    cache = args.raw_cache.expanduser().resolve()
    helpers.guard_output(cache,output,HERE)
    py = args.python.expanduser().absolute()
    overlays = [str(path.expanduser().resolve()) for path in args.overlay]
    if not py.is_file() or not os.access(py,os.X_OK) or any(not Path(p).is_dir() for p in overlays):
        parser.error("existing executable/interpreter overlays required")
    lock = str(Path.home() / ".cache/vuk-research-envs/accelerator.lock")
    env = helpers.child_environment(overlays,lock)
    if args.expected_gpu_uuid:
        if not args.expected_gpu_uuid.startswith("GPU-") or len(args.expected_gpu_uuid) > 128:
            parser.error("invalid expected GPU UUID")
        env["RSI_PUBLIC_CUDA_EXPECTED_UUID"] = args.expected_gpu_uuid
    env.update(OPENBLAS_NUM_THREADS="2",TOKENIZERS_PARALLELISM="false")
    output.mkdir(parents=True,exist_ok=False)
    job = {"python":str(py),"overlays":overlays,"canonical_lock":lock,"raw_cache":str(cache),
           "preflight_only":args.preflight_only,"spec_sha256":helpers.sha(SPEC),"source_sha256":spec["source_sha256"],
           "entrypoint_sha256":helpers.sha(Path(__file__)),
           "expected_gpu_uuid":args.expected_gpu_uuid or env.get("RSI_PUBLIC_CUDA_EXPECTED_UUID"),
           "operator_idle_assertion":args.idle_confirmed,"source_preparation_clock_started":started}
    save(output / "job.json",job)
    budget = spec["preflight_whole_work_seconds"] if args.preflight_only else spec["whole_work_seconds"]
    deadline = started + budget
    result = {"status":"blocked","effective_reward":None,"reward_consumed":False,
              "llm_weights_updated":False,"session_accepted":False,"accepted92_equivalent":False}
    supervisor = {"work_seconds_limit":budget,"source_preparation_elapsed_deducted":True,
                  "job_sha256":helpers.sha(output / "job.json"), "spec_sha256":job["spec_sha256"],
                  "entrypoint_sha256":job["entrypoint_sha256"]}
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("whole example deadline exhausted during source setup")
        argv = [str(py),"-B",str(Path(__file__).resolve()),"--_execute","--output",str(output),"--_deadline",str(deadline)]
        supervisor["argv"] = argv
        run = helpers.supervisor(HERE / "process_control.py")
        completed = run(argv,cwd=output,env=env,timeout=remaining,max_output_bytes=200000)
        supervisor.update(returncode=completed.returncode,metadata=completed.supervisor_metadata)
        (output / "workflow.stdout.log").write_text(completed.stdout)
        (output / "workflow.stderr.log").write_text(completed.stderr)
        if (completed.returncode != 0 or completed.supervisor_metadata["status"] != "ok"
                or completed.supervisor_metadata["cleanup_errors"] or completed.supervisor_metadata["discovery_errors"]):
            raise ValueError("whole example execution/supervision failed")
        result.update(read(output / "workflow.json"))
        if result["status"] not in ("preflight_only","completed_pending_independent_review"):
            raise ValueError("workflow JSON gate failed")
    except BaseException as exc:
        supervisor.update(error_type=type(exc).__name__,metadata=getattr(exc,"supervisor_metadata",supervisor.get("metadata")))
        result.update(status="blocked",reason=type(exc).__name__ + ": " + str(exc))
    save(output / "whole-job-supervisor.json",supervisor)
    save(output / "result.json",result)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0 if result["status"] in ("preflight_only","completed_pending_independent_review") else 2


if __name__ == "__main__":
    sys.exit(main())
