#!/bin/sh
# bench3 validity check on one CUDA GPU. Run from kernels/ under the host accelerator lease, as root on Linux so the
# kernel runs under a uid of its own (isolation "uid"). Raw results go to remote/out/; the exit code is remote/gate.py's:
# 0 only if the jail self-test and every expected row pass. Collection steps never stop the script; the gate decides.
# Needs: python with torch+CUDA, numpy. No network. Peak ~2.3 GB VRAM. ~25 min.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
ISO=${ISO:-uid}  # pgroup only for a labelled run without the jail (BENCH_JAIL=0 or not root); the escape hack is then not a valid test
OUT=${OUT:-remote/out}
mkdir -p "$OUT"
: > "$OUT/unchanged.jsonl"
df -h /dev/shm > "$OUT/shm.txt" 2>&1
"$PY" -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))" > "$OUT/env.txt" 2>&1
id >> "$OUT/env.txt"
if [ "$ISO" = uid ]; then
  "$PY" remote/jail_test.py > "$OUT/jail_test.json" 2>&1
  jail=$?
  if [ "$jail" -ne 0 ]; then  # a failed jail must not spend the GPU slot
    cat "$OUT/jail_test.json"
    echo "jail_test failed: GPU collection not started" >&2
    exit 1
  fi
else
  echo '{"ok": false, "why": "not run: ISO=pgroup"}' > "$OUT/jail_test.json"
  jail=0
fi
for run in 1 2; do  # start/ kernels call the library op, so they are timed against it (the default fastest baseline beats them)
  for t in swiglu softmax rmsnorm cross_entropy newton_schulz attention; do
    line=$("$PY" -W ignore bench3.py "$t" "start/$t.py" --baseline library --trials "${TRIALS:-15}" 2>>"$OUT/stderr.txt" | tail -1)
    echo "{\"run\": $run, \"task\": \"$t\", \"result\": ${line:-null}}" >> "$OUT/unchanged.jsonl"
  done
done
"$PY" -W ignore hacks/run.py attention --bench bench3.py --trials "${HACK_TRIALS:-5}" > "$OUT/hacks_scan.txt" 2>&1
"$PY" -W ignore hacks/run.py attention --bench bench3.py --no-scan --trials "${HACK_TRIALS:-5}" > "$OUT/hacks_noscan.txt" 2>&1
"$PY" remote/gate.py "$OUT" --isolation "$ISO" > "$OUT/gate.json"
gate=$?
cat "$OUT/jail_test.json" "$OUT/gate.json"
[ "$jail" -eq 0 ] && [ "$gate" -eq 0 ]
