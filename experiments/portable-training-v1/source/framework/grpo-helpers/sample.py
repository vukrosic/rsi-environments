"""Feasibility check for training a model on the kernel tasks: can a small model write kernels that bench3 scores?
Inference only, no weight updates. The model writes n kernels per task; each is timed by bench3 against the reference.
  python sample.py --model Qwen/Qwen2.5-0.5B-Instruct --n 8 --out feas      on the GPU host (the model downloads there)
  python sample.py --echo --n 2 --trials 2 --out feas-echo                     no model: each "sample" is start/<task>.py
Writes OUT/<task>/<i>.txt (the model's reply), OUT/<task>/<i>.py (the code taken from it), OUT/results.jsonl, and
prints per task: replies with code, correct kernels, best speedup, and the best sample timed again. The last line is JSON."""
import argparse, ast, json, os, re, subprocess, sys, time
from pathlib import Path

here = Path(__file__).resolve().parent
INPUTS = {  # what TASKS in tasks.py makes, in words
    "rmsnorm": "x of 8192 by 1024 and w of 1024, float32",
    "softmax": "x of 4096 by 4096, float32",
    "swiglu": "a and b of 8192 by 2816, float32",
    "cross_entropy": "logits of 8192 by 4096, float32, and y of 8192 integer labels (int64)",
    "attention": "q, k, v of batch 8, 8 heads, 1024 positions, head size 64, float32",
    "newton_schulz": "g of 10 matrices of 704 by 256, float32",
    "attention_bwd": "q, k, v and do (the output gradient) of batch 8, 8 heads, 1024 positions, head size 64, float32",
    "selective_scan": "x and dt of batch 8, 128 steps, 256 channels; A of 256 by 16; Bm and C of batch 8, 128 steps, 16; float32",
    "sliding_window_attention": "q, k, v of batch 1, 8 heads, 2048 positions, head size 64, float32",
    "decode_attention": "q of batch 4, 8 heads, 1 position, head size 128; k and v of batch 4, 8 heads, 4096 positions, head size 128; float32",
    "gqa_attention": "q of batch 1, 32 heads, 1024 positions, head size 64; k and v of batch 1, 8 heads, 1024 positions, head size 64; float32",
    "int4_matmul": "x of 32 by 4096 float16; w of 4096 by 2048 uint8 (two 4-bit weights per byte); scale of 4096 by 32 float16",
}
PROMPT = """Write a fast PyTorch kernel for an NVIDIA GPU (device "cuda").

The reference, from tasks.py:
```python
{ref}
```
{timed}
Inputs: {inputs}. Your kernel must return the same values as the reference (relative error at most {tol}), as a tensor
with the reference's dtype, device and shape. Its speedup is the baseline's time over yours; faster is better. You may use any
PyTorch operation or torch.compile. It may not read files, start processes or use threads.

Reply with one Python code block that defines kernel{args} and imports what it needs."""


def baseline(name):
    """The timing baseline's source: PyTorch's own op for the task (lib_<name> in tasks.py), or None (timed against the
    reference itself)."""
    src = (here / "tasks.py").read_text()
    fn = next((n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "lib_" + name), None)
    return fn and ast.get_source_segment(src, fn)


def reference(name):
    src = (here / "tasks.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    tol = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "TASKS")
    tols = {k.value: v.elts[2].value for k, v in zip(tol.value.keys, tol.value.values)}
    return ast.get_source_segment(src, fn), "(" + ", ".join(a.arg for a in fn.args.args) + ")", tols[name]


def code(reply):
    """The last code block that defines kernel(). Failing that, the last block with exactly one top-level function
    whose argument count matches is taken, with `kernel = <name>` appended (small models often rename it)."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, re.S)
    named = [b for b in blocks if re.search(r"^def kernel\(", b, re.M)]
    if named:
        return named[-1]
    if re.search(r"^def kernel\(", reply, re.M):
        return reply
    for b in reversed(blocks):
        try:
            fns = [n for n in ast.parse(b).body if isinstance(n, ast.FunctionDef)]
        except SyntaxError:
            continue
        if len(fns) == 1:
            return b.rstrip() + f"\n\nkernel = {fns[0].name}  # renamed by sample.code()\n"
    return None


def generate(a, prompts):
    if a.echo:
        return {t: ["```python\n" + (here / "start" / f"{t}.py").read_text() + "```"] * a.n for t in prompts}, 0
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to("cuda").eval()
    if a.adapter:  # a LoRA adapter saved by grpo_pilot.py or sft.py, attached the same way
        import grpo_pilot
        grpo_pilot.attach(model, a.rank, 2 * a.rank)
        sd = torch.load(a.adapter, map_location="cuda")
        missing = model.load_state_dict(sd, strict=False)
        assert not missing.unexpected_keys and len(sd) == len(grpo_pilot.lora_named(model)), missing
    torch.manual_seed(a.seed)
    out, ntok = {}, 0
    for t, p in prompts.items():
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to("cuda")
        with torch.no_grad():
            g = model.generate(ids, do_sample=True, temperature=a.temp, top_p=0.95, max_new_tokens=a.max_new,
                               num_return_sequences=a.n, pad_token_id=tok.eos_token_id)
        new = g[:, ids.shape[1]:]
        ntok += int((new != tok.eos_token_id).sum())
        out[t] = tok.batch_decode(new, skip_special_tokens=True)
    del model
    torch.cuda.empty_cache()  # the bench gets the whole GPU
    return out, ntok


def bench(t, path, trials):
    r = subprocess.run([sys.executable, "-W", "ignore", str(here / "bench3.py"), t, str(path), "--trials", str(trials)],
                       cwd=here, capture_output=True, text=True, timeout=1800)
    lines = r.stdout.strip().splitlines()
    try:
        return json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        return {"ok": False, "why": "crashed: " + (r.stderr.strip().splitlines() or [""])[-1][:200]}


def prompts_for(tasks):
    out = {}
    for t in tasks:
        ref, args, tol = reference(t)
        b = baseline(t)
        timed = ("It is timed against the reference itself (PyTorch has no fused op for it).\n" if b is None else
                 f"It is timed against this baseline, PyTorch's own op for the same math:\n```python\n{b}\n```\n")
        out[t] = PROMPT.format(ref=ref, inputs=INPUTS[t], tol=tol, args=args, timed=timed)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--tasks", nargs="+", default=list(INPUTS))
    p.add_argument("--n", type=int, default=8, help="samples per task")
    p.add_argument("--trials", type=int, default=5, help="bench3 trials per sample")
    p.add_argument("--retrials", type=int, default=15, help="bench3 trials when the best sample is timed again")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="feas")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--adapter", help="LoRA adapter .pt to attach (default: the base model)")
    p.add_argument("--rank", type=int, default=16, help="the adapter's LoRA rank")
    p.add_argument("--echo", action="store_true", help="no model: test the pipeline with the start kernels")
    a = p.parse_args()
    out = (here / a.out).resolve()
    prompts = prompts_for(a.tasks)
    t0 = time.time()
    replies, ntok = generate(a, prompts)
    gen_s = time.time() - t0
    rows, summary = [], {}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w") as f:
        for t, rs in replies.items():
            (out / t).mkdir(exist_ok=True)
            (out / t / "prompt.txt").write_text(prompts[t])
            for i, reply in enumerate(rs):
                (out / t / f"{i}.txt").write_text(reply)
                c = code(reply)
                row = {"task": t, "i": i, "has_code": c is not None}
                if c is not None:
                    (out / t / f"{i}.py").write_text(c)
                    row["bench"] = bench(t, out / t / f"{i}.py", a.trials)
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
            mine = [r for r in rows if r["task"] == t]
            ok = [r["bench"]["speedup"] for r in mine if r.get("bench", {}).get("ok")]
            summary[t] = {"samples": len(mine), "has_code": sum(r["has_code"] for r in mine), "correct": len(ok),
                          "best_speedup": max(ok) if ok else None, "best_retimed": None}
            if ok:  # the max of n noisy timings is biased up: time the best sample again with more trials
                best = max((r for r in mine if r.get("bench", {}).get("ok")), key=lambda r: r["bench"]["speedup"])
                again = bench(t, out / t / f"{best['i']}.py", a.retrials)
                summary[t]["best_retimed"] = again.get("speedup") if again.get("ok") else 0
            s = summary[t]
            print(f"{t:14} code {s['has_code']}/{s['samples']}  correct {s['correct']}/{s['samples']}  best {s['best_speedup']}"
                  f"  retimed {s['best_retimed']}", flush=True)
    res = {"model": None if a.echo else a.model, "adapter": a.adapter, "n": a.n, "trials": a.trials, "retrials": a.retrials, "temp": a.temp, "tasks": summary,
           "gen_tokens": ntok, "gen_s": round(gen_s), "total_s": round(time.time() - t0)}
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
