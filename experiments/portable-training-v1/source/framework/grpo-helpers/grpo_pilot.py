"""Training pilot on the kernel tasks: a small open model writes kernels, bench3 scores them, and a LoRA adapter is
updated from those rewards (GRPO). Smallest version: no held-out set and no claim that it learns.
  python grpo_pilot.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 20 --out pilot     on the GPU host
Each step: one task (round robin over --tasks), --group samples from the current policy, each benched by bench3.
Sampling is this file's own loop, plain softmax(logits / temp) with no top-p, top-k, penalty or generation defaults,
so the update's log-likelihood is the sampler's. Each step records the largest gap between the log-prob the sampler
drew each token with and the one the update recomputes. Weights load in float32: on SmolLM2-135M the gap was
3.6e-5 nats in float32 and 0.28 in bfloat16 (KV-cached vs full forward), so bfloat16 is not on-policy; the update
therefore uses a PPO-clipped ratio against the sampler's log-probs (eps 0.2), which equals the plain policy gradient
when the two agree. Default tasks: the 9 training tasks; decode_attention, selective_scan, int4_matmul are held out.
Reward: 0 when the candidate is at fault (no code, scan rejection, wrong, crash inside the kernel: bench3 "fault":
"candidate"), else bench3's speedup clipped to [0, 4]. Any other outcome (jail setup failure, bench crash without a
verdict, timeout of the bench itself) is excluded: kept in the log with its reason, left out of the mean, std and
gradient. Advantage: reward minus the mean of the counted samples, over their std. One Adam step on the adapter
only; the base weights stay frozen. On CUDA the policy moves to the CPU while bench3 times, so the GPU is idle.
Writes OUT/steps.jsonl (every reward and failure reason), OUT/samples/<step>/<i>.{txt,py}, OUT/hashes.json (base
weights before and after, adapter before and after), OUT/adapter.pt. The last line printed is JSON."""
import argparse, fcntl, hashlib, json, math, time
from pathlib import Path
import torch
import sample

TRAIN = ["rmsnorm", "softmax", "swiglu", "cross_entropy", "attention", "attention_bwd", "newton_schulz",
         "sliding_window_attention", "gqa_attention"]  # decode_attention, selective_scan, int4_matmul stay held out
CLIP = 0.2


class LoRA(torch.nn.Module):
    """base(x) + scale * B A x, with A, B in float32; B starts at zero, so step 0 is the base model."""
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        dev = base.weight.device
        self.a = torch.nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features))
        self.b = torch.nn.Parameter(torch.zeros(base.out_features, r, device=dev))

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.a.T) @ self.b.T * self.scale).to(x.dtype)


def attach(model, r, alpha, names=("q_proj", "v_proj")):
    model.requires_grad_(False)
    for full, mod in list(model.named_modules()):
        for n, child in list(mod.named_children()):
            if n in names and isinstance(child, torch.nn.Linear):
                setattr(mod, n, LoRA(child, r, alpha))
    return [p for p in model.parameters() if p.requires_grad]


def digest(named):
    h = hashlib.sha256()
    for name, t in sorted(named):
        h.update(name.encode())
        h.update(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def base_named(model):
    return [(n, p) for n, p in model.named_parameters() if not n.endswith((".a", ".b"))]


def lora_named(model):
    return [(n, p) for n, p in model.named_parameters() if n.endswith((".a", ".b"))]


@torch.no_grad()
def sample_group(model, ids, n, temp, max_new, eos):
    """n samples from softmax(logits / temp) with a KV cache. Returns tokens (n, L) and each token's log-prob."""
    out = model(ids.repeat(n, 1), use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1].float()
    done = torch.zeros(n, dtype=torch.bool, device=ids.device)
    toks, lps = [], []
    for _ in range(max_new):
        lp = torch.log_softmax(logits / temp, -1)
        t = torch.multinomial(lp.exp(), 1)[:, 0]
        t = torch.where(done, torch.full_like(t, eos), t)  # finished samples repeat the end token; cut off later
        toks.append(t)
        lps.append(lp.gather(1, t[:, None])[:, 0])
        done |= t == eos
        if done.all():
            break
        out = model(t[:, None], past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1].float()
    return torch.stack(toks, 1), torch.stack(lps, 1)


def token_logprobs(model, full, start, temp, chunk=256):
    """log-prob of full[start:] given what comes before, under softmax(logits / temp). The output layer runs on
    `chunk` positions at a time, so a 150k vocabulary never needs (length x vocab) floats at once."""
    h = model.model(full.unsqueeze(0)).last_hidden_state[0, start - 1:-1]
    head, out = model.get_output_embeddings(), []
    for i in range(0, h.shape[0], chunk):
        lg = head(h[i:i + chunk]).float() / temp
        out.append(torch.log_softmax(lg, -1).gather(1, full[start + i:start + i + chunk].unsqueeze(1))[:, 0])
    return torch.cat(out)


def reward(has_code, b):
    """(reward, None) for a counted sample, (None, reason) for one left out of the update."""
    if not has_code:
        return 0.0, None
    if b.get("ok") is True:
        s = b.get("speedup")
        if isinstance(s, (int, float)) and math.isfinite(s):
            return float(min(max(s, 0.0), 4.0)), None
        return None, f"ok without a finite speedup: {s!r}"
    if b.get("fault") == "candidate":
        return 0.0, None
    return None, f"not the candidate's zero: {b.get('fault', 'no fault label')}, {b.get('why')}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--tasks", nargs="+", default=TRAIN)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--group", type=int, default=8, help="samples per step")
    p.add_argument("--trials", type=int, default=3, help="bench3 trials per sample")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--out", default="pilot")
    p.add_argument("--save-every", type=int, default=0, help="also save the adapter as adapter_<step>.pt this often")
    p.add_argument("--lock", help="flock this file for each step only, so other GPU users (the bench queue) run between steps; "
                                  "the policy waits on the CPU while unlocked")
    p.add_argument("--ckpt", action="store_true", help="gradient checkpointing, for models whose activations do not fit")
    p.add_argument("--stay-on-gpu", action="store_true", help="never park the model in CPU RAM (bench and between steps); "
                                                              "saves ~3 GB RSS on the 7.8 GB host, keeps ~3 GB VRAM resident")
    a = p.parse_args()
    park = a.device if a.stay_on_gpu else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = (sample.here / a.out).resolve()
    (out / "samples").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(park if a.lock else a.device)
    params = attach(model, a.rank, 2 * a.rank)
    if a.ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False  # the sampler passes use_cache=True itself
    opt = torch.optim.Adam(params, lr=a.lr)
    hashes = {"model": a.model, "base_start": digest(base_named(model)), "adapter_start": digest(lora_named(model)),
              "trainable_params": sum(p.numel() for p in params)}
    prompts = sample.prompts_for(a.tasks)
    t_start, log = time.time(), open(out / "steps.jsonl", "w")
    for step in range(a.steps):
        lk, wait_s = open(a.lock, "a") if a.lock else None, 0.0
        if lk:  # held for this step only; closing the file releases it
            t0 = time.time()
            fcntl.flock(lk, fcntl.LOCK_EX)
            wait_s = time.time() - t0
            model.to(a.device)
        task = a.tasks[step % len(a.tasks)]
        ids = tok.apply_chat_template([{"role": "user", "content": prompts[task]}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)["input_ids"].to(a.device)
        t0 = time.time()
        model.eval()
        new, samp_lp = sample_group(model, ids, a.group, a.temp, a.max_new, tok.eos_token_id)
        gen_s = time.time() - t0
        if a.device == "cuda":
            model.to(park)
            torch.cuda.empty_cache()  # bench3 times on an idle GPU
        d = out / "samples" / str(step)
        d.mkdir(exist_ok=True)
        rows, t0 = [], time.time()
        for i in range(a.group):
            seq = new[i]
            stop = (seq == tok.eos_token_id).nonzero()
            seq = seq[: int(stop[0]) + 1] if len(stop) else seq  # keep the first end token: it is part of the action
            text = tok.decode(seq, skip_special_tokens=True)
            (d / f"{i}.txt").write_text(text)
            c = sample.code(text)
            b = {}
            if c is not None:
                (d / f"{i}.py").write_text(c)
                try:
                    b = sample.bench(task, d / f"{i}.py", a.trials)
                except Exception as e:  # the bench itself failed, e.g. its 1800 s timeout: not the candidate's zero
                    b = {"ok": False, "why": f"bench call raised {type(e).__name__}: {e}"[:300]}
            r, excluded = reward(c is not None, b)
            rows.append({"tokens": seq, "samp_lp": samp_lp[i, :len(seq)], "has_code": c is not None, "bench": b,
                         "r": r, "excluded": excluded})
        bench_s = time.time() - t0
        if a.device == "cuda":
            model.to(a.device)
        counted = [r for r in rows if r["excluded"] is None]
        rs = torch.tensor([r["r"] for r in counted])
        std = float(rs.std()) if len(rs) > 1 else 0.0
        rec = {"step": step, "task": task, "rewards": [r["r"] for r in rows], "counted": len(counted),
               "excluded": [r["excluded"] for r in rows], "has_code": sum(r["has_code"] for r in rows),
               "correct": sum(r["bench"].get("ok") is True for r in rows), "mean": float(rs.mean()) if len(rs) else None,
               "std": std, "why": [r["bench"].get("why", "no code" if not r["has_code"] else None) for r in rows],
               "bench": [r["bench"] for r in rows], "gen_s": round(gen_s, 1), "bench_s": round(bench_s, 1)}
        if std < 1e-6:  # fewer than 2 counted samples, or all scored the same: no advantage, no update
            rec.update(update=False, loss=None, grad_norm=None, max_lp_gap=None)
        else:
            adv = (rs - rs.mean()) / (std + 1e-4)
            model.train()
            opt.zero_grad()
            n_tok = sum(len(r["tokens"]) for r in counted)
            loss_sum, gap, clipped = 0.0, 0.0, 0
            for r, A in zip(counted, adv.tolist()):
                if A == 0 or len(r["tokens"]) == 0:
                    continue
                full = torch.cat([ids[0], r["tokens"]])
                tok_lp = token_logprobs(model, full, ids.shape[1], a.temp)
                old = r["samp_lp"].to(tok_lp.device)
                gap = max(gap, float((tok_lp.detach() - old).abs().max()))
                # PPO-clipped ratio against the sampler's own log-probs. In float32 the ratio is 1 to ~1e-5 and the
                # gradient equals the plain policy gradient; in bfloat16 it corrects the sampler/update mismatch.
                ratio = torch.exp(tok_lp - old)
                surr = torch.minimum(ratio * A, ratio.clamp(1 - CLIP, 1 + CLIP) * A)
                clipped += int(((ratio - 1).abs() > CLIP).sum())
                loss = -surr.sum() / n_tok  # one constant normaliser for the group
                loss.backward()
                loss_sum += float(loss)
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            rec.update(update=True, loss=round(loss_sum, 6), grad_norm=round(gn, 6), max_lp_gap=round(gap, 6),
                       clipped_tokens=clipped, tokens=n_tok)
        if a.save_every and (step + 1) % a.save_every == 0:
            torch.save({n: t.detach().cpu() for n, t in lora_named(model)}, out / f"adapter_{step + 1}.pt")
        rec["lock_wait_s"] = round(wait_s, 1)
        if lk:
            model.to(park)
            torch.cuda.empty_cache()
            lk.close()
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"step {step:3} {task:14} code {rec['has_code']}/{a.group} correct {rec['correct']}/{a.group} "
              f"counted {rec['counted']}/{a.group} mean reward {rec['mean']} update {rec['update']}", flush=True)
    hashes.update(base_end=digest(base_named(model)), adapter_end=digest(lora_named(model)))
    hashes["base_unchanged"] = hashes["base_start"] == hashes["base_end"]
    hashes["adapter_changed"] = hashes["adapter_start"] != hashes["adapter_end"]
    torch.save({n: t.detach().cpu() for n, t in lora_named(model)}, out / "adapter.pt")
    (out / "hashes.json").write_text(json.dumps(hashes, indent=1))
    steps = [json.loads(l) for l in open(out / "steps.jsonl")]
    res = {**hashes, "steps": len(steps), "updates": sum(s["update"] for s in steps), "samples": len(steps) * a.group,
           "bench_calls": sum(s["has_code"] for s in steps), "correct": sum(s["correct"] for s in steps),
           "excluded": sum(a.group - s["counted"] for s in steps),
           "max_lp_gap": max([s["max_lp_gap"] for s in steps if s["max_lp_gap"] is not None], default=None),
           "mean_reward_by_step": [None if s["mean"] is None else round(s["mean"], 3) for s in steps], "minutes": round((time.time() - t_start) / 60, 1),
           "gpu": torch.cuda.get_device_name() if a.device == "cuda" else a.device}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
