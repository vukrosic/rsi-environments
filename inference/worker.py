"""Evaluator-owned child. Baseline and candidate run in separate processes."""
import argparse
import importlib.util
import json
import time
import secrets
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from assets import local_model
from cache_probes import run as run_cache_probes

def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
        if torch.mps.driver_allocated_memory() > 4_000_000_000:
            raise MemoryError('sampled MPS driver memory exceeds 4 GB')

def load_engine(path):
    spec = importlib.util.spec_from_file_location('submission', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Engine

def run(source, inputs, output, device):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(local_model(), local_files_only=True,
        dtype=torch.float32, attn_implementation='sdpa').eval().to(device)
    init_start = time.perf_counter()
    engine = load_engine(source)(model)
    engine_init_s = time.perf_counter() - init_start
    if torch.get_num_threads() != 2:
        raise ValueError('submission changed CPU thread budget')
    task = torch.load(inputs, weights_only=True)
    rows = []
    setup_start = time.perf_counter()
    with torch.inference_mode():
        # Two warm requests; startup/conversion/loading excluded from timing.
        warm_cases = []
        warm_generator = torch.Generator().manual_seed(secrets.randbits(63))
        for length in (32, 128, 512):
            # Distinct from every scored prompt, so cross-request reuse gets no
            # artificially warmed answers. Warm only shapes and dispatch paths.
            for _ in range(2):
                warm_cases.append({'ids': torch.randint(3, model.config.vocab_size,
                    (1, length), generator=warm_generator)})
        for warm in warm_cases:
            logits = engine.prefill(warm['ids'].to(device))
            engine.decode(logits.argmax(-1, keepdim=True))
        sync(device)
        warmup_s = time.perf_counter() - setup_start
        for case in task['cases']:
            ids = case['ids'].to(device)
            before = ids.clone()
            sync(device)
            start = time.perf_counter()
            logits = engine.prefill(ids)
            sync(device)
            prefill_s = time.perf_counter() - start
            first = logits.detach().float().cpu().clone()
            trace = []
            tokens = []
            sync(device)
            start = time.perf_counter()
            for _ in range(task['decode_steps']):
                token = logits.argmax(-1, keepdim=True)
                tokens.append(token)
                logits = engine.decode(token)
                trace.append(logits.clone())
            sync(device)
            decode_s = time.perf_counter() - start
            # Copy outside timer; all timed outputs checked, including every token.
            greedy = torch.stack(trace).detach().float().cpu().clone()
            token_ids = torch.stack(tokens).cpu().clone()
            engine.prefill(ids)
            forced = []
            for t in case['forced']:
                forced_token = t.to(device).clone()
                forced.append(engine.decode(forced_token).detach().float().cpu().clone())
                if not torch.equal(forced_token.cpu(), t):
                    raise ValueError('forced token mutation')
            # Baseline checks its cached outputs against an uncached full-context
            # pass. The candidate receives no oracle through its Engine API;
            # filesystem confidentiality requires an external sandbox.
            if source.name == 'reference.py':
                full_ids = torch.cat([ids, torch.cat(tokens, dim=1)], dim=1)
                uncached = model(input_ids=full_ids, use_cache=False, logits_to_keep=1).logits[:, -1, :].float().cpu()
                torch.testing.assert_close(uncached, greedy[-1], atol=0.002, rtol=0.002)
                torch.testing.assert_close(model(input_ids=torch.cat([ids, case['forced'].flatten().view(1,-1).to(device)], dim=1),
                    use_cache=False, logits_to_keep=1).logits[:, -1, :].float().cpu(), forced[-1], atol=0.002, rtol=0.002)
            if not torch.equal(ids, before):
                raise ValueError('input mutation')
            rows.append(dict(key=case['key'], length=ids.shape[1],
                prefill_s=prefill_s, decode_s=decode_s, first=first,
                greedy=greedy, tokens=token_ids, forced=torch.stack(forced)))
    memory = {'allocated_bytes': torch.mps.current_allocated_memory(),
              'driver_bytes': torch.mps.driver_allocated_memory()} if device == 'mps' else {}
    probes=[]
    if 'cache_probe_inputs' in task:
        with torch.inference_mode():
            oracle=(lambda ids:model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[:,-1,:]) if source.name=='reference.py' else None
            probes=run_cache_probes(engine,task['cache_probe_inputs'],device,oracle)
        sync(device)
    torch.save(dict(rows=rows, cache_probes=probes, warmup_s=warmup_s, engine_init_s=engine_init_s,
        mps_memory=memory, torch_threads=torch.get_num_threads()), output)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('inputs', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    a = p.parse_args()
    run(a.source, a.inputs, a.output, a.device)
