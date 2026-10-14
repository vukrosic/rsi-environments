"""Untimed request-cache correctness: reset, prefix reuse, branches, batches."""
import torch

VERSION = 1

class ContractViolation(ValueError):
    """An evaluator-owned output-shape or input-mutation check failed."""

def inputs(nonce,vocab_size):
    generator=torch.Generator().manual_seed(int(nonce[-15:],16))
    def draw(shape):
        return torch.randint(3,vocab_size,shape,generator=generator)
    return {'a':draw((1,7)), 'b':draw((1,11)), 'extension':draw((1,4)),
            'branches':draw((3,1,1)), 'batch':draw((2,9)), 'batch_tokens':draw((2,1))}

def run(engine,probes,device,oracle=None):
    """Reference oracle is an independent uncached full-context model call."""
    rows=[]
    context=None
    def record(key,logits):
        actual=logits.detach().float().cpu().clone()
        if actual.ndim != 2 or actual.shape[0] != context.shape[0] or not torch.isfinite(actual).all():
            raise ContractViolation(f'cache probe {key}: shape/nonfinite output')
        if oracle is not None:
            expected=oracle(context).detach().float().cpu()
            torch.testing.assert_close(actual,expected,atol=.002,rtol=.002)
            if not torch.equal(actual.argmax(-1),expected.argmax(-1)):
                raise AssertionError(f'cache oracle token mismatch: {key}')
        rows.append({'key':key, 'context':context.detach().cpu().clone(), 'logits':actual})
    def prefill(key,ids):
        nonlocal context
        value=ids.to(device).clone()
        before=value.clone()
        context=before
        record(key,engine.prefill(value))
        if not torch.equal(before,value):
            raise ContractViolation(f'cache probe {key}: prompt mutation')
    def decode(key,token):
        nonlocal context
        value=token.to(device).clone()
        before=value.clone()
        context=torch.cat([context,before],dim=1)
        record(key,engine.decode(value))
        if not torch.equal(before,value):
            raise ContractViolation(f'cache probe {key}: token mutation')
    a,b=probes['a'],probes['b']
    branches=probes['branches']
    prefill('request-a',a)
    decode('request-a-step',branches[0])
    prefill('request-b',b)
    prefill('request-a-after-b',a)
    decode('branch-left-1',branches[0])
    decode('branch-left-2',branches[1])
    prefill('branch-right-reset',a)
    decode('branch-right-1',branches[2])
    prefill('repeated-prefix-extension',torch.cat([a,probes['extension']],dim=1))
    prefill('shorter-prefix-reset',a[:,:3])
    prefill('batch-two-reset',probes['batch'])
    decode('batch-two-step',probes['batch_tokens'])
    prefill('batch-one-after-two',a)
    return rows
