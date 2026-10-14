"""Synthetic CPU-only fixtures; these are not outer policy-training evidence."""
import argparse
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks import InvalidCandidate
from protocol import get_protocol
from tensor_hashes import tensor_state_sha256
from worker import train
from grader import grade

SOURCE = '''import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(4096))
    def forward(self, x): return self.bias.expand(*x.shape, 4096)
def build_model(): return Model()
def build_optimizer(m): return torch.optim.SGD(m.parameters(), lr=0.1)
def train_step(m, opt, x, y, fraction):
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(m(x).reshape(-1,4096), y.reshape(-1))
    loss.backward()
    opt.step()
    return -1000000.0
'''

class DiagnosticTests(unittest.TestCase):
    def test_protocol_compatibility_and_diagnostic(self):
        standard = get_protocol('standard')
        self.assertEqual((standard.steps,standard.batch,standard.context,standard.eval_batches,
                          standard.train_seconds_limit,standard.process_timeout), (500,32,256,60,300,600))
        self.assertEqual(standard.record()['version'], 'training-v1.1-fixed-steps')
        self.assertNotIn('diagnostic', standard.record())
        tiny = get_protocol('smoke')
        self.assertEqual((tiny.steps,tiny.batch,tiny.context,tiny.eval_batches,
                          tiny.train_seconds_limit,tiny.process_timeout), (3,2,16,2,30,90))
        p = get_protocol('smoke','cpu-training-v1')
        self.assertEqual((p.steps,p.batch,p.context,p.eval_batches,p.train_seconds_limit,p.process_timeout),
                         (20,4,64,4,90,150))
        self.assertEqual(p.record()['version'], 'training-diagnostic-cpu-v1')
        self.assertEqual(p.record()['diagnostic'], 'cpu-training-v1')
        for profile,diag in [('standard','cpu-training-v1'),('smoke','unknown')]:
            with self.assertRaises(ValueError): get_protocol(profile,diag)

    def test_hash_canonical_and_bfloat16(self):
        state = {'z':torch.tensor(3.),'a':torch.arange(4).to(torch.bfloat16)}
        digest = tensor_state_sha256(state)
        self.assertEqual(digest, tensor_state_sha256(dict(reversed(list(state.items())))))
        self.assertEqual(digest, tensor_state_sha256({k:v.clone() for k,v in state.items()}))
        altered = {k:v.clone() for k,v in state.items()}
        altered['a'][0] = 1
        self.assertNotEqual(digest,tensor_state_sha256(altered))
        self.assertNotEqual(tensor_state_sha256({'x':torch.zeros(2,2)}),
                            tensor_state_sha256({'x':torch.zeros(4)}))
        self.assertNotEqual(tensor_state_sha256({'x':torch.zeros(4)}),
                            tensor_state_sha256({'x':torch.zeros(4,dtype=torch.float64)}))

    def test_device_guard_precedes_data_loading(self):
        for device in ('mps','cuda','cuda:0'):
            args=argparse.Namespace(profile='smoke',diagnostic='cpu-training-v1',device=device)
            for operation in (train,grade):
                with self.subTest(device=device,operation=operation), self.assertRaisesRegex(InvalidCandidate,'requires device cpu'):
                    operation(args)

    def test_unchanged_parameters_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = SOURCE[:SOURCE.index('def train_step')] + 'def train_step(*args): return -1000000.0\n'
            (root/'candidate.py').write_text(source)
            np.save(root/'train.npy',np.zeros(2048,dtype=np.uint16))
            args = argparse.Namespace(candidate=str(root/'candidate.py'),train=str(root/'train.npy'),
                    checkpoint=str(root/'weights.pt'),profile='smoke',diagnostic='cpu-training-v1',device='cpu',seed=0)
            with self.assertRaisesRegex(InvalidCandidate,'did not change'):
                train(args)
            self.assertFalse((root/'weights.pt').exists())

    def test_actual_checkpoints_and_independent_losses(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'candidate.py').write_text(SOURCE)
            # Deliberately synthetic uniform target, held out in a separate fixture file.
            np.save(root/'train.npy',np.zeros(2048,dtype=np.uint16))
            np.save(root/'heldout.npy',np.zeros(2048,dtype=np.uint16))
            args=argparse.Namespace(candidate=str(root/'candidate.py'),train=str(root/'train.npy'),
                  checkpoint=str(root/'weights.pt'),heldout=str(root/'heldout.npy'),
                  profile='smoke',diagnostic='cpu-training-v1',device='cpu',seed=0,
                  reference_checkpoint=str(root/'initial-weights.pt'))
            result=train(args)
            self.assertEqual(result['steps'],20)
            self.assertEqual(result['training_tokens'],5120)
            for name,key in [('initial-weights.pt','initial_tensor_sha256'),('weights.pt','final_tensor_sha256')]:
                self.assertEqual(result[key],tensor_state_sha256(torch.load(root/name,weights_only=True)))
            self.assertNotEqual(result['initial_tensor_sha256'],result['final_tensor_sha256'])
            self.assertEqual(result['changed_state_tensors'],1)
            self.assertGreater(result['state_update_l2'],0)
            measured=grade(args)
            self.assertEqual(measured['eval_tokens'],1024)
            self.assertEqual(measured['loaded_final_tensor_sha256'],result['final_tensor_sha256'])
            self.assertEqual(measured['reference_tensor_sha256'],result['initial_tensor_sha256'])
            self.assertAlmostEqual(measured['before_held_out_loss'],math.log(4096),places=5)
            self.assertGreater(measured['held_out_loss'],0)
            self.assertLess(measured['after_held_out_loss'],measured['before_held_out_loss'])
            args.diagnostic=None
            with self.assertRaisesRegex(InvalidCandidate,'diagnostic-only'): grade(args)

if __name__=='__main__': unittest.main()
