import io
from pathlib import Path
import sys
import tarfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reproduction_check import inspect_members, REQUIRED_FILES

class PackageSafetyTests(unittest.TestCase):
    def inspect(self,name,kind=tarfile.REGTYPE):
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w') as archive:
            entry=tarfile.TarInfo(name);entry.type=kind;entry.size=0;entry.linkname='/tmp/target' if kind==tarfile.SYMTYPE else ''
            archive.addfile(entry,io.BytesIO())
        buffer.seek(0)
        with tarfile.open(fileobj=buffer,mode='r') as archive:inspect_members(archive)
    def test_disallowed_members_rejected(self):
        for name,kind in [('rsi/../escape.py',tarfile.REGTYPE),('/absolute.py',tarfile.REGTYPE),('rsi/link.py',tarfile.SYMTYPE),('rsi/data/train.npy',tarfile.REGTYPE),('rsi/private/agent.jsonl',tarfile.REGTYPE)]:
            with self.subTest(name=name),self.assertRaises(ValueError):self.inspect(name,kind)
    def test_missing_runtime_dependencies_rejected(self):
        for missing in ('training-env/tensor_hashes.py','training-env/process_control.py'):
            buffer=io.BytesIO()
            with tarfile.open(fileobj=buffer,mode='w') as archive:
                for name in sorted(REQUIRED_FILES-{missing}):
                    entry=tarfile.TarInfo('rsi/'+name)
                    archive.addfile(entry,io.BytesIO())
            buffer.seek(0)
            with self.subTest(missing=missing), tarfile.open(fileobj=buffer,mode='r') as archive:
                with self.assertRaisesRegex(ValueError,missing): inspect_members(archive)

    def test_incomplete_source_package_rejected(self):
        with self.assertRaisesRegex(ValueError,'missing required'):self.inspect('rsi/training-env/starter.py')


class PackageBudgetTests(unittest.TestCase):
    def test_source_member_count_and_sizes_are_bounded(self):
        class Archive:
            def __init__(self,members):self.members=members
            def getmembers(self):return self.members
        for count,size in ((513,0),(1,2_000_001),(33,2_000_000)):
            members=[]
            for i in range(count):
                member=tarfile.TarInfo(f'rsi/training-env/source{i}.py');member.size=size;members.append(member)
            with self.subTest(count=count,size=size),self.assertRaisesRegex(ValueError,'exceeds'):
                inspect_members(Archive(members))

    def test_cumulative_deadline_rejects_before_new_process(self):
        import time
        from reproduction_check import run
        from process_control import CommandTimeout
        with patch('reproduction_check.run_command') as launch:
            with self.assertRaises(CommandTimeout):
                run(Path(sys.executable),Path('.'),['-c','pass'],{'TRAINING_REPRO_DEADLINE':str(time.monotonic()-1)})
            launch.assert_not_called()

if __name__=='__main__':unittest.main()
