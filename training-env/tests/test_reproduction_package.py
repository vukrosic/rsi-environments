import io
from pathlib import Path
import sys
import tarfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reproduction_check import inspect_members

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
    def test_incomplete_source_package_rejected(self):
        with self.assertRaisesRegex(ValueError,'missing required'):self.inspect('rsi/training-env/starter.py')

if __name__=='__main__':unittest.main()
