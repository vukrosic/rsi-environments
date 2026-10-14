"""Reject malformed/mismatched manifests before pretrained assets are loaded."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from assets import HERE,validate_manifest

class AssetManifestTests(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads((HERE/'model.json').read_text())
        self.manifest=json.loads((HERE/'assets-manifest.json').read_text())
    def test_current_pinned_manifest_is_valid(self):
        self.assertIs(validate_manifest(self.manifest,self.spec),self.manifest)
    def test_empty_or_missing_weight_manifest_rejected(self):
        for missing in (None,'model.safetensors'):
            manifest=copy.deepcopy(self.manifest)
            if missing:del manifest['files'][missing]
            else:manifest['files']={}
            with self.assertRaises(ValueError):validate_manifest(manifest,self.spec)
    def test_revision_dtype_and_digest_mismatch_rejected(self):
        for field,value in [('revision','main'),('dtype','float16'),('attention','eager')]:
            manifest=copy.deepcopy(self.manifest);manifest[field]=value
            with self.assertRaises(ValueError):validate_manifest(manifest,self.spec)
        for record in [{'bytes':True,'sha256':'a'*64},{'bytes':1,'sha256':'bad'},{'bytes':-1,'sha256':'a'*64}]:
            manifest=copy.deepcopy(self.manifest);manifest['files']['config.json']=record
            with self.assertRaises(ValueError):validate_manifest(manifest,self.spec)
    def test_asset_path_escape_rejected(self):
        manifest=copy.deepcopy(self.manifest);manifest['files']['../private']={'bytes':1,'sha256':'a'*64}
        with self.assertRaises(ValueError):validate_manifest(manifest,self.spec)
    def test_checks_survive_optimized_python(self):
        script="from assets import validate_manifest; validate_manifest({'files':{}},{'id':'x','revision':'r','dtype':'float32','attention':'sdpa'})"
        result=subprocess.run([sys.executable,'-O','-c',script],cwd=HERE,text=True,capture_output=True,timeout=15)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('ValueError',result.stderr)

if __name__=='__main__':unittest.main()
