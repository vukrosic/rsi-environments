"""Bounded CPU loader-overhead check with immutable receipts; no timing reward."""
import argparse
import json
import platform
import time
from pathlib import Path
import data

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--bypass',action='store_true')
    args=ap.parse_args();output=Path(args.output)
    if output.exists():ap.error('preserve old measurements; choose a new output')
    runs=[]
    data._SEMANTIC_CACHE.clear() # a deliberate fresh measurement, not a candidate assertion
    for _ in range(2):
        started=time.perf_counter();meta=data.load_data(args.data,use_cache=not args.bypass)[3]
        runs.append({'seconds':time.perf_counter()-started,'validation':meta['validation']})
    receipt={'runs':runs,'cpu_only':True,'gpu_used':False,'purpose':'functional overhead only; not a benchmark reward',
             'filesystem_cache_controlled':False,'python':platform.python_version(),'platform':platform.platform()}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'output':str(output),'seconds':[r['seconds'] for r in runs]}))
if __name__=='__main__':main()
