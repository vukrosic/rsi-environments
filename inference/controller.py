"""Supervised evaluator entrypoint. Includes snapshot/preflight/loading/grading."""
import json
import sys
from pathlib import Path
from scorer import _evaluate_with_errors

def main(config, output):
    args = json.loads(Path(config).read_text())
    result = _evaluate_with_errors(**args)
    target = Path(output)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,allow_nan=False)+'\n')
    temporary.replace(target)

if __name__ == '__main__':
    main(*sys.argv[1:])
