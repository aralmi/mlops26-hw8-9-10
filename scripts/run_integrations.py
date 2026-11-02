"""Machine-readable results in addition to the real unittest transcript."""
import argparse
import importlib.util
import json
import sys
import unittest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1]/'integration/test_incidents.py'
    spec = importlib.util.spec_from_file_location('incident_regressions', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    def identifier(test):
        return test.test_case.id() if hasattr(test, 'test_case') else test.id()
    payload = dict(tests=result.testsRun,
                   failures=[{'test':identifier(t), 'case':str(t), 'traceback':trace} for t,trace in result.failures],
                   errors=[{'test':identifier(t), 'case':str(t), 'traceback':trace} for t,trace in result.errors],
                   skipped=[{'test':t.id(), 'reason':reason} for t,reason in result.skipped],
                   successful=result.wasSuccessful())
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
