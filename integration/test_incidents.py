"""The same seven regressions run against original/fixed code and FastAPI.

Set HW8910_PROJECT to the target checkout. No project imports in this file.
All subprocesses, workspaces and loopback HTTP connections are real.
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = Path(os.environ.get('HW8910_PROJECT', Path(__file__).resolve().parents[1])).resolve()
RUNNER = os.environ.get('HW8910_RUNNER', 'lab.py')
SERVICE = os.environ.get('HW8910_SERVICE', 'service.py')


class IncidentIntegrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='hw8910-test-')
        self.addCleanup(temp.cleanup)
        self.workspace = Path(temp.name) / 'lab'
        self.command('init', expected=0)

    def command(self, *args, expected=None):
        result = subprocess.run([sys.executable, str(PROJECT/RUNNER), '--workspace', str(self.workspace), *args],
                                cwd=PROJECT, capture_output=True, text=True, timeout=180)
        if expected is not None:
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def read(self, name):
        return json.loads((self.workspace/name).read_text())

    def pointer(self):
        return hashlib.sha256((self.workspace/'accepted.json').read_bytes()).hexdigest()

    def http(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.url + path, data=data,
            headers={'Content-Type': 'application/json'} if data else {})
        try:
            response = urllib.request.urlopen(request, timeout=10)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            return response.status, json.load(response)

    def start_server(self, warmup=0):
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        args = [sys.executable, str(PROJECT/SERVICE), '--workspace', str(self.workspace),
                '--port', str(port), '--warmup-seconds', str(warmup)]
        if SERVICE == 'api.py':
            args.extend(['--backend', 'stub'])
        output = tempfile.TemporaryFile(mode='w+')
        self.addCleanup(output.close)
        process = subprocess.Popen(args, cwd=PROJECT, stdout=output, stderr=subprocess.STDOUT)
        self.addCleanup(self.stop_server, process)
        self.url = f'http://127.0.0.1:{port}'
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output.seek(0)
                self.fail('HTTP process exited: ' + output.read())
            try:
                if self.http('/health')[0] == 200:
                    return process
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(.03)
        self.fail('Server did not expose /health')

    @staticmethod
    def stop_server(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def wait_ready(self):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.http('/ready')[0] == 200:
                return
            time.sleep(.03)
        self.fail('Model never became ready')

    def test_cache_survives_restart_but_changed_input_invalidates(self):
        before = self.pointer()
        failed = self.command('run', '--fail-after-prepare')
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn('PREPARED', failed.stdout)
        self.assertEqual(self.pointer(), before)
        recovered = self.command('run', expected=0)
        self.assertIn('CACHE_HIT', recovered.stdout)
        with (self.workspace/'input.txt').open('a') as stream:
            stream.write('Третий пример\n')
        changed = self.command('run', expected=0)
        self.assertIn('PREPARED', changed.stdout, 'Changed bytes incorrectly reused old cache')

    def test_incomplete_and_foreign_reports_cannot_replace_pointer(self):
        before = (self.workspace/'accepted.json').read_bytes()
        for report in ('report-incomplete.json', 'report-A.json'):
            with self.subTest(report=report):
                (self.workspace/'accepted.json').write_bytes(before)
                result = self.command('run', '--report', report)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual((self.workspace/'accepted.json').read_bytes(), before)

    def test_repeat_in_separate_processes_has_one_publication(self):
        self.command('run', expected=0)
        first = (self.workspace/'accepted.json').read_bytes()
        self.command('run', expected=0)
        events = [json.loads(line) for line in (self.workspace/'publications.jsonl').read_text().splitlines()]
        self.assertEqual(len([e for e in events if e['model_id'] == 'fixture-B']), 1)
        self.assertEqual((self.workspace/'accepted.json').read_bytes(), first)

    def test_reject_and_atomic_replace_error_preserve_bytes(self):
        before = self.pointer()
        result = self.command('run', '--report', 'report-reject.json')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.pointer(), before)
        script = '''
import json, pathlib, sys
from unittest.mock import patch
import release
root=pathlib.Path(sys.argv[1]); target=root/'accepted.json'; before=target.read_bytes()
m=json.loads((root/'manifest-B.json').read_text()); r=json.loads((root/'report-B.json').read_text())
with patch('os.replace', side_effect=OSError('simulated replacement error')):
    try: release.publish(m,r,target)
    except OSError: pass
    else: raise AssertionError('Atomic replace was not used')
assert target.read_bytes()==before, 'Previous pointer was truncated'
assert not list(root.glob('*.tmp')), 'Temporary file leaked'
print('ATOMIC_ERROR_PRESERVED')
'''
        result = subprocess.run([sys.executable, '-c', script, str(self.workspace)], cwd=PROJECT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('ATOMIC_ERROR_PRESERVED', result.stdout)

    def test_http_ready_waits_for_warmup(self):
        self.start_server(warmup=2)
        self.assertEqual(self.http('/health')[0], 200)
        self.assertEqual(self.http('/ready')[0], 503)
        self.assertEqual(self.http('/generate', {'prompt': 'test'})[0], 503)
        self.assertFalse(self.http('/debug')[1]['warmup_finished'])
        self.wait_ready()
        self.assertTrue(self.http('/debug')[1]['warmup_finished'])
        self.assertEqual(self.http('/generate', {'prompt': 'test'})[0], 200)

    def test_http_loaded_version_changes_only_after_restart_and_rollback(self):
        accepted_a = (self.workspace/'accepted.json').read_bytes()
        server = self.start_server()
        self.wait_ready()
        self.command('run', expected=0)
        status, response = self.http('/generate', {'prompt': 'test'})
        self.assertEqual(status, 200)
        self.assertEqual(response['model_id'], 'fixture-A')
        self.assertIn('fixture-A', response['answer'])
        self.stop_server(server)
        server = self.start_server()
        self.wait_ready()
        status, response = self.http('/generate', {'prompt': 'test'})
        self.assertEqual(status, 200)
        self.assertEqual(response['model_id'], 'fixture-B')
        self.assertIn('fixture-B', response['answer'])
        # Both versions have this publish API: no new route/flag required on the starter.
        script = "import json,pathlib,release,sys; p=pathlib.Path(sys.argv[1]); release.publish(json.loads((p/'manifest-A.json').read_text()),json.loads((p/'report-A.json').read_text()),p/'accepted.json')"
        done = subprocess.run([sys.executable, '-c', script, str(self.workspace)], cwd=PROJECT,
                              capture_output=True, text=True, timeout=20)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads((self.workspace/'accepted.json').read_bytes()), json.loads(accepted_a))
        self.stop_server(server)
        self.start_server()
        self.wait_ready()
        self.assertEqual(self.http('/generate', {'prompt': 'test'})[1]['model_id'], 'fixture-A')

    def test_http_invalid_requests_never_call_backend(self):
        self.start_server()
        self.wait_ready()
        for payload in ({'prompt': ''}, {'prompt': 'test', 'max_new_tokens': 0},
                        {'prompt': 'test', 'max_new_tokens': 100000},
                        {'prompt': 'test', 'max_new_tokens': '32'},
                        {'prompt': 'test', 'max_new_tokens': True}, ['not', 'an', 'object']):
            with self.subTest(payload=payload):
                before = self.http('/debug')[1]['calls']
                self.assertTrue(400 <= self.http('/generate', payload)[0] < 500)
                self.assertEqual(self.http('/debug')[1]['calls'], before)
        self.assertEqual(self.http('/generate', {'prompt': 'test', 'max_new_tokens': 8})[0], 200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
