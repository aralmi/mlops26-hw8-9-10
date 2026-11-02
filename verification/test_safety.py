import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import journal
import lab
import release


class AdditionalSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'lab'
        lab.initialize(self.root)
        self.manifest = journal.read(self.root/'manifest-B.json')
        self.report = journal.read(self.root/'report-B.json')

    def test_crash_between_pointer_and_journal_recovers_once(self):
        with self.assertRaises(RuntimeError):
            journal.commit(self.root, self.manifest, self.report, fail_after_pointer=True)
        self.assertEqual(journal.read(self.root/'accepted.json'), self.manifest)
        self.assertTrue((self.root/'release-intent.json').exists())
        journal.commit(self.root, self.manifest, self.report)
        journal.commit(self.root, self.manifest, self.report)
        events = journal.read(self.root/'publication-events.json')
        self.assertEqual(sum(e['model_id']=='fixture-B' for e in events.values()), 1)
        self.assertFalse((self.root/'release-intent.json').exists())

    def test_managed_rollback_uses_previously_accepted_manifest(self):
        accepted_a = (self.root/'accepted.json').read_bytes()
        a_id = next(iter(journal.read(self.root/'publication-events.json')))
        journal.commit(self.root, self.manifest, self.report)
        journal.rollback(self.root, a_id)
        self.assertEqual((self.root/'accepted.json').read_bytes(), accepted_a)
        events = journal.read(self.root/'publication-events.json')
        self.assertEqual(sum(e['action']=='rollback' for e in events.values()), 1)
        journal.rollback(self.root, a_id)
        self.assertEqual(journal.read(self.root/'publication-events.json'), events)

    def test_real_file_mutation_rejects_before_publication(self):
        before = (self.root/'accepted.json').read_bytes()
        (self.root/'model-B.txt').write_text('changed after eval')
        with self.assertRaises(ValueError):
            journal.commit(self.root, self.manifest, self.report)
        self.assertEqual((self.root/'accepted.json').read_bytes(), before)

    def test_release_identity_includes_runtime_and_decision(self):
        first = journal.identity(self.manifest, self.report)
        other = copy.deepcopy(self.manifest)
        other['runtime']['generation']['max_new_tokens'] = 16
        other['runtime_sha256'] = release.digest(release.canonical(other['runtime']))
        report = {**self.report, 'runtime_sha256': other['runtime_sha256']}
        self.assertNotEqual(first, journal.identity(other, report))
        self.assertNotEqual(first, journal.identity(self.manifest, {**self.report, 'evaluation_run_id':'other-eval'}))

    def test_ready_failure_if_artifact_is_missing(self):
        from api_core import ServiceState
        (self.root/'model-A.txt').unlink()
        state = ServiceState(self.root)
        state.thread.join(timeout=5)
        self.assertFalse(state.ready)
        self.assertIsNotNone(state.error)

    def test_refused_release_leaves_history_and_pointer_unchanged(self):
        before = (self.root/'accepted.json').read_bytes()
        events = (self.root/'publication-events.json').read_bytes()
        with self.assertRaises(ValueError):
            journal.commit(self.root, self.manifest, {**self.report, 'passed':False})
        self.assertEqual((self.root/'accepted.json').read_bytes(), before)
        self.assertEqual((self.root/'publication-events.json').read_bytes(), events)


if __name__ == '__main__':
    unittest.main(verbosity=2)
