"""CPU-only queue tests; launch only short local shell fixtures."""
import concurrent.futures
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'cluster' / 'fleet_queue.py'
spec = importlib.util.spec_from_file_location('fleet_queue', SOURCE)
fleet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fleet)


class FleetQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / 'project'
        self.root.mkdir()
        (self.root / 'launcher.sh').write_text('sleep 0.4\n')
        self.qdir = self.base / 'queue'
        (self.qdir / 'claims').mkdir(parents=True)
        self.active, self.last, self.log = {}, {}, io.StringIO()
        self.patches = [patch.object(fleet, 'LOCK_DIR', str(self.base / 'locks')),
                        patch.object(fleet, 'GRACE', 0),
                        patch.object(fleet, 'gpus_free', return_value={0: 0, 1: 0}),
                        patch.object(fleet.shutil, 'disk_usage',
                                     return_value=SimpleNamespace(free=100 * 1024**3))]
        for item in self.patches:
            item.start()
        self.addCleanup(self.tmp.cleanup)
        for item in self.patches:
            self.addCleanup(item.stop)
        self.addCleanup(self.wait_children)
        self.write_jobs(' | first')

    def wait_children(self):
        for child in self.active.values():
            if isinstance(child, subprocess.Popen):
                child.wait(timeout=5)

    def write_jobs(self, *lines):
        (self.qdir / 'jobs.txt').write_text('\n'.join(lines) + '\n')

    def dispatch(self, host='test', only=None, deadline=None, **changes):
        options = dict(qdir=str(self.qdir), root=str(self.root), launcher='launcher.sh',
                       host=host, only={0} if only is None else only, deadline=deadline,
                       python=sys.executable, active=self.active, last=self.last, log=self.log)
        options.update(changes)
        fleet.dispatch(**options)

    def claims(self):
        return list((self.qdir / 'claims').iterdir())

    def test_atomic_claim_across_competing_gpus(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as workers:
            list(workers.map(lambda gpu: self.dispatch(only={gpu}), [0, 1]))
        self.assertEqual(len(self.claims()), 1)
        self.assertEqual(len(self.active), 1)
        meta = json.loads((self.claims()[0] / 'claim.json').read_text())
        self.assertIn('wrapper_pid', meta)
        self.wait_children()
        self.assertEqual((self.claims()[0] / 'rc').read_text().strip(), '0')

    def test_target_filter_keeps_other_host_job_unclaimed(self):
        self.write_jobs('FLEET_TARGET=elsewhere | first', 'FLEET_TARGET=test | second')
        self.dispatch()
        self.assertEqual(len(self.claims()), 1)
        meta = json.loads((self.claims()[0] / 'claim.json').read_text())
        self.assertIn('second', meta['line'])

    def test_malformed_lines_do_not_starve_valid_jobs(self):
        self.write_jobs('missing separator', 'MISSING_EQUALS | first', ' | valid')
        self.dispatch()
        self.assertEqual(len(self.claims()), 1)
        self.assertIn('invalid job line 1', self.log.getvalue())
        self.assertIn('invalid job line 2', self.log.getvalue())

    def test_deadline_and_stop_do_not_claim(self):
        self.dispatch(deadline=time.time() - 1)
        self.assertEqual(self.claims(), [])
        (self.qdir / 'STOP_test').touch()
        self.dispatch()
        self.assertEqual(self.claims(), [])

    def test_deadline_checked_again_before_claim(self):
        with patch.object(fleet.time, 'time', side_effect=[100, 100, 201]):
            self.dispatch(deadline=200)
        self.assertEqual(self.claims(), [])

    def test_environment_drops_inherited_knobs_preserves_explicit_defaults(self):
        with patch.dict(os.environ, {'NGLAB_QKN': '1', 'NGLAB_PY': 'wrong',
                                     'NGLAB_ROOT': 'wrong', 'STUDENT_TEMP': '0.5'}):
            env = fleet.job_environment('NGLAB_LR=0.0003', str(self.root), sys.executable)
        self.assertNotIn('NGLAB_QKN', env)
        self.assertEqual(env['NGLAB_ROOT'], str(self.root))
        self.assertEqual(env['NGLAB_PY'], sys.executable)
        self.assertEqual(env['NGLAB_LR'], '0.0003')
        self.assertEqual(env['STUDENT_TEMP'], '0.5')
        env = fleet.job_environment('NGLAB_PY=/explicit/python', str(self.root), sys.executable)
        self.assertEqual(env['NGLAB_PY'], '/explicit/python')

    def test_active_child_prevents_second_dispatch_even_if_memory_is_zero(self):
        self.write_jobs(' | first', ' | second')
        self.dispatch()
        self.dispatch()
        self.assertEqual(len(self.claims()), 1)

    def test_wrapper_lock_survives_agent_restart_and_cross_root_queue(self):
        self.dispatch()
        other_root = self.base / 'student'
        other_root.mkdir()
        (other_root / 'launcher.sh').write_text('exit 0\n')
        other_queue = self.base / 'student_queue'
        (other_queue / 'claims').mkdir(parents=True)
        (other_queue / 'jobs.txt').write_text(' | other\n')
        self.dispatch(qdir=str(other_queue), root=str(other_root), active={}, last={})
        self.assertEqual(list((other_queue / 'claims').iterdir()), [])
        self.wait_children()
        lock = fleet.lock_file('gpu0.lock')
        self.assertIsNotNone(lock)
        os.close(lock)

    def test_fresh_gpu_check_blocks_new_occupant(self):
        with patch.object(fleet, 'gpus_free', side_effect=[{0: 0}, {0: 2000}]):
            self.dispatch()
        self.assertEqual(self.claims(), [])

    def test_disk_gate_and_spawn_failure_are_not_retried(self):
        with patch.object(fleet.shutil, 'disk_usage',
                          return_value=SimpleNamespace(free=19 * 1024**3)):
            self.dispatch()
        self.assertEqual(self.claims(), [])
        with patch.object(fleet.subprocess, 'Popen', side_effect=OSError('fixture failure')):
            self.dispatch()
        claim = self.claims()[0]
        self.assertEqual((claim / 'rc').read_text().strip(), '127')
        self.assertIn('fixture failure', json.loads((claim / 'claim.json').read_text())['spawn_error'])
        with patch.object(fleet.subprocess, 'Popen') as popen:
            self.dispatch()
            popen.assert_not_called()

    def test_same_queue_host_agent_is_exclusive(self):
        key = hashlib.sha256((os.path.realpath(self.qdir) + 'test').encode()).hexdigest()[:16]
        fd = fleet.lock_file(f'agent_{key}.lock')
        try:
            with patch.dict(os.environ, {'FLEET_HOST': 'test'}), patch.object(
                    sys, 'argv', ['fleet_queue.py', str(self.qdir), str(self.root), 'launcher.sh']):
                with self.assertRaises(SystemExit) as result:
                    fleet.main()
                self.assertEqual(result.exception.code, 1)
        finally:
            os.close(fd)


if __name__ == '__main__':
    unittest.main()
