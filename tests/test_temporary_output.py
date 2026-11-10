import support  # noqa: F401

import argparse
import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from chat_benchmark.cli import capture_file, load_run, save_run
from chat_benchmark.protocol import plan
from chat_benchmark.temporary import cleanup, recover_stale, sample_file
from chat_benchmark.workload import canonical_text, reference_digest


class TemporaryOutputTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.root = Path(self.workspace.name)
        self.files = self.root / 'sample-temp'
        self.run_id = 'f' * 32

    def tearDown(self):
        self.workspace.cleanup()

    def test_file_exists_only_inside_its_scope(self):
        with sample_file(self.run_id, canonical_text(13), self.files) as path:
            directory = path.parent
            self.assertEqual(path.read_text(), canonical_text(13))
            self.assertEqual(path.name, 'sample.txt')
        self.assertFalse(directory.exists())

    def test_exception_removes_file_and_directory(self):
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            with sample_file(self.run_id, canonical_text(13), self.files) as path:
                directory = path.parent
                raise RuntimeError('failed')
        self.assertFalse(directory.exists())

    def test_cleanup_refuses_external_path_and_keeps_unknown_files(self):
        directory = self.root / (self.run_id + '-outside')
        directory.mkdir()
        (directory/'owner.json').write_text(json.dumps({'run_id':self.run_id}))
        (directory/'sample.txt').write_text('keep')
        with self.assertRaises(ValueError):
            cleanup(directory, self.run_id, self.files)
        self.assertEqual((directory/'sample.txt').read_text(), 'keep')
        self.files.mkdir()
        owned = self.files / (self.run_id+'-owned')
        owned.mkdir()
        (owned/'owner.json').write_text(json.dumps({'run_id':self.run_id}))
        (owned/'sample.txt').write_text('sample')
        (owned/'unrelated.txt').write_text('keep')
        with self.assertRaises(ValueError):
            cleanup(owned,self.run_id,self.files)
        self.assertTrue((owned/'sample.txt').exists())
        self.assertTrue((owned/'unrelated.txt').exists())

    def test_recovers_only_stale_owned_directories(self):
        self.files.mkdir()
        old = self.files / (self.run_id+'-old')
        old.mkdir()
        (old/'owner.json').write_text(json.dumps({'run_id':self.run_id}))
        (old/'sample.txt').write_text('discard')
        os.utime(old,(time.time()-7200,time.time()-7200))
        fresh = self.files / (self.run_id+'-fresh')
        fresh.mkdir()
        (fresh/'owner.json').write_text(json.dumps({'run_id':self.run_id}))
        self.assertEqual(recover_stale(self.files),1)
        self.assertFalse(old.exists())
        self.assertTrue(fresh.exists())

    def fixture(self, calls=1, literal=True):
        transcript=self.root/'session.jsonl'
        # Real clients record the whole command, so the generated rows are in the tool input.
        body=canonical_text(13) if literal else "python -c \"print('0001 amber birch cedar delta ember frost grove harbor')\""

        rows=[{'type':'turn_context','payload':{'turn_id':'t1','model':'model','effort':'high','service_tier':'default'}},
              {'timestamp':'2026-10-30T10:00:00Z','type':'event_msg','payload':{'type':'task_started','turn_id':'t1'}},
              {'timestamp':'2026-10-30T10:00:02Z','type':'response_item','payload':{'type':'custom_tool_call','input':f"@'\n{body}\n'@ | write-sample --run-id {self.run_id} --index 1"}}]
        if calls>1:
            rows.append({'type':'response_item','payload':{'type':'custom_tool_call','input':'some other tool'}})
        rows.append({'timestamp':'2026-10-30T10:00:02Z','type':'token_usage_record','payload':{
            'response_id':'r1','turn_id':'t1','usage':{'input_tokens':200,'cached_input_tokens':100,
            'cache_write_input_tokens':0,'output_tokens':100,'reasoning_output_tokens':0}}})
        transcript.write_text('\n'.join(map(json.dumps,rows)))
        run=dict(run_id=self.run_id,started_ms=time.time()*1000,created_at='2026-10-30T10:00:00Z',
                 plan=plan(durations=[2],warmup_seconds=0,max_seconds=45),samples={},samples_expected=1,
                 settings=dict(model='model',effort='high',service_tier='default'),
                 workload='fixed-temp-file-v1',accounting_start_request_id='r1',
                 session=dict(provider='codex',session_id='session-123',path=str(transcript),home=str(self.root)),
                 results_directory=str(self.root/'results'/'run'))
        save_run(self.root,run)
        return argparse.Namespace(command='write-sample',run_id=self.run_id,index=1,text=canonical_text(13))

    def record(self,args):
        output=io.StringIO()
        with patch('chat_benchmark.temporary.temp_root',return_value=self.files), \
             patch('chat_benchmark.cli.ensure_collector',return_value={}), contextlib.redirect_stdout(output):
            capture_file(args,self.root)
        return json.loads(output.getvalue())

    def test_full_file_workflow_records_usage_and_deletes_without_echo(self):
        result=self.record(self.fixture())
        self.assertTrue(result['temporary_files_deleted'])
        self.assertEqual(result['progress']['reason'],'stages_complete')
        self.assertEqual(list(self.files.iterdir()),[])
        report=json.loads((self.root/'results/run/benchmark.json').read_text())
        sample=report['samples'][0]
        self.assertEqual(sample['payload_sha256'],reference_digest(13))
        self.assertEqual(sample['continuity'],'single_file_payload')
        self.assertEqual(sample['payload_words'],117)
        self.assertEqual(sample['temporary_cleanup'],'deleted')
        self.assertIsNone(sample['text_span_ms'])
        self.assertEqual(sample['effective_tps'],50) # Request latency, not disk-writing time.
        self.assertEqual(sample['standard_tokens'],155)
        self.assertEqual(sample['standard_tps'],77.5)
        self.assertNotIn(canonical_text(13),json.dumps(report))
        self.assertNotIn('0001 amber',json.dumps(result))

    def test_reference_mismatch_is_cleaned_and_rejected(self):
        args=self.fixture();args.text=args.text.replace('amber','changed',1)
        result=self.record(args)
        self.assertEqual(result['progress']['reason'],'reference_mismatch')
        self.assertTrue(result['temporary_files_deleted'])
        self.assertEqual(list(self.files.iterdir()),[])

    def test_scripted_rows_are_rejected_as_synthesized(self):
        result=self.record(self.fixture(literal=False))
        self.assertEqual(result['progress']['reason'],'synthesized_output')
        report=json.loads((self.root/'results/run/benchmark.json').read_text())
        self.assertEqual(report['samples'][0]['status'],'invalid_output')
        self.assertIsNone(report['samples'][0]['standard_tps'])
        self.assertIsNone(report['summary']['generation']['tps'])

    def test_multiple_tool_calls_are_rejected(self):
        result=self.record(self.fixture(calls=2))
        self.assertEqual(result['progress']['reason'],'interrupted_output')

    def test_identical_retry_does_not_add_a_sample_or_leave_file(self):
        args=self.fixture();self.record(args)
        result=self.record(args)
        self.assertTrue(result['already_recorded'])
        self.assertEqual(len(load_run(self.root,self.run_id)['samples']),1)
        self.assertEqual(list(self.files.iterdir()),[])

    def test_recording_failure_cleans_file(self):
        args=self.fixture()
        with patch('chat_benchmark.temporary.temp_root',return_value=self.files), \
             patch('chat_benchmark.cli.capture_sample',side_effect=RuntimeError('capture failed')):
            with self.assertRaises(RuntimeError):
                capture_file(args,self.root)
        self.assertEqual(list(self.files.iterdir()),[])

    def test_cleanup_failure_does_not_report_success(self):
        args=self.fixture()
        with patch('chat_benchmark.temporary.temp_root',return_value=self.files), \
             patch('chat_benchmark.cli.ensure_collector',return_value={}), \
             patch('chat_benchmark.temporary.cleanup',side_effect=OSError('file locked')):
            with self.assertRaisesRegex(OSError,'file locked'):
                capture_file(args,self.root)
        self.assertEqual(load_run(self.root,self.run_id)['samples']['1']['temporary_cleanup'],'failed')


if __name__=='__main__':unittest.main()
