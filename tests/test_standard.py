import support  # noqa: F401

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chat_benchmark.history import history
from chat_benchmark.reports import standardize_report
from chat_benchmark.standard import STANDARD_ID, reference_tokens, standard_metrics, tokenizer
from chat_benchmark.workload import canonical_text, reference_digest, text_digest


class StandardTests(unittest.TestCase):
    def sample(self, provider='codex', duration=10000, tokens=867):
        return dict(provider=provider, request_id='r1', reference_rows=64, reference_matches=True,
                    payload_sha256=reference_digest(64), continuity='single_file_payload',
                    duration_ms=duration, output_tokens=tokens, reasoning_tokens=0,
                    status='completed', phase='measured', effective_tps=tokens*1000/duration)

    def test_one_offline_unit_is_independent_of_provider_and_native_usage(self):
        tokenizer.cache_clear()
        with patch('socket.socket', side_effect=AssertionError('network used')):
            a = standard_metrics(self.sample('codex', 10000, 867))
            b = standard_metrics(self.sample('claude', 10000, 1594))
        self.assertEqual(a, b)
        self.assertEqual(a['standard_tokenizer'], STANDARD_ID)
        self.assertEqual(a['standard_tokens'], 767)
        self.assertAlmostEqual(a['standard_tps'], 76.7)

    def test_double_time_halves_standard_rate_despite_native_tokens(self):
        a = standard_metrics(self.sample('codex', 10000, 867))
        b = standard_metrics(self.sample('claude', 20000, 1594))
        self.assertEqual(a['standard_tps'], b['standard_tps'] * 2)

    def test_corrupted_vocabulary_is_rejected(self):
        tokenizer.cache_clear()
        try:
            with patch('pathlib.Path.read_bytes', return_value=b'corrupt'):
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    tokenizer()
        finally:
            tokenizer.cache_clear()

    def test_unverified_interrupted_and_invalid_samples_never_get_standard_rate(self):
        for changes in [dict(payload_sha256='wrong'), dict(reference_matches=False),
                        dict(continuity='interrupted'), dict(status='failed'),
                        dict(reference_rows=None), dict(reference_rows=64.0),dict(reference_rows=1000000)]:
            sample = {**self.sample(), **changes}
            self.assertIsNone(standard_metrics(sample)['standard_tokens'])
        for duration in [0,-1,None,float('inf'),float('nan')]:
            self.assertIsNone(standard_metrics({**self.sample(), 'duration_ms':duration})['standard_tps'])

    def test_normalized_reference_count_and_fixed_lengths(self):
        self.assertEqual([reference_tokens(n) for n in (32,64,192,384)], [383,767,2303,4607])
        sample = self.sample()
        sample['payload_sha256'] = text_digest('\r\n'+canonical_text(64).replace('\n','\r\n')+'\r\n')
        self.assertEqual(standard_metrics(sample)['standard_tokens'],767)

    def test_retroactive_report_preserves_native_usage_and_accounting_excludes_drift_and_warmup(self):
        a, b = self.sample(), {**self.sample('claude'), 'request_id':'r2'}
        report = dict(samples=[a,b], warmups=[{**a,'phase':'warmup'}],
                      summary={'settings_drift':[{'request_id':'r2'}]},
                      accounting={'tokens':{'output_tokens':9999},'client_estimated_usd':.123})
        before = copy.deepcopy(report)
        standardize_report(report)
        self.assertEqual(report['accounting'],before['accounting'])
        self.assertEqual([s['output_tokens'] for s in report['samples']],[867,867])
        self.assertEqual(report['summary']['standard_tps']['n'],1)
        self.assertEqual(report['summary']['weighted_standard_tps'],76.7)
        snapshot = copy.deepcopy(report)
        self.assertEqual(standardize_report(report),snapshot)

    def test_history_upgrades_verified_snapshots_and_ignores_stale_standard_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root/'results'/'old'
            directory.mkdir(parents=True)
            sample = {**self.sample(),'standard_tokens':999999,'standard_tps':999999}
            report = dict(run_id='run1',workload='fixed-temp-file-v1',samples=[sample],created_at='2026-10-30')
            (directory/'benchmark.json').write_text(json.dumps(report))
            groups = history(root)
        self.assertEqual(groups[0]['standard_tps']['median'],76.7)
        self.assertEqual(groups[0]['effective_tps']['median'],86.7)


if __name__ == '__main__':
    unittest.main()
