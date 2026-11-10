import support  # noqa: F401

import unittest

from chat_benchmark.reports import chat_table
from chat_benchmark.workload import reference_digest
from chat_benchmark.standard import reference_tokens


class ChatFormatTests(unittest.TestCase):
    def report(self):
        report = {'status':'complete', 'plan':{'durations_seconds':[10,30,60],'warmups':1},
                'summary':{'settings_drift':[]}, 'samples':[
                    {'sample_index':1,'phase':'warmup','duration_ms':1000,'effective_tps':9999,'ttft_ms':1},
                    {'sample_index':2,'request_id':'r2','reference_rows':64,'duration_ms':10720,
                     'effective_tps':148.33,'ttft_ms':862,'status':'completed','reference_matches':True},
                    {'sample_index':3,'request_id':'r3','reference_rows':192,'duration_ms':28320,
                     'effective_tps':155.57,'ttft_ms':896,'status':'completed','reference_matches':True},
                    {'sample_index':4,'request_id':'r4','reference_rows':384,'duration_ms':54680,
                     'effective_tps':157.82,'ttft_ms':828,'status':'completed','reference_matches':True}]}
        for sample in report['samples'][1:]:
            sample.update(continuity='single_file_payload', payload_sha256=reference_digest(sample['reference_rows']))
        return report

    def test_seconds_milliseconds_and_provider_counts_are_formatted_without_warmup(self):
        table=chat_table(self.report())
        self.assertIn('| Stage | Rows | Request time | Standard TPS | Native TTFT |',table)
        for rows, target, seconds, ttft in [(64,10,10.72,862),(192,30,28.32,896),(384,60,54.68,828)]:
            self.assertIn(f'| {target} s | {rows} | {seconds:.2f} s | {reference_tokens(rows)/seconds:,.2f} | {ttft} ms |',table)
        self.assertNotIn('9999',table)

    def test_unrun_or_unexposed_values_remain_missing(self):
        report=self.report()
        report['samples']=report['samples'][:2]
        report['samples'][1]['ttft_ms']=None
        report['status']='wall_time_cap'
        table=chat_table(report)
        self.assertIn(f'| 10 s | 64 | 10.72 s | {reference_tokens(64)/10.72:,.2f} | N/A |',table)
        self.assertIn('| 30 s | 192 | N/A | N/A | N/A |',table)
        self.assertIn('Run status: wall time cap.',table)

    def test_settings_mismatch_cannot_look_like_valid_measurement(self):
        report=self.report()
        report['summary']['settings_drift']=[{'request_id':'r3'}]
        table=chat_table(report)
        self.assertIn('| 30 s | 192 | N/A | N/A | N/A |',table)
        self.assertIn('Excluded invalid or mismatched samples: 30 s.',table)

    def test_token_numerator_and_common_workload_rate_are_exposed(self):
        report=self.report()
        report['samples'][1]['output_tokens']=1589
        table=chat_table(report)
        self.assertIn('| Stage | Standard tokens | Native output tokens | Native TPS | Fixed-text rows/s |',table)
        self.assertIn(f'| 10 s | {reference_tokens(64):,} | 1,589 | 148.33 | 5.97 |',table)
        report['summary']['settings_drift']=[{'request_id':'r2'}]
        self.assertIn('| 10 s | N/A | N/A | N/A | N/A |',chat_table(report))


if __name__=='__main__': unittest.main()
