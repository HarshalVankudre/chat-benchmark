import support  # noqa: F401

import json
import tempfile
import unittest
from pathlib import Path

from chat_benchmark.accounting import account
from chat_benchmark.cli import checkpoint
from chat_benchmark.history import history
from chat_benchmark.metrics import Request, stats
from chat_benchmark.protocol import plan, progress, stage_target
from chat_benchmark.reports import build_report
from chat_benchmark.sessions import Session, parse_codex, parse_claude
from chat_benchmark.telemetry import normalize, enrich
from chat_benchmark.workload import canonical_text, reference_digest, rows_for, text_digest


def sample(index=1, phase='measured', seconds=10):
    return Request('codex', 'session-123', f'r{index}', model='model', effort='high', service_tier='default',
                   duration_ms=seconds * 1000, output_tokens=1000, reasoning_tokens=0,
                   input_tokens=10000, cached_input_tokens=9000, cache_write_input_tokens=0,
                   visible_words=rows_for(seconds)*9, text_span_ms=seconds*1000,
                   text_block_count=1, continuity='single_text_block', sample_index=index,
                   reference_matches=True, reference_rows=rows_for(seconds), phase=phase,
                   target_seconds=seconds, timing_source='native fixture', client_version='test')


class SweepTests(unittest.TestCase):
    def run_data(self, p=None):
        return dict(run_id='a'*32, started_ms=1000, plan=p or plan(), samples={},
                    settings=sample().settings(), samples_expected=3, workload='fixed-continuous-text-v1')

    def test_default_is_short_and_fixed_across_model_speeds(self):
        run = self.run_data()
        self.assertEqual(run['plan']['durations_seconds'], [10,30,60])
        self.assertEqual(run['plan']['max_wall_seconds'], 180)
        self.assertEqual(progress(run, [], 1000)['target_rows'], 32)
        r = sample(1, 'warmup', 5)
        run['samples']['1'] = {}
        a = progress(run, [r], 5000)
        r.text_span_ms *= 2
        b = progress(run, [r], 5000)
        self.assertEqual(a['target_rows'], b['target_rows'])
        self.assertEqual(a['target_rows'], 64)
        self.assertGreater(b['predicted_seconds'], a['predicted_seconds'])

    def test_budget_stops_between_bursts_without_shrinking_reference(self):
        run = self.run_data()
        run['samples']['1'] = {}
        r = sample(1, 'warmup', 5)
        state = progress(run, [r], 179000)
        self.assertEqual(state['action'], 'stop')
        self.assertEqual(state['reason'], 'remaining_budget')
        self.assertIsNone(state['target_rows'])

    def test_warmup_excluded_and_all_stages_finish(self):
        run = self.run_data()
        requests = [sample(1,'warmup',5)] + [sample(i+2,seconds=t) for i,t in enumerate([10,30,60])]
        run['samples'] = {str(r.sample_index): {} for r in requests}
        state = progress(run, requests, 110000)
        self.assertEqual(state['reason'], 'stages_complete')
        report = build_report({'provider':'codex','session_id':'session-123'},requests,run['settings'],run=run,progress=state)
        self.assertEqual(report['summary']['output_tokens'],3000)
        self.assertEqual(len(report['warmups']),1)
        self.assertIsNone(report['summary']['effective_tps']['median_ci95'])
        self.assertEqual(report['quality']['effective_tps']['status'],'provisional')

    def test_stops_for_interruption_reference_drift_and_settings(self):
        for field,value,reason in [('continuity','interrupted','interrupted_output'),
                                  ('reference_matches',False,'reference_mismatch'),
                                  ('effort','max','settings_changed')]:
            run=self.run_data(); r=sample(1,'warmup',5); setattr(r,field,value)
            run['samples']['1']={}
            self.assertEqual(progress(run,[r],5000)['reason'],reason)

    def test_missing_usage_has_bounded_wait(self):
        run=self.run_data();run['samples']['1']={'captured_ms':1000}
        self.assertEqual(progress(run,[],2000)['action'],'wait_for_usage')
        self.assertEqual(progress(run,[],18000)['reason'],'missing_usage')

    def test_reference_hash_is_identical_and_rejects_edits(self):
        text=canonical_text(64)
        self.assertEqual(reference_digest(64),text_digest(text.replace('\n','\r\n')+'\r\n'))
        self.assertNotEqual(reference_digest(64),text_digest(text.replace('amber','other',1)))
        self.assertTrue(canonical_text(192).startswith(text+'\n'))

    def test_native_message_span_and_thinking_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'s.jsonl'
            rows=[{'type':'event_msg','payload':{'type':'item_completed','item':{'type':'AgentMessage','id':'m1'},'started_at_ms':1000,'completed_at_ms':3000}},
                  {'type':'response_item','payload':{'type':'message','role':'assistant','id':'m1','content':[{'text':canonical_text(13)}]}},
                  {'type':'response_item','payload':{'type':'custom_tool_call','input':'burst --run-id '+'a'*32+' --index 1'}},
                  {'type':'token_usage_record','timestamp':'2026-10-30T10:00:00Z','payload':{'response_id':'r1','usage':{'output_tokens':100}}}]
            p.write_text('\n'.join(map(json.dumps,rows)))
            req=parse_codex(Session('codex','session-123',p,Path(directory)))[0][0]
            self.assertEqual(req.text_span_ms,2000)
            self.assertEqual(req.continuity,'single_text_block')
            self.assertEqual(req.run_id,'a'*32)
            self.assertEqual(req.visible_sha256,reference_digest(13))
            rows.insert(1,{'type':'event_msg','payload':{'type':'item_completed','item':{'type':'Reasoning','id':'think'},'started_at_ms':2000,'completed_at_ms':2500}})
            p.write_text('\n'.join(map(json.dumps,rows)))
            self.assertEqual(parse_codex(Session('codex','session-123',p,Path(directory)))[0][0].continuity,'interrupted')

    def test_full_accounting_counts_overhead_without_guessing_charge(self):
        requests=[sample(i) for i in range(4)]
        requests[1].phase='warmup'
        result=account(requests,'r1')
        self.assertEqual(result['tokens']['output_tokens'],3000)
        self.assertEqual(result['requests'],3)
        self.assertIsNone(result['actual_charged_usd'])
        self.assertIsNone(result['client_estimated_usd'])

    def test_same_reference_different_provider_tokens_can_mean_twice_the_time(self):
        claude=sample(seconds=60);claude.output_tokens=8634;claude.duration_ms=54623;claude.payload_bytes=20736
        codex=sample(seconds=60);codex.output_tokens=4707;codex.duration_ms=26951.4658203125;codex.payload_bytes=20736
        self.assertAlmostEqual(claude.effective_tps,158.06528,places=4)
        self.assertAlmostEqual(codex.effective_tps,174.64727,places=4)
        self.assertAlmostEqual(claude.reference_rows_per_second,7.03001,places=4)
        self.assertAlmostEqual(codex.reference_rows_per_second,14.24783,places=4)
        self.assertAlmostEqual(codex.payload_bytes_per_second/claude.payload_bytes_per_second,
                               claude.duration_ms/codex.duration_ms)
        claude.reference_matches=False
        self.assertIsNone(claude.reference_rows_per_second)
        self.assertIsNone(claude.payload_bytes_per_second)

    def test_history_separates_duration_and_model_and_excludes_bad_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for i in range(12):
                r=sample(i,seconds=10 if i<10 else 30)
                if i==11:r.reference_matches=False
                report={'run_id':str(i),'workload':'fixed-continuous-text-v1','created_at':str(i).zfill(3),'samples':[r.to_dict()]}
                p=root/'results'/str(i);p.mkdir(parents=True);(p/'benchmark.json').write_text(json.dumps(report))
            groups=history(root)
            self.assertEqual(len(groups),2)
            ten=next(g for g in groups if g['settings']['target_seconds']==10)
            self.assertEqual(ten['distinct_runs'],10)
            self.assertIsNotNone(ten['effective_tps']['median_ci95'])
            thirty=next(g for g in groups if g['settings']['target_seconds']==30)
            self.assertEqual(thirty['distinct_runs'],1)

    def test_claude_native_cost_is_estimate_and_scoped_by_request(self):
        payload={'resourceLogs':[{'resource':{'attributes':[{'key':'service.name','value':{'stringValue':'claude-code'}}]},
                                 'scopeLogs':[{'logRecords':[{'timeUnixNano':'1','attributes':[
                                     {'key':'session.id','value':{'stringValue':'session-123'}},
                                     {'key':'event.name','value':{'stringValue':'api_request'}},
                                     {'key':'request_id','value':{'stringValue':'r1'}},
                                     {'key':'cost_usd','value':{'doubleValue':0.123}}]}]}]}]}
        events=normalize(payload,'logs')
        r=sample();enrich([r],events)
        self.assertEqual(r.client_estimated_usd,0.123)
        self.assertEqual(account([r],'r1')['client_estimated_usd'],0.123)


if __name__=='__main__':unittest.main()
