"""Read-only software checks of frozen experiment code; no model or network.

The F4/F6/F8 expected flags come from the existing hand-specified fixtures.
F10 expected states below are specified before executing either harness.
Scripted report dictionaries are software fixtures, never experiment outputs.
"""
from pathlib import Path
import sys, json, copy, hashlib, socket, contextlib
from types import SimpleNamespace
from unittest.mock import patch
from collections import defaultdict

sys.dont_write_bytecode = True
W = Path(__file__).resolve().parent
ROOT = W.parent
DATASET = sys.argv[1]
CONTEXT = ROOT
sys.path.insert(0, str(ROOT))

from tests.detector_cases import CASES, fixture, action
from experiments.evaluation.runtime_verifier import verify_runtime_report
from experiments.evaluation.deterministic_evaluator import evaluate_report, evaluation_result_to_dict
from experiments.runners.run_smoke_test import run_api_method
from experiments.harness.m4_patched import M4PatchedHarness
import pandas as pd

def deny(*args, **kwargs):
    raise AssertionError('Network/API construction/credential access forbidden')

class ScriptedClient:
    model = 'NO_API_SOFTWARE_FIXTURE'
    def __init__(self, reports):
        self.reports = copy.deepcopy(reports)
        self.messages = []
    def generate_structured_report(self, *, messages):
        self.messages.append(copy.deepcopy(messages))
        if len(self.messages) > len(self.reports):
            raise AssertionError('Unexpected extra report request')
        return SimpleNamespace(parsed_report=copy.deepcopy(self.reports[len(self.messages)-1]),
            raw_response={'software_fixture':True}, input_tokens=0, output_tokens=0,
            cost_usd=0., latency_sec=0., model_name=self.model)

rows = []
state_rows = []
patches = json.loads((ROOT/'harness_configs/selected_patches.json').read_text(encoding='utf-8'))
chosen = patches[f'{DATASET}:1701:trace']
ds_key = 'AI4I' if DATASET == 'AI4I' else 'Azure'

with contextlib.ExitStack() as stack:
    for name in ['socket.socket.connect','socket.socket.connect_ex','socket.create_connection',
                 'experiments.llm.openai_client.OpenAIClient.__init__',
                 'experiments.llm.openai_client.load_openai_api_key']:
        stack.enter_context(patch(name, side_effect=deny))
    for c in CASES:
        if c['dataset'] != ds_key:
            continue
        common = dict(report=c['report'],allowed_schema=c['schema'],action_ontology=c['ontology'])
        for route in ['runtime','offline']:
            result = (verify_runtime_report(**common,generation_evidence=c['observations']) if route == 'runtime'
                else evaluation_result_to_dict(evaluate_report(**common,task=c['task'],proxy_evidence=c['targets'])))
            actual = {k:result[k] for k in c['expected']}
            assert actual == c['expected'], (c['id'],route,c['expected'],actual)
            rows.append(dict(id=c['id'],dataset=DATASET,scope=c['scope'],route=route,
                             category=c['category'],expected=c['expected'],actual=actual))

    # All F10 cases exercise the actual production initial/repair/filter flow.
    # Expected repair counts and final flags are hand specified, not obtained
    # from a verifier helper. The shared dictionary fixtures carry no paid data.
    for method in ['M3','M4']:
        for ident in ['no_repair','wrong_gold_only','corrected_value','persistent_value',
                      'introduced_window','introduced_schema','filter_clears_action',
                      'action_cleared_value_persists']:
            b = fixture(ds_key)
            valid = copy.deepcopy(b['report'])
            first = copy.deepcopy(valid)
            second = copy.deepcopy(valid)
            expected_count, expected_failed, expected_pre = 1, False, False
            if ident in ['no_repair','wrong_gold_only']:
                expected_count,expected_pre = 0,None
                if ident == 'wrong_gold_only':
                    b['task']['gold_label'] = 'TWF' if ds_key == 'AI4I' else 'comp1'
            elif ident in ['corrected_value','persistent_value','introduced_window','introduced_schema']:
                first['evidence'][0]['value'] = '999999'
                if ident == 'persistent_value':
                    second = copy.deepcopy(first)
                    expected_failed,expected_pre = True,True
                elif ident == 'introduced_window':
                    second['evidence'][0]['window'] = 'after_failure'
                    expected_failed,expected_pre = True,True
                elif ident == 'introduced_schema':
                    second['confidence'] = 'unrecognized_level'
                    expected_failed,expected_pre = True,True
            else:
                b['report'] = first
                action(b,kind='shutdown',refs=['E1'])
                first = copy.deepcopy(b['report'])
                second = copy.deepcopy(first)
                expected_pre = True
                if ident == 'action_cleared_value_persists':
                    first['evidence'][0]['value'] = '999999'
                    second['evidence'][0]['value'] = '999999'
                    expected_failed = True
            client = ScriptedClient([first,second] if expected_count else [first])
            common = dict(task=b['task'],proxy_evidence=pd.DataFrame(b['observations']),
                allowed_schema=b['schema'],action_ontology=b['ontology'],
                action_preconditions={},generic_action_rules={})
            with patch('experiments.runners.run_smoke_test.evaluate_report',side_effect=deny), \
                 patch('experiments.harness.m4_patched.evaluate_report',side_effect=deny):
                out = (run_api_method(method='M3',client=client,**common) if method == 'M3'
                    else M4PatchedHarness(patch=chosen,client=client,dummy_mode=False).run(**common))
            actual = dict(repair_count=out.repair_count,F10=int(out.repair_failed),
                          pre_filter_failed=out.raw_response['repair_attempt_failed'])
            expected = dict(repair_count=expected_count,F10=int(expected_failed),pre_filter_failed=expected_pre)
            assert actual == expected, (DATASET,method,ident,actual,expected)
            assert len(client.messages) == 1 + expected_count
            scored = evaluation_result_to_dict(evaluate_report(task=b['task'],report=out.report,
                proxy_evidence=b['targets'],allowed_schema=b['schema'],action_ontology=b['ontology'],
                repair_failed=out.repair_failed))
            assert scored['F10'] == int(expected_failed)
            if ident == 'wrong_gold_only':
                assert scored['F1'] == 1 and scored['F10'] == 0
            if ident in ['filter_clears_action','action_cleared_value_persists']:
                assert out.raw_response['action_filter']['removed_count'] == 1
            state_rows.append(dict(dataset=DATASET,method=method,id=ident,expected=expected,actual=actual,
                                   final_F1=scored['F1'],scripted_report_requests=len(client.messages)))

groups = defaultdict(lambda:dict(positive=0,negative=0,TP=0,FN=0,TN=0,FP=0))
for r in rows:
    g=groups[(r['scope'],r['route'],r['category'])]
    e=r['expected'][r['category']];a=r['actual'][r['category']]
    g['positive' if e else 'negative']+=1
    g['TP' if e and a else 'FN' if e else 'FP' if a else 'TN']+=1
for r in state_rows:
    g=groups[('repair_control_flow',r['method'],'F10')]
    e=r['expected']['F10'];a=r['actual']['F10']
    g['positive' if e else 'negative']+=1
    g['TP' if e and a else 'FN' if e else 'FP' if a else 'TN']+=1
hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (CONTEXT/'experiments').rglob('*.py')}
result = dict(status='PASS',dataset=DATASET,context="exported_algorithm_source",
    generation_calls=0,network_calls=0,fit_calls=0,new_empirical_test_evaluations=0,
    detector_rows=rows,repair_states=state_rows,
    summary=[dict(scope=k[0],route=k[1],category=k[2],**v) for k,v in groups.items()],
    source_sha256=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(ROOT/'outputs/detectors').mkdir(parents=True,exist_ok=True)
(ROOT/'outputs/detectors'/f'{DATASET}_detector_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['status','dataset','generation_calls','summary']},indent=2))
