"""One archived-report measurement contract for LLM and non-LLM systems.

Primary metrics always use the unchanged evaluator. Factual/complete-reference
audits are supplemental; they never change runtime, selection or target rules.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import os
import time
import sys
import platform
from functools import lru_cache
from importlib.metadata import version
import numpy as np
import pandas as pd
from .contracts import digest
from experiments.llm.observable_records import json_value
from experiments.evaluation.observable_matching import values_equal,times_equal
from experiments.runners.run_search import evaluate_harness_result
from experiments.harness.base import HarnessResult

FIELDS=('source','field','statistic','direction','value','window','role','event_time')

@lru_cache(maxsize=1)
def environment_metadata():
    packages={}
    for name in ('numpy','pandas','scikit-learn','openai'):
        try:packages[name]=version(name)
        except Exception:packages[name]=None
    return dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),
        packages=packages,clock='time.perf_counter/time.process_time',native_function_timing='read-only sys.setprofile intervals')

def factual_audit(report,observations,*,input_ids_supplied=True):
    records=observations.to_dict('records') if hasattr(observations,'to_dict') else list(observations)
    by_id={str(r['evidence_id']):r for r in records};claims=report.get('evidence') or [];actions=report.get('next_steps') or []
    claim_ids=[str(c.get('evidence_id','')) for c in claims];matches=[];errors=[];value_matches=[]
    for c in claims:
        ident=str(c.get('evidence_id',''));r=by_id.get(ident);wrong=[]
        if claim_ids.count(ident)==1 and any(c.get('source')==o.get('source') and c.get('field')==o.get('field') and values_equal(c.get('value'),o.get('value')) for o in records):
            value_matches.append(ident)
        if r is None or claim_ids.count(ident)>1:wrong=['unknown_or_duplicate_id']
        else:
            for k in FIELDS:
                if k=='event_time':ok=times_equal(c.get(k),r.get(k))
                elif k=='value':ok=values_equal(c.get(k),r.get(k))
                else:ok=str(c.get(k,''))==str(r.get(k,''))
                if not ok:wrong.append(k)
        (errors if wrong else matches).append(dict(evidence_id=ident,fields=wrong))
    complete=0;valid=0;dangling=[];all_in_input=0;signatures=[]
    for i,a in enumerate(actions):
        refs=list(map(str,a.get('rationale_evidence_ids') or []))
        local=bool(refs) and all(r in claim_ids for r in refs)
        in_input=bool(refs) and all(r in by_id for r in refs) if input_ids_supplied else bool(refs) and all(r in value_matches for r in refs)
        complete+=int(local);all_in_input+=int(in_input)
        valid+=int(local and in_input)
        if not(local and in_input):dangling.append(i)
        signatures.append(digest({k:a.get(k) for k in ('action_type','target','rationale_evidence_ids')}))
    return dict(factual_precision=len(value_matches)/len(claims) if claims else 0.,factual_matches=len(value_matches),
        metadata_copy_precision=(len(matches)/len(claims) if claims else 0.) if input_ids_supplied else None,
        input_ids_supplied=input_ids_supplied,
        factual_claims=len(claims),
        factual_errors=[dict(evidence_id=i,fields=['source_field_value_or_duplicate']) for i in claim_ids if i not in value_matches],
        metadata_copy_errors=errors if input_ids_supplied else None,
        evidence_coverage=int(bool(claims)),action_coverage=int(bool(actions)),
        all_refs_in_report_ratio=complete/len(actions) if actions else None,
        all_refs_in_input_ratio=all_in_input/len(actions) if actions and input_ids_supplied else None,
        all_refs_complete_ratio=valid/len(actions) if actions else None,dangling_action_indices=dangling,
        n_actions=len(actions),n_evidence=len(claims),duplicate_actions=len(signatures)-len(set(signatures)))

def stage_reports(result):
    raw=result.raw_response if isinstance(result.raw_response,dict) else {}
    initial=raw.get('initial_report');pre=raw.get('pre_filter_report')
    if initial is None: initial=result.report
    if pre is None: pre=initial
    stages={'initial':initial,'pre_filter':pre,'final':result.report}
    repair=raw.get('repair_report')
    if result.repair_count and repair is not None:stages['post_repair']=repair
    return deepcopy(stages)

def measure_result(*,task,targets,observations,allowed_schema,action_ontology,action_preconditions,generic_action_rules,result,usage_rows=None,input_ids_supplied=True,association_guide=None):
    stages=stage_reports(result);metrics={}
    for name,report in stages.items():
        # Intermediate repair_failure is not the final report's F10.
        uses=usage_rows[:1] if name=='initial' and usage_rows else usage_rows
        kwargs={}
        if uses is not None:
            kwargs=dict(input_tokens=sum(r['input_tokens'] for r in uses),output_tokens=sum(r['output_tokens'] for r in uses),
                        cost_usd=sum(r['cost_usd_uncached'] for r in uses))
        intermediate=replace(result,report=report,repair_failed=result.repair_failed if name=='final' else False,**kwargs)
        primary=evaluate_harness_result(task=task,proxy_evidence=targets,allowed_schema=allowed_schema,
            action_ontology=action_ontology,action_preconditions=action_preconditions,
            generic_action_rules=generic_action_rules,harness_result=intermediate)
        if usage_rows is None and name!='final':
            primary.update(cost_per_report=None,input_tokens=None,output_tokens=None)
        metrics[name]=dict(primary=primary,supplemental=factual_audit(report,observations,input_ids_supplied=input_ids_supplied),
                          confidence=report.get('confidence'))
    initial=metrics['initial']['primary']['label_correct'];final=metrics['final']['primary']['label_correct']
    extra=result.extra or {};audit=extra.get('action_filter') or {}
    transition=dict(repair_supported=result.method in {'M3','M4'},repair_count=result.repair_count,
        repair_failed=result.repair_failed,repair_attempt_failed=extra.get('repair_attempt_failed'),
        label_fix=int(initial==0 and final==1),label_regression=int(initial==1 and final==0),
        repair_label_fix=None,repair_label_regression=None,
        filter_removed=audit.get('removed_count'),filter_downgraded=audit.get('downgraded_count'))
    from .formal_metrics import supplemental_report_audit,source_packet_audit
    for name,report in stages.items():
        metrics[name]['supplemental'].update(supplemental_report_audit(report,observations,allowed_schema,association_guide))
        from dataclasses import asdict
        from experiments.evaluation.evidence_matcher import match_generated_evidence
        from experiments.evaluation.action_evaluator import evaluate_actions
        matched=match_generated_evidence(generated_evidence=report.get('evidence') or [],proxy_evidence=targets,
            allowed_schema=allowed_schema,include_context_for_precision=True,include_context_in_recall=False)
        action_match=evaluate_actions(report,matched.matched_generated_ids,action_ontology,action_preconditions or {},generic_action_rules or {})
        metrics[name]['offline_matching_details']=dict(evidence=asdict(matched),actions=asdict(action_match))
    transition.update(filter_supported=bool(audit),filter_audit=deepcopy(audit),
        pre_filter_actions=len(stages['pre_filter'].get('next_steps') or []),
        final_actions=len(result.report.get('next_steps') or []),
        filter_label_unchanged=stages['pre_filter'].get('predicted_label')==result.report.get('predicted_label'),
        filter_evidence_unchanged=stages['pre_filter'].get('evidence')==result.report.get('evidence'))
    if result.repair_count:
        after=metrics.get('post_repair',metrics['pre_filter'])['primary']['label_correct']
        transition.update(repair_label_fix=int(initial==0 and after==1),repair_label_regression=int(initial==1 and after==0))
    return json_value(dict(version='common_measurement_v1',stages=metrics,stage_reports=stages,usage_rows=usage_rows,
        transitions=transition,source_access=result.execution_log(),source_packet=source_packet_audit(observations),
        evaluation_packet=dict(offline_only=True,gold_label=task.get('gold_label'),
            proxy_targets=targets.to_dict('records') if hasattr(targets,'to_dict') else list(targets),
            allowed_schema_sha256=digest(allowed_schema),action_ontology_sha256=digest(action_ontology)),
        end_to_end_latency_sec=result.latency_sec,measurement_is_offline=True))

def prompt_observation_registry(*,task,observed,allowed_schema,method,patch=None,association_guide=None):
    """Exactly the citable input view; M0/M1 do not receive evidence IDs.

    Do not audit an M4 against a richer unformatted table. Value-level factual
    accuracy and literal metadata copying are separate measurements.
    """
    if method in {'M0','M1'}:
        from experiments.data.generation_evidence import generation_task
        clean=generation_task(task);records=[]
        for source,fields in allowed_schema['allowed_sources'].items():
            for field in fields:
                if field not in clean or isinstance(clean[field],(dict,list)):continue
                records.append(dict(evidence_id='CTX_'+field,source=source,field=field,value=clean[field]))
        return records,False
    from nonllm_baselines.views import build_view
    view=build_view(task=task,generation_evidence=observed,allowed_schema=allowed_schema,
                    method='M4' if patch is not None else 'M3',patch=patch,association_guide=association_guide)
    return view.records,True

def measure_nonllm(*,task,targets,observations,allowed_schema,action_ontology,action_preconditions,generic_action_rules,rendered,method,latency_sec=0.,association_guide=None):
    observations=observations.to_dict('records') if hasattr(observations,'to_dict') else list(observations)
    final=rendered['report'];raw=rendered.get('raw_report',final)
    usage=rendered.get('usage',{});inherited=usage.get('inherited_label_usage') if usage.get('fully_nonLLM_diagnosis') is False else None
    if usage.get('fully_nonLLM_diagnosis') is False and (not inherited or not {'input_tokens','output_tokens','cost_usd','latency_sec'}<=set(inherited)):
        raise ValueError('Same-label template must retain known parent LLM usage; never report zero end-to-end API cost')
    from experiments.llm.report_contract import REPORT_CONTRACT_VERSION
    extra=dict(protocol={'report_contract_version':REPORT_CONTRACT_VERSION,'mode':'nonLLM_or_conditional_template'})
    if rendered.get('action_filter'):extra['action_filter']=rendered['action_filter']
    result=HarnessResult(method=method,harness_id=method,task_id=str(task['task_id']),dataset=task['dataset'],
        report=final,model_name='nonLLM',latency_sec=latency_sec,
        accessed_sources=sorted({r['source'] for r in observations}),
        raw_response=dict(initial_report=raw,pre_filter_report=raw),
        extra=extra)
    uses=[] if inherited is None else [dict(input_tokens=inherited['input_tokens'],output_tokens=inherited['output_tokens'],cost_usd_uncached=inherited['cost_usd'])]
    if inherited:result=replace(result,latency_sec=latency_sec+inherited['latency_sec'])
    return measure_result(task=task,targets=targets,observations=observations,allowed_schema=allowed_schema,
        action_ontology=action_ontology,action_preconditions=action_preconditions,generic_action_rules=generic_action_rules,result=result,usage_rows=uses,association_guide=association_guide)

def summarize(frames,configs):
    from experiments.optimization.candidate_acceptance import diagnostic_summary
    frame=pd.DataFrame(frames)
    return diagnostic_summary(frame,configs)

def atomic_archive(path,packet):
    path=Path(path)
    if path.exists(): raise FileExistsError('Measurement archives are immutable')
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(json_value(packet),ensure_ascii=False,allow_nan=False)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('x',encoding='utf-8') as f:f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    return digest(json.loads(text))

class StageTimer:
    def __init__(self):self.events=[]
    def measure(self,name):
        from contextlib import contextmanager
        @contextmanager
        def span():
            start=time.perf_counter();cpu=time.process_time()
            try:yield
            finally:
                end=time.perf_counter()
                self.events.append(dict(stage=name,start_monotonic=start,end_monotonic=end,wall_sec=end-start,cpu_sec=time.process_time()-cpu))
        return span()

class NativeStageProfiler:
    """Read-only call timing; no replacement of frozen functions or returns.

    Events carry intervals/depth. Nested verification inside the action filter
    must not be added again to the filter's inclusive elapsed time.
    """
    TARGETS={'verify_runtime_report':'runtime_verification','finalize_verified_report':'action_filter'}
    def __init__(self):self.events=[];self.active={}
    def _profile(self,frame,event,arg):
        name=frame.f_code.co_name;path=frame.f_code.co_filename.replace('\\','/')
        if name not in self.TARGETS or '/experiments/' not in path:return
        key=id(frame)
        if event=='call':self.active[key]=(time.perf_counter(),time.process_time(),len(self.active))
        elif event=='return' and key in self.active:
            started,cpu,depth=self.active.pop(key)
            self.events.append(dict(stage=self.TARGETS[name],start_monotonic=started,end_monotonic=time.perf_counter(),
                                    cpu_sec=time.process_time()-cpu,nesting_depth=depth))
    def __enter__(self):
        if sys.getprofile() is not None:raise RuntimeError('Do not replace an existing profiler')
        sys.setprofile(self._profile);return self
    def __exit__(self,*args):sys.setprofile(None)
