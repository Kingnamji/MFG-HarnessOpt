"""Six valid source/evidence/guide conditions; M0–M4 remain unchanged.

Raw packets read only observed columns, before the registered cutoff. The
frozen primary proxy score and a separate raw-fact audit are BOTH reported:
source-native observations are not silently relabelled into proxy targets.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import pandas as pd
from .contracts import digest
from .protocol import registry,require_partition
from experiments.data.generation_evidence import generation_task,visible_evidence,validate_observation_windows
from experiments.llm.observable_records import observation_records,json_value
from experiments.llm.prompts import (build_m2_prompt,build_system_prompt,format_allowed_schema,
    format_action_ontology,format_label_decision_guide,format_azure_association_guide)
from experiments.llm.report_contract import format_record_contract,format_diagnostic_contract

ROOT=Path(__file__).resolve().parents[1]
CELLS=('S0E0G0','S0E0G1','S1E0G0','S1E0G1','S1E1G0','S1E1G1')
GUIDE_ALIGNMENT_VERSION='explicit_guide_only_v1_20261025'
RAW_AI={'Type':'type','Air temperature [K]':'air_temperature','Process temperature [K]':'process_temperature',
        'Rotational speed [rpm]':'rotational_speed','Torque [Nm]':'torque','Tool wear [min]':'tool_wear'}

def _time(value):return pd.Timestamp(value).isoformat()

@lru_cache(maxsize=6)
def _source_file(dataset,source):
    if dataset=='AI4I':
        return pd.read_csv(ROOT/'Dataset/AI4I 2020/ai4i2020.csv',usecols=['UDI']+list(RAW_AI))
    folder=ROOT/'Dataset/Microsoft Azure Predictive Maintenance'
    specs={'telemetry':('PdM_telemetry.csv',['datetime','machineID','volt','rotate','pressure','vibration']),
           'errors':('PdM_errors.csv',['datetime','machineID','errorID']),
           'maintenance':('PdM_maint.csv',['datetime','machineID','comp']),
           'machines':('PdM_machines.csv',['machineID','model','age'])}
    name,cols=specs[source]
    return pd.read_csv(folder/name,usecols=cols,parse_dates=['datetime'] if 'datetime' in cols else None)

@dataclass(frozen=True)
class RawPacket:
    dataset:str
    task_id:str
    packet_json:str
    citations_json:str
    provenance_json:str
    @property
    def data(self):return json.loads(self.packet_json)
    @property
    def citations(self):return json.loads(self.citations_json)
    @property
    def provenance(self):return json.loads(self.provenance_json)

def source_native_packet(*,task,dataset,evaluation_seal=None,system_id=None,signature=None):
    """No labels, failure table, proxy target or derived-evidence values read."""
    ident=str(task['task_id']);index=registry(dataset)
    require_partition(dataset,[ident],{'search','validation','test'},purpose='observable source packet')
    meta=index[ident];records=[];packet={};prov=dict(task_id=ident,label_columns_read=False,source_records=[])
    if meta['split']=='test':
        if evaluation_seal is None:raise ValueError('Test observations may be constructed only for sealed final evaluation')
        evaluation_seal.authorize(dataset=dataset,candidate_id=system_id,signature=signature,task_ids=[ident])
    def cite(source,field,value,window,event_time=None):
        eid='R'+str(len(records)+1)
        records.append(dict(evidence_id=eid,source=source,field=field,statistic='observed_value',
            direction='observed',value=json_value(value),window=window,event_time=event_time,
            role='contextual' if source in {'machines','maintenance'} or field=='type' else 'symptomatic'))
        return eid
    if dataset=='AI4I':
        if str(task.get('udi')) not in {meta['udi'],str(float(meta['udi']))}:raise ValueError('Forged AI4I join identity')
        frame=_source_file(dataset,'features');row=frame.loc[frame.UDI.eq(int(float(meta['udi'])))]
        if len(row)!=1:raise ValueError('Unambiguous source UDI required')
        for original,field in RAW_AI.items():
            value=row.iloc[0][original];cite('machine_features',field,value,'static_instance')
        packet=dict(machine_features=[{k:r[k] for k in ('evidence_id','field','value')} for r in records])
        prov['source_records']=[dict(source='machine_features',udi=meta['udi'])]
    else:
        cutoff=pd.Timestamp(meta['failure_time']);machine=int(float(meta['machineID']))
        if pd.Timestamp(task.get('failure_time'))!=cutoff or int(float(task.get('machineID')))!=machine:
            raise ValueError('Raw packet anchor must match the registered observation cutoff')
        packet=dict(cutoff=cutoff.isoformat(),sources={})
        for source in ('telemetry','errors','maintenance','machines'):
            data=_source_file(dataset,source);frame=data.loc[data.machineID.eq(machine)].copy()
            if source=='telemetry':
                baseline=frame.loc[frame.datetime.ge(cutoff-pd.Timedelta(days=30))&frame.datetime.lt(cutoff-pd.Timedelta(days=7))]
                baseline_end=cutoff-pd.Timedelta(days=7)
                if baseline.empty:baseline_end=cutoff-pd.Timedelta(hours=24)
                mask=(frame.datetime.ge(cutoff-pd.Timedelta(days=30))&frame.datetime.lt(baseline_end)) | (frame.datetime.ge(cutoff-pd.Timedelta(hours=24))&frame.datetime.lt(cutoff))
                frame=frame.loc[mask].sort_values('datetime',kind='stable')
                fields=['volt','rotate','pressure','vibration'];window='telemetry_24h_before_failure'
                packet['telemetry_baseline_interval']=[_time(cutoff-pd.Timedelta(days=30)),_time(baseline_end)]
                packet['telemetry_window_interval']=[_time(cutoff-pd.Timedelta(hours=24)),_time(cutoff)]
            elif source in ('errors','maintenance'):
                delta=pd.Timedelta(hours=24) if source=='errors' else pd.Timedelta(days=30)
                frame=frame.loc[frame.datetime.ge(cutoff-delta)&frame.datetime.lt(cutoff)].sort_values('datetime',kind='stable')
                fields=['errorID'] if source=='errors' else ['comp'];window='error_24h_before_failure' if source=='errors' else 'maintenance_30d_before_failure'
            else:fields=['model','age'];window='static'
            rows=[]
            for row in frame.to_dict('records'):
                event=_time(row['datetime']) if source!='machines' else None
                values=[json_value(row[f]) for f in fields]
                # Historical baseline values are input context, not current
                # evidence or action prerequisites. Only last-24h telemetry is citable.
                citable=source!='telemetry' or pd.Timestamp(row['datetime'])>=cutoff-pd.Timedelta(hours=24)
                ids=[cite(source,f,row[f],window,event) for f in fields] if citable else []
                rows.append([event,ids,values])
            packet['sources'][source]=dict(columns=['event_time','evidence_ids','values'],value_fields=fields,rows=rows)
            prov['source_records'].append(dict(source=source,machineID=machine,rows=len(frame)))
        prov['cutoff']=_time(cutoff)
        assert all(pd.Timestamp(r['event_time'])<cutoff for r in records if r['event_time'])
    # Uniform metadata is a citation convention, not a diagnostic annotation.
    packet['citation_convention']=dict(statistic='observed_value',direction='observed',
        role='machines/maintenance/type=contextual; other raw measurements=symptomatic',
        windows={r['source']:r['window'] for r in records},baseline_history='uncitable context; empty evidence_ids')
    return RawPacket(dataset,ident,json.dumps(json_value(packet),ensure_ascii=False,separators=(',',':'),allow_nan=False),
        json.dumps(json_value(records),ensure_ascii=False,allow_nan=False),json.dumps(prov,ensure_ascii=False))

@dataclass(frozen=True)
class SixConditionInput:
    dataset:str
    cell:str
    task_id:str
    split:str
    messages:tuple
    citations_json:str
    provenance_json:str
    exact_full_M2:bool
    @property
    def request_sha256(self):return digest(self.messages)
    @property
    def citations(self):return json.loads(self.citations_json)
    @property
    def provenance(self):return json.loads(self.provenance_json)

def ablation_signature(cell):
    if cell not in CELLS:raise ValueError('Unknown cell')
    return digest(dict(cell=cell,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

def explicit_guide_block(allowed_schema,association_guide,*,enabled):
    """Vary explicit guidance only; retain generic contracts/domain features.

    The enabled spelling matches the frozen full M2 prompt exactly. G0 does
    not remove knowledge already present in observed/derived evidence.
    """
    label_guide=format_label_decision_guide(allowed_schema if enabled else None)
    prior=format_azure_association_guide(association_guide if enabled else None)
    return 'Label decision guide:\n'+label_guide+'\n\nAzure search-derived association guide:\n'+prior


def build_six_condition(*,cell,task,generation_evidence,allowed_schema,action_ontology,association_guide=None,raw_packet=None,evaluation_seal=None):
    if cell not in CELLS:raise ValueError('Only six valid hierarchical cells; S0E1 is not identified')
    dataset='AI4I' if 'TWF' in allowed_schema['candidate_labels'] else 'PdM'
    ident=str(task['task_id']);require_partition(dataset,[ident],{'search','validation','test'},purpose='ablation packet')
    split=registry(dataset)[ident]['split'];S=int(cell[1]);E=int(cell[3]);G=int(cell[5])
    if split=='test':
        if evaluation_seal is None:raise ValueError('Seal ablation definitions before constructing test inputs')
        evaluation_seal.authorize(dataset=dataset,candidate_id='M2_'+cell,signature=ablation_signature(cell),task_ids=[ident])
    if dataset=='PdM' and association_guide:
        control=association_guide.get('leakage_control',{})
        if control.get('uses_split')!='search' or control.get('uses_validation_labels') is not False or control.get('uses_test_labels') is not False:
            raise ValueError('Prior must be fitted only on search')
    clean=generation_task(task)
    observed=visible_evidence(generation_evidence)
    if not observed.empty and (not observed.task_id.astype(str).eq(ident).all()):raise ValueError('Cross-task evidence')
    validate_observation_windows(observed,pd.DataFrame([clean]))
    citations=[];prov=dict(cell=cell,split=split,task_id=ident,guide_on=bool(G),source_on=bool(S),evidence_on=bool(E),
                         primary_evaluator_unchanged=True,proxy_target_visible=False)
    guide_on=explicit_guide_block(allowed_schema,association_guide,enabled=True)
    guide_off=explicit_guide_block(allowed_schema,association_guide,enabled=False)
    guides=guide_on if G else guide_off
    if S and E:
        # G0 inherits the exact competent full M2 context; never replace it
        # with a shorter reporting prompt or modify frozen M0--M4 builders.
        user=build_m2_prompt(task=clean,proxy_evidence=observed,allowed_schema=allowed_schema,
                            action_ontology=action_ontology,association_guide=association_guide)
        if user.count(guide_on)!=1:raise ValueError('Frozen M2 guide boundary changed')
        if not G:user=user.replace(guide_on,guide_off,1)
        citations=observation_records(observed,max_rows=10)
    else:
        if S and not E:
            raw=raw_packet or source_native_packet(task=clean,dataset=dataset,evaluation_seal=evaluation_seal,system_id='M2_'+cell,signature=ablation_signature(cell))
            if raw.task_id!=ident or raw.dataset!=dataset:raise ValueError('Raw packet task mismatch')
            citations=raw.citations;source_text=raw.packet_json;prov['source_provenance']=raw.provenance
            task_context=dict(dataset=clean.get('dataset'),task_id=ident)
            contract=format_record_contract()+'\nUse the source-native packet and its uniform citation convention. Evidence contains raw observations; perform any diagnostic derivation yourself and do not invent a derived evidence ID.'
        else:
            source_text='No task-specific observations are supplied.';task_context=dict(dataset=clean.get('dataset'),task_id=ident)
            contract=format_record_contract()+'\nThere are no citable task observations. Leave evidence and actions empty when no grounded rationale IDs are available; do not invent measurements.'
        # Generic diagnosis competence is common to G0 and G1.
        contract+='\n'+format_diagnostic_contract()
        user='Task: Source-aware structured manufacturing diagnostic report.\n\n'+contract
        user+='\n\nTask:\n'+json.dumps(json_value(task_context),ensure_ascii=False,separators=(',',':'),allow_nan=False)
        user+='\n\nAllowed diagnostic schema:\n'+format_allowed_schema(allowed_schema)
        user+='\n\n'+guides
        user+='\n\nObserved source packet:\n'+source_text
        user+='\n\nAction ontology:\n'+format_action_ontology(action_ontology)
        user+='\nSelect one candidate label, express uncertainty with confidence/limitations, and return the unchanged structured report schema. No runtime repair or action filtering is added in this ablation.'
    messages=(dict(role='system',content=build_system_prompt()),dict(role='user',content=user))
    if user.count(guides)!=1:raise ValueError('Ambiguous explicit guide block')
    non_guide_user=user.replace(guides,'<EXPLICIT_GUIDE_PRIOR_BLOCK>',1)
    prov.update(guide_alignment_version=GUIDE_ALIGNMENT_VERSION,
                guide_block_sha256=digest(guides),
                non_guide_messages_sha256=digest((messages[0],dict(role='user',content=non_guide_user))),
                guide_off_removes_only_explicit_guidance=True)
    if sum(len(m['content']) for m in messages)>200000:raise ValueError('Source-native context exceeds the registered bound; do not silently truncate')
    prov.update(visible_citations=len(citations),citation_packet_sha256=digest(citations),
        primary_proxy_scope='frozen constructed-evidence targets; raw facts may be outside this target universe',
        supplemental_scope='label-blind supplied raw/constructed citation universe',
        source_metadata_is_not_output_supports=True)
    return SixConditionInput(dataset,cell,ident,split,messages,json.dumps(json_value(citations),ensure_ascii=False,allow_nan=False),
                             json.dumps(prov,ensure_ascii=False),cell=='S1E1G1')

def run_six_condition(packet,*,client,iteration=0,draw_id=1):
    if packet.split=='test' and client._test_authorization is None:raise ValueError('Seal final ablation evaluation before test')
    with client.at(candidate_id='M2_'+packet.cell,iteration=iteration,task_id=packet.task_id,max_report_calls=1,draw_id=draw_id):
        # Exactly one generation. No repair, filter or manual substitution.
        return client.generate_structured_report(messages=list(packet.messages))

def evaluate_six_response(*,packet,response,task,targets,allowed_schema,action_ontology,action_preconditions,generic_action_rules,usage_rows=None,association_guide=None):
    from .measurement import measure_result
    from experiments.harness.base import HarnessResult
    report=response.parsed_report
    result=HarnessResult(method='M2',harness_id='M2_'+packet.cell,task_id=packet.task_id,dataset=task['dataset'],
        report=report,input_tokens=response.input_tokens,output_tokens=response.output_tokens,cost_usd=response.cost_usd,
        latency_sec=response.latency_sec,model_name=response.model_name,
        accessed_sources=sorted({r['source'] for r in packet.citations}),raw_response=response.raw_response)
    return measure_result(task=task,targets=targets,observations=packet.citations,allowed_schema=allowed_schema,
        action_ontology=action_ontology,action_preconditions=action_preconditions,generic_action_rules=generic_action_rules,result=result,usage_rows=usage_rows,association_guide=association_guide)
