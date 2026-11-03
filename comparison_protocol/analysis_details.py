"""Offline accounting and report summaries; never used for generation/selection."""
from collections import defaultdict
from pathlib import Path
import json,math
from .formal_metrics import break_even, rate

def group_key(r):
    return ':'.join(str(r.get(k,'unspecified')) for k in ('dataset_key','candidate_id','run_id','draw_id'))

def report_action_summary(records):
    groups=defaultdict(list)
    for r in records:groups[group_key(r)].append(r)
    result={}
    for key,rs in groups.items():
        stages={}
        for name in ('initial','post_repair','pre_filter','final'):
            rows=[r['common_measurement']['stages'][name] for r in rs if name in r['common_measurement']['stages']]
            if not rows:continue
            counts=[x['supplemental']['n_actions'] for x in rows]
            nonempty=[x for x,n in zip(rows,counts) if n>0]
            details=[x.get('offline_matching_details',{}).get('actions') for x in rows]
            known=all(x is not None and 'unsafe_actions' in x for x in details)
            total=sum(counts)
            stages[name]=dict(n_reports=len(rows),n_nonempty_reports=len(nonempty),n_empty_reports=len(rows)-len(nonempty),
                action_coverage=rate(len(nonempty),len(rows)),total_actions=total,
                unsafe_macro_all=sum(x['primary']['unsafe_recommendation_rate'] for x in rows)/len(rows),
                unsafe_macro_nonempty=rate(sum(x['primary']['unsafe_recommendation_rate'] for x in nonempty),len(nonempty)),
                unsafe_micro=rate(sum(x['unsafe_actions'] for x in details),total) if known else None,
                unsafe_action_count=sum(x['unsafe_actions'] for x in details) if known else None,
                linked_action_count=sum(x['evidence_linked_actions'] for x in details) if known else None,
                linked_action_micro=rate(sum(x['evidence_linked_actions'] for x in details),total) if known else None,
                exact_action_counts_available=known)
        transitions=[r['common_measurement']['transitions'] for r in rs]
        active=[t for t in transitions if t.get('filter_supported')]
        attempted=[t for t in transitions if t.get('repair_count',0)>0]
        result[key]=dict(stages=stages,transitions=dict(n_reports=len(rs),repair_attempted_reports=len(attempted),
            repair_label_fixes=sum(t.get('repair_label_fix') or 0 for t in attempted),
            repair_label_damage=sum(t.get('repair_label_regression') or 0 for t in attempted),
            repair_attempt_failed_reports=sum(bool(t.get('repair_attempt_failed')) for t in attempted),
            final_repair_failed_reports=sum(bool(t.get('repair_failed')) for t in attempted),
            filter_evaluated_reports=len(active),
            removed_actions=sum(t['filter_removed'] for t in active) if active and all(t.get('filter_removed') is not None for t in active) else None,
            downgraded_actions=sum(t['filter_downgraded'] for t in active) if active and all(t.get('filter_downgraded') is not None for t in active) else None,
            label_preservation_violations=sum(t.get('filter_label_unchanged') is False for t in active),
            evidence_preservation_violations=sum(t.get('filter_evidence_unchanged') is False for t in active)),
            interpretation='Report-policy compliance/coverage; no physical safety or human usefulness inference.')
    return result

def report_failure_summary(records):
    groups=defaultdict(list)
    for r in records:groups[group_key(r)].append(r)
    result={}
    for key,rs in groups.items():
        result[key]={}
        for name in ('initial','post_repair','pre_filter','final'):
            rows=[r['common_measurement']['stages'][name]['primary'] for r in rs if name in r['common_measurement']['stages']]
            if rows:result[key][name]={f'F{i}':dict(count=sum(bool(x[f'F{i}']) for x in rows),n_evaluated_reports=len(rows),
                rate=sum(bool(x[f'F{i}']) for x in rows)/len(rows)) for i in range(1,11)}
    return dict(by_system_run_draw_stage=result,
        denominators='Separate report-stage denominators; never pool stages or methods as independent cases.',
        sentinel_scope='Natural flags do not estimate detector sensitivity. Independent F4/F6/F8 fixtures are a separate artifact.')

def _known_sum(rows,field):
    values=[r.get(field) for r in rows]
    return sum(values) if all(v is not None and math.isfinite(v) for v in values) else None

def campaign_economics(campaign):
    """Full standalone setup attribution and separately deduplicated real spend."""
    result={};ledger=campaign.budget.rows
    for (ds,run,arm),selection in campaign.selections.items():
        cid=selection['selected_candidate_id']
        own=[u for u in ledger if (u.get('dataset'),str(u.get('run_id')),u.get('arm'),u.get('allocation_group'))==(ds,str(run),arm,'search_primary')]
        shared=[u for u in ledger if (u.get('dataset'),str(u.get('run_id')),u.get('arm'))==(ds,str(run),'shared')
                and u.get('campaign_phase') in ('search_reference','validation_reference')]
        n_arms=sum(d==ds and r==run for d,r,a in campaign.selections)
        selected=[r for draw in (1,2,3) for r in campaign.test_records[ds,1701 if cid=='M3' else run,cid,draw]]
        baseline=[r for draw in (1,2,3) for r in campaign.test_records[ds,1701,'M3',draw]]
        bases={}
        for basis in ('uncached','actual'):
            field='cost_usd_'+basis
            def deploy(rows):
                costs=[]
                for record in rows:
                    usage=record['common_measurement'].get('usage_rows')
                    costs.append(_known_sum(usage,field) if usage is not None else None)
                return sum(costs)/len(costs) if costs and all(c is not None for c in costs) else None
            own_cost=_known_sum(own,field);shared_cost=_known_sum(shared,field)
            setup=own_cost+shared_cost if own and shared and own_cost is not None and shared_cost is not None else None
            before,after=deploy(baseline),deploy(selected)
            value=break_even(setup,before,after) if all(v is not None for v in (setup,before,after)) else dict(
                reports=None,reason='Unknown usage: no complete break-even estimate')
            bases[basis]=dict(standalone_setup=setup,arm_setup=own_cost,shared_reference_setup=shared_cost,
                shared_cost_equal_share=shared_cost/n_arms if shared_cost is not None else None,
                baseline_report_cost=before,selected_report_cost=after,break_even=value)
        result[f'{ds}:{run}:{arm}']=dict(selected_candidate_id=cid,bases=bases,
            deployment_draws=[1,2,3],baseline_reports=len(baseline),selected_reports=len(selected),
            setup_scope='Own search+primary validation plus full common M3 search+validation reference setup for standalone use.',
            shared_policy='Standalone totals are logical attribution, not additive campaign invoices. Equal-share field is descriptive only.',
            excluded_from_deployment_setup='Post-selection sensitivity audit and final test are research evaluation costs; phase ledger reports them separately.')
    return dict(by_optimizer=result,campaign_real_cost={b:_known_sum(ledger,'cost_usd_'+b) for b in ('actual','uncached')},
        campaign_real_calls=len(ledger),campaign_real_spend_deduplicated=True,
        legacy_warning='Historical reference-patch optimization costs are not reconstructed by this campaign.')

def _trace_payload(messages):
    for message in messages or []:
        text=message.get('content','');marker='Selected failure traces:\n'
        if isinstance(text,str) and marker in text:
            value,_=json.JSONDecoder().raw_decode(text.split(marker,1)[1].lstrip())
            if not isinstance(value,list):raise ValueError('Invalid archived trace payload')
            return value
    return None

def feedback_failure_summary(root):
    """Selection, actual prompt inclusion, and paired child flags; no causality claim."""
    root=Path(root);index={}
    for p in root.glob('**/*_search_draw1/*.json'):
        if p.name.endswith('_failure.json'):continue
        r=json.loads(p.read_text(encoding='utf-8'));key=(r['dataset_key'],str(r.get('run_id')),r['candidate_id'],r['task_id'])
        if r.get('comparison_split','search')!='search':raise ValueError('Held-out report mislabeled as search lineage')
        if key in index:raise ValueError('Ambiguous search report lineage')
        index[key]=r['metrics']
    groups={};exposures=[]
    from .recovery_jobs import canonical_feedback_archives
    for path in canonical_feedback_archives(root):
        parts=path.relative_to(root).parts
        ds=next((p for p in parts if p in ('AI4I','PdM')),None)
        run=next((p for p in parts if p in ('1701','1702','1703')),None)
        iteration=next((p for p in parts if p.startswith('iter_')),'unknown')
        key=f'{ds}:{run}:{iteration}'
        selected=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if any(b.get('split')!='search' for b in selected):raise ValueError('Held-out bundle in search feedback archive')
        counts={f'F{i}':dict(selected_bundle_count=sum(f'F{i}' in b.get('failure_ids',[]) for b in selected),
            prompt_exposures=0,paired_scored_exposures=0,cleared_exposures=0,persisted_exposures=0,unscored_exposures=0) for i in range(1,11)}
        prompts=list(path.parent.parent.glob('proposals/*_proposal_result.json'))
        unknown=0
        for pp in prompts:
            proposal=json.loads(pp.read_text(encoding='utf-8'));payload=_trace_payload(proposal.get('proposer_messages'))
            if payload is None:unknown+=1;continue
            cid=proposal['candidate_id']
            for bundle in payload:
                for flag in bundle.get('failure_ids',[]):
                    if flag not in counts:raise ValueError('Unknown failure type in prompt')
                    counts[flag]['prompt_exposures']+=1
                    metrics=index.get((ds,run,cid,bundle['task_id']))
                    if metrics is None:counts[flag]['unscored_exposures']+=1;status='unscored'
                    else:
                        counts[flag]['paired_scored_exposures']+=1
                        status='persisted' if metrics[flag] else 'cleared';counts[flag][status+'_exposures']+=1
                    exposures.append(dict(dataset=ds,run_id=run,iteration=iteration,candidate_id=cid,
                        task_id=bundle['task_id'],failure=flag,status=status,proposal=str(pp.relative_to(root))))
        groups[key]=dict(n_selected_bundles=len(selected),selected_success_controls=sum(not b.get('failure_ids') for b in selected),
            n_proposal_archives=len(prompts),unknown_prompt_payloads=unknown,by_failure=counts)
    return dict(groups=groups,paired_exposures=exposures,
        counting='Selected bundle counts per iteration; prompt exposures per proposal, including repeated tasks. Cleared is an observed child-report flag transition, not causal attribution.',
        inclusion_scope='Prompt inclusion is observable; actual model attention/use is not measured. Rejected/unscored children never count as fixes.')
