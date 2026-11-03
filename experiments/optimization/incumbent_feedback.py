"""One search task per context bundle, with current and recent paired reports."""
import pandas as pd


def select_incumbent_context(repo, *, incumbent_id, iteration, max_bundles,
                             task_ids=None, datasets=None):
    if max_bundles <= 0 or iteration <= 0:
        raise ValueError('Positive context budget and iteration required')
    frame = repo.metrics.copy()
    if task_ids is not None:
        frame = frame.loc[frame.task_id.astype(str).isin(task_ids)]
    if datasets is not None:
        frame = frame.loc[frame.dataset.isin(datasets)]
    frame = frame.loc[frame.method.isin(['M3','M4'])]
    if 'iteration' not in frame: frame['iteration'] = 0
    frame['iteration'] = pd.to_numeric(frame.iteration, errors='coerce').fillna(0)
    frame = frame.loc[frame.iteration < iteration]
    selected = []

    def bundle(row):
        b = repo.build_trace_bundle(task_id=str(row.task_id), method=str(row.method),
                                    harness_id=str(row.harness_id)).to_dict()
        if b['split'] != 'search':
            raise ValueError('Paired feedback must use search traces only')
        if not isinstance(b['report'], dict):
            raise ValueError('Paired feedback is missing a saved report')
        return b

    def reference(role, row, primary):
        b = bundle(row)
        return dict(role=role, method=b['method'], harness_id=b['harness_id'],
                    label_correct=b['violation_context'].get('label_correct'),
                    report=b['report'], same_as_primary=b['harness_id']==primary['harness_id'])

    for task_id, group in frame.groupby('task_id', sort=True):
        base = group.loc[group.method.eq('M3')]
        if base.empty: raise ValueError(f'Missing paired M3 task: {task_id}')
        base = base.iloc[-1]
        current = base if incumbent_id == 'M3' else group.loc[group.harness_id.eq(incumbent_id)]
        if isinstance(current, pd.DataFrame):
            if len(current) != 1: raise ValueError(f'Missing/ambiguous incumbent task: {task_id}')
            current = current.iloc[0]
        recent = group.loc[group.method.eq('M4') & group.iteration.eq(iteration-1)]
        regressions = recent.loc[recent.label_correct.eq(0)] if current.label_correct == 1 else recent.iloc[:0]
        if not regressions.empty:
            primary = regressions.sort_values('harness_id').iloc[0]
            category, priority = 'recent_label_regression', 0
        elif current.label_correct == 0:
            primary, category, priority = current, 'incumbent_residual_label_error', 1
        elif base.label_correct == 0 and current.label_correct == 1:
            primary, category, priority = current, 'preserve_incumbent_improvement', 2
        elif any(float(current.get(f'F{i}',0) or 0) > 0 for i in range(3,9)):
            primary, category, priority = current, 'incumbent_other_failure', 3
        else:
            primary, category, priority = current, 'preserve_success_control', 4
        b = bundle(primary)
        b['paired_context'] = dict(category=category, priority=priority,
            iteration=iteration, incumbent_candidate_id=str(current.harness_id),
            recent_candidate_ids=sorted(recent.harness_id.astype(str).tolist()),
            recent_regression_candidate_ids=sorted(regressions.harness_id.astype(str).tolist()))
        b['comparison_reports'] = [reference('incumbent', current, b)]
        if str(base.harness_id) != str(current.harness_id):
            b['comparison_reports'].append(reference('M3', base, b))
        if not recent.empty and str(primary.harness_id) not in set(recent.harness_id.astype(str)):
            b['comparison_reports'].append(reference('recent_candidate', recent.sort_values('harness_id').iloc[0], b))
        # The explicit comparisons replace the old unpaired M3 duplicate.
        b['M3_reference'] = None
        b['selection_reasons'] = [category, 'unique_task_with_paired_reports']
        selected.append(b)

    selected.sort(key=lambda b:(b['paired_context']['priority'], b['predicted_label'] or '', b['task_id']))
    all_recent = [b['task_id'] for b in selected if b['paired_context']['priority']==0]
    selected = selected[:max_bundles]
    for rank, b in enumerate(selected,1): b['rank']=rank
    coverage = dict(policy='incumbent_recent_unique_task_v1', requested_task_budget=max_bundles,
        incumbent_candidate_id=incumbent_id, iteration=iteration,
        recent_regression_task_ids=all_recent,
        included_task_ids=[b['task_id'] for b in selected],
        omitted_recent_regression_task_ids=sorted(set(all_recent)-{b['task_id'] for b in selected}),
        categories={c:sum(b['paired_context']['category']==c for b in selected)
                    for c in sorted({b['paired_context']['category'] for b in selected})})
    return selected, coverage


def compact_comparisons(bundle, compact_report):
    """Keep only report differences; never copy per-task targets into context."""
    primary = compact_report(bundle.get('report')) or {}
    output = []
    for reference in bundle.get('comparison_reports', []):
        report = compact_report(reference.get('report')) or {}
        delta = {k:v for k,v in report.items() if k != 'evidence' and v != primary.get(k)}
        before = {str(e.get('evidence_id')):e for e in primary.get('evidence', [])}
        changes = []
        for e in report.get('evidence', []):
            eid=str(e.get('evidence_id'))
            fields={k:v for k,v in e.items() if k!='evidence_id' and v != before.get(eid,{}).get(k)}
            if fields: changes.append(dict(evidence_id=eid, changed_fields=fields))
        removed=sorted(set(before)-{str(e.get('evidence_id')) for e in report.get('evidence',[])})
        if changes: delta['evidence_changes']=changes
        if removed: delta['removed_evidence_ids']=removed
        output.append({**{k:reference.get(k) for k in ['role','method','harness_id','label_correct']},
                       'report_delta_from_primary':delta})
    return output


def pack_task_observations(context):
    """Lossless primary-claim references to the single observation copy.

    Claims with any metadata difference remain verbatim. Output supports and
    narrative always remain visible; only byte-equivalent copied fields share
    the input record. This is context serialization, never report editing.
    """
    observations=context.get('available_observations',[])
    by_id={str(r.get('evidence_id')):r for r in observations}
    fields=('source','field','statistic','direction','value','window','role')
    for claim in (context.get('generated_report') or {}).get('evidence',[]):
        row=by_id.get(str(claim.get('evidence_id')), {})
        if all(k in claim and k in row and claim[k]==row[k] for k in fields):
            for k in fields:claim.pop(k)
            claim['copy_observation_fields']=True
    # Task ID is already on the containing bundle. Optional auxiliaries are
    # omitted only when absent or exactly redundant; temporal values survive.
    for row in observations:
        row.pop('task_id',None)
        if row.get('raw_value')==row.get('value'):row.pop('raw_value',None)
        if row.get('event_time') in [None,'NaT','']:row.pop('event_time',None)
        if row.get('strength')==0:row.pop('strength',None)
    return context
