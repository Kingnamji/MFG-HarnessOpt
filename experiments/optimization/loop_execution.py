"""Proposer-owned T x K search, using the common loop I/O and evaluator.

Each slot makes one complete proposal. No deterministic variant generator is
used, and no executable patch content is added or rewritten by this module.
"""
import copy
import json
from pathlib import Path
import pandas as pd

from experiments.optimization.candidate_acceptance import (
    project_search_feedback, diagnostic_summary, guard_candidate_summary,
)
from experiments.optimization.candidate_screening import patch_signature
from experiments.optimization.selection_metrics import compute_j_mfg_for_group
from experiments.utils.api_budget import RequestBudget, current_api_budget


def execute_proposer_loop(*, loop_config, configs):
    c = loop_config
    if c.proposer_mode == 'llm' and c.mode != 'api':
        raise ValueError('LLM proposer requires mode=api. For zero-API checks inject a fake SDK or explicitly select a legacy dry_run proposer.')
    if c.split != 'search':
        raise ValueError('Optimization is restricted to search split')
    if c.mode not in {'api', 'dummy'} or not c.datasets or set(c.datasets) - {'AI4I', 'PdM'}:
        raise ValueError('Invalid loop mode or datasets')
    if c.mode == 'api' and len(c.datasets) != 1:
        raise ValueError('Run dataset-specific optimization separately with a shared budget ledger')
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
           for x in [c.iterations, c.candidates_per_iteration, c.max_candidates]):
        raise ValueError('T, K and screening capacity must be positive integers')
    if c.max_candidates < c.candidates_per_iteration:
        raise ValueError('Screening capacity must cover all K proposed candidates')
    opt = configs.experiment['optimization']
    if c.candidates_per_iteration > int(opt['max_new_candidates_per_iteration']):
        raise ValueError('K exceeds the configured per-iteration candidate limit')
    maximum = int(opt['max_total_candidates']) - len(opt['seed_methods'])
    if c.iterations*c.candidates_per_iteration > maximum:
        raise ValueError('T x K exceeds the explicit total new-candidate limit')
    if int(configs.experiment['m3_fixed_repair']['max_repair_steps']) != 1:
        raise ValueError('The Paper-aligned M3 repair bound must remain one')
    if c.mode != 'api':
        return _execute(c, configs)
    if c.max_api_calls is None or c.max_cost_usd is None or not c.budget_ledger_path:
        raise ValueError('API mode requires explicit call/USD limits and a shared budget ledger path')
    active = current_api_budget()
    if active is not None:
        if (active.path != Path(c.budget_ledger_path).resolve()
                or active.max_calls != c.max_api_calls or active.max_usd != c.max_cost_usd):
            raise ValueError('Loop allocation must match the already active shared budget')
        return _execute(c, configs)
    with RequestBudget(c.budget_ledger_path, max_calls=c.max_api_calls,
                       max_usd=c.max_cost_usd).activate():
        return _execute(c, configs)


def _execute(c, configs):
    # Import lazily so these public I/O boundaries remain injectable in offline
    # integration tests and command-line module execution has no import cycle.
    from experiments.optimization import optimization_loop as io
    root, seed = Path(c.output_root), Path(c.seed_search_root)
    if not seed.exists():
        raise FileNotFoundError(f'Seed search root not found: {seed}')
    if (root/'loop_config.json').exists():
        raise FileExistsError('Use a fresh optimization output directory')
    root.mkdir(parents=True, exist_ok=True)
    baseline = io.load_csv_if_exists(seed/'metrics_by_task.csv')
    if baseline.empty or 'method' not in baseline:
        raise ValueError('New M3 search seed metrics are required')
    baseline = baseline.loc[baseline.method.eq('M3')].copy()
    wanted = []
    for ds, key, n in [('AI4I', 'ai4i', c.ai4i_n), ('PdM', 'pdm', c.pdm_n)]:
        if ds not in c.datasets:
            continue
        manifest = pd.read_csv(configs.paths['dataset'][key]['tasks'], usecols=['task_id','split'])
        if c.task_ids_by_dataset is not None and ds not in c.task_ids_by_dataset:
            raise ValueError('Explicit task manifest is missing the selected dataset')
        ids = c.task_ids_by_dataset.get(ds) if c.task_ids_by_dataset is not None else None
        if ids is None:
            ids = manifest.loc[manifest.split.eq('search'), 'task_id'].head(n).astype(str).tolist()
        if len(ids) != n or len(set(ids)) != n or not set(ids).issubset(
                set(manifest.loc[manifest.split.eq('search'), 'task_id'].astype(str))):
            raise ValueError('Explicit loop task manifest must contain exactly N unique search tasks')
        wanted.extend(ids)
    baseline = baseline.loc[baseline.task_id.astype(str).isin(wanted)]
    if set(baseline.task_id.astype(str)) != set(wanted):
        raise ValueError('M3 reference does not cover the exact candidate task manifest')
    project_search_feedback(baseline, pd.DataFrame(), configs)
    if c.mode == 'api':
        # An old M3 report skipped the newly required check/repair opportunity;
        # it cannot be a fair baseline for candidates using the revised harness.
        from experiments.evaluation.runtime_verifier import RUNTIME_VERIFIER_VERSION
        seed_repo=io.load_trace_repository(seed)
        for row in baseline.itertuples():
            trace=seed_repo.get_trace_row(str(row.task_id),'M3',str(row.harness_id)) or {}
            verified=(trace.get('raw_response') or {}).get('final_runtime_verification') or {}
            if verified.get('runtime_version') != RUNTIME_VERIFIER_VERSION:
                raise ValueError('Fresh M3 seed reports under the current runtime verifier are required; historical pilot outputs remain audit-only')
    budget = current_api_budget()
    if budget:
        budget.require_capacity(c.iterations*c.candidates_per_iteration*(len(wanted)+1))
    io.write_json(root/'loop_config.json', {**c.to_dict(),
        'candidate_generation':'proposer_only', 'effective_task_ids':wanted,
        'maximum_new_candidates':c.iterations*c.candidates_per_iteration})
    incumbent = {'base_method':'M3','configuration':configs.methods['M3']}
    incumbent_score = compute_j_mfg_for_group(baseline, configs)
    io.write_json(root/'M3_search_reference.json', dict(method='M3',J_mfg=incumbent_score,
        metrics=diagnostic_summary(baseline,configs),
        acceptance_policy=configs.experiment.get('candidate_acceptance',{})))
    search_roots, records, rejected = [], [], []
    seen = set()
    repo_root = io.merge_metrics_for_next_repository(output_root=root,
        seed_search_root=seed,candidate_search_roots=search_roots)
    try:
        for iteration in range(1,c.iterations+1):
            folder = io.ensure_dir(root/f'iter_{iteration:02d}')
            selected_dir = io.ensure_dir(folder/'selected_failures')
            proposal_dir = io.ensure_dir(folder/'proposals')
            candidate_dir = io.ensure_dir(folder/'proposer_candidates')
            validated = io.ensure_dir(folder/'validated_candidates')
            screened = io.ensure_dir(folder/'screened_candidates')
            searched = io.ensure_dir(folder/'candidate_search')
            selected, count = io.run_failure_selection_step(
                repo=io.load_trace_repository(repo_root),output_dir=selected_dir,
                failure_ids=c.failure_ids,max_failure_bundles=c.max_failure_bundles,
                configs=configs,datasets=baseline.dataset.unique().tolist(),
                task_ids=baseline.task_id.astype(str).tolist(),
                incumbent_candidate_id=incumbent.get('candidate_id','M3'),current_iteration=iteration)
            if not count:
                io.write_json(folder/'stop_reason.json', {'reason':'no_observed_search_failures'})
                break
            history = io.load_csv_if_exists(repo_root/'metrics_by_task.csv')
            history = history.loc[history.method.eq('M4')] if not history.empty else pd.DataFrame()
            feedback = project_search_feedback(baseline, history, configs,
                incumbent_id=incumbent.get('candidate_id','M3'))
            feedback['candidates'].extend(copy.deepcopy(rejected))
            io.write_json(folder/'search_feedback.json',feedback)
            attempts, peers = [], []
            for slot in range(1,c.candidates_per_iteration+1):
                cid = f'{c.candidate_prefix}_iter{iteration:02d}_candidate{slot:02d}'
                request = copy.deepcopy(feedback)
                request['proposal_request'] = dict(iteration=iteration,slot=slot,
                    candidates_per_iteration=c.candidates_per_iteration)
                request['earlier_proposals'] = copy.deepcopy(peers)
                entry = dict(candidate_id=cid,slot=slot,proposer_mode=c.proposer_mode,
                    parent_candidate_id=incumbent.get('candidate_id','M3'),
                    feedback_candidate_ids=[v['candidate_id'] for v in feedback['candidates']])
                try:
                    if budget:
                        with budget.at(dataset=c.datasets[0],method='Proposer',iteration=iteration,candidate_id=cid):
                            path = io.run_proposer_step(selected_failures_path=selected,
                                output_dir=proposal_dir,configs=configs,iteration=iteration,
                                proposer_mode=c.proposer_mode,current_harness=incumbent,
                                search_feedback=request,candidate_id=cid)
                    else:
                        path = io.run_proposer_step(selected_failures_path=selected,
                            output_dir=proposal_dir,configs=configs,iteration=iteration,
                            proposer_mode=c.proposer_mode,current_harness=incumbent,
                            search_feedback=request,candidate_id=cid)
                except ValueError:
                    result_path = proposal_dir/f'{cid}_proposal_result.json'
                    if not result_path.exists():
                        raise
                    result = io.read_json(result_path)
                    if result.get('accepted') is not False or not result.get('validation_errors'):
                        raise
                    entry.update(status='proposal_rejected',errors=result['validation_errors'])
                    rejected.append(dict(candidate_id=cid,eligible=False,
                        rejection_reasons=result['validation_errors'],metrics=[]))
                else:
                    patch = io.read_json(path)
                    destination = candidate_dir/path.name
                    # Copy bytes; not even candidate IDs or guardrail fields are rewritten.
                    io.copy_file(Path(path),destination)
                    entry.update(status='proposed',patch_path=str(destination),
                                 patch_signature=patch_signature(patch))
                    peers.append(dict(candidate_id=cid,patch_signature=entry['patch_signature'],
                                      prompt_instructions=patch.get('prompt_instructions',{})))
                attempts.append(entry)
                io.write_json(folder/'proposal_manifest.json',attempts)
            n_valid = n_screened = 0
            best_id = best_score = best_path = None
            if peers:
                io.run_candidate_validation_step(patch_dir=candidate_dir,output_dir=validated)
                n_valid = io.count_valid_candidates(validated)
                n_screened = io.run_candidate_screening_step(patch_dir=candidate_dir,
                    validation_dir=validated,output_dir=screened,max_candidates=c.max_candidates)
                screening_summary = io.read_json(screened/'screening_summary.json')
                for cid, original in screening_summary.get('duplicates',{}).items():
                    rejected.append(dict(candidate_id=cid,eligible=False,
                        rejection_reasons=[f'effective_duplicate_of:{original}'],metrics=[]))
                if n_screened:
                    n_screened = io.run_candidate_search_step(configs=configs,mode=c.mode,
                        patch_dir=screened/'selected_candidate_patches',output_dir=searched,
                        split='search',ai4i_n=c.ai4i_n,pdm_n=c.pdm_n,
                        model_profile_name=c.model_profile_name,require_proxy_evidence=c.require_proxy_evidence,
                        iteration=iteration,datasets_to_run=set(c.datasets),seen_signatures=seen,
                        task_ids_by_dataset=c.task_ids_by_dataset)
                    duplicate_file = searched/'effective_duplicate_candidates.json'
                    if duplicate_file.exists():
                        for duplicate in io.read_json(duplicate_file):
                            cid = io.read_json(duplicate['patch_path'])['candidate_id']
                            rejected.append(dict(candidate_id=cid,eligible=False,
                                rejection_reasons=['effective_duplicate_of_previously_evaluated_patch'],metrics=[]))
                    summary = io.load_csv_if_exists(searched/'candidate_search_summary.csv')
                    if not summary.empty:
                        metrics = io.load_csv_if_exists(searched/'metrics_by_task.csv')
                        summary = guard_candidate_summary(summary,metrics,baseline,configs,scope='search')
                        summary.to_csv(searched/'candidate_search_summary.csv',index=False)
                        eligible = summary.loc[summary.diagnostic_eligible].sort_values(
                            ['J_mfg','unsafe_recommendation_rate','unsupported_claim_rate','cost_per_report'],
                            ascending=[False,True,True,True])
                        if not eligible.empty:
                            best = eligible.iloc[0]
                            best_id,best_score,best_path = str(best.candidate_id),float(best.J_mfg),str(best.patch_path)
                            if best_score > incumbent_score:
                                incumbent_score,incumbent = best_score,io.read_json(best_path)
            io.write_json(folder/'proposal_manifest.json',attempts)
            if (searched/'metrics_by_task.csv').exists():
                search_roots.append(searched)
            records.append(io.IterationRecord(iteration=iteration,repository_root=str(repo_root),
                selected_failures_path=str(selected),proposal_patch_path=str(proposal_dir),
                variant_dir=str(candidate_dir),validated_dir=str(validated),screened_dir=str(screened),
                candidate_search_dir=str(searched),n_selected_failures=count,
                n_generated_variants=len(peers),n_valid_candidates=n_valid,n_screened_candidates=n_screened,
                best_candidate_id=best_id,best_candidate_j_mfg=best_score,
                best_candidate_patch_path=best_path,proposal_records=attempts))
            repo_root = io.merge_metrics_for_next_repository(output_root=root,
                seed_search_root=seed,candidate_search_roots=search_roots)
            io.write_loop_outputs(output_root=root,iteration_records=records,candidate_search_roots=search_roots)
        io.write_json(root/'run_status.json',dict(status='completed',incumbent_candidate_id=incumbent.get('candidate_id','M3'),
            completed_iterations=len(records),new_API_calls_measured_by='shared budget ledger; fake SDK in offline tests'))
    except Exception as exc:
        io.write_json(root/'run_status.json',dict(status='stopped',failure_type=type(exc).__name__,
            reason=str(exc),completed_iterations=len(records)))
        raise
    finally:
        io.write_loop_outputs(output_root=root,iteration_records=records,candidate_search_roots=search_roots)
        if budget:
            io.write_json(root/'api_budget_snapshot.json',budget.snapshot())
