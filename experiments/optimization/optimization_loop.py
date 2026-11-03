"""
Iterative optimization loop for domain-constrained M4 harness search.

This module implements the paper-level iterative optimization procedure:

    seed search traces
    -> for t = 1..T:
        select representative manufacturing failure traces
        ask the Proposer for K complete constrained M4 patches
        validate candidates
        screen candidates
        evaluate screened M4 candidates on search split/subset
        add candidate traces to the optimization repository
    -> export final screened candidates for validation

Important:
- Only search-split traces are used during optimization.
- Validation/test results are never fed back into this loop.
- Datasets, gold labels, evaluator, metric definitions, proxy evidence builder,
  test split, candidate labels, and hidden supports annotations remain frozen.
- M4 patches are generated and evaluated under guardrail validation.

This module is an orchestrator. It calls existing modules:
    failure_selector.py
    harness_proposer.py
    candidate_validator.py
    candidate_screening.py
    run_search_candidate.py

Typical pilot usage:

    python -m experiments.optimization.optimization_loop ^
      --seed-search-root outputs/search/api_small_test ^
      --output-root outputs/optimization_loop/pilot_iterative_opt ^
      --mode api ^
      --iterations 4 ^
      --ai4i-limit 1 ^
      --pdm-limit 1 ^
      --candidates-per-iteration 3 ^
      --max-candidates 3 ^
      --max-api-calls 100 ^
      --max-usd 3 ^
      --budget-ledger outputs/optimization_loop/budget.json ^
      --datasets AI4I ^
      --failure-ids F2,F5,F7,F10

Outputs:
    outputs/optimization_loop/<run_name>/
    ├─ loop_config.json
    ├─ loop_summary.csv
    ├─ loop_summary.json
    ├─ combined_candidate_metrics.csv
    ├─ combined_candidate_failure_traces.csv
    ├─ final_candidate_ranking.csv
    ├─ final_selected_candidate.json
    ├─ final_selected_candidate_patches/
    └─ iter_01/
       ├─ selected_failures/
       ├─ proposals/
       ├─ proposer_candidates/
       ├─ validated_candidates/
       ├─ screened_candidates/
       └─ candidate_search/
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.optimization.failure_selector import (
    FailureSelectionConfig,
    select_failure_traces,
    write_selected_csv,
    write_selected_jsonl,
    write_summary_json,
    make_selection_summary,
)
from experiments.optimization.harness_proposer import propose_harness_patch
from experiments.optimization.candidate_validator import validate_candidate_patches
from experiments.optimization.candidate_screening import (
    screen_candidates,
    write_screening_outputs,
    patch_signature,
)
from experiments.optimization.trace_repository import TraceRepository, load_trace_repository
from experiments.runners.run_search_candidate import (
    find_patch_files,
    run_search_candidate,
)
from experiments.utils.config_utils import (
    ExperimentConfigs,
    load_experiment_configs,
    get_split_sizes,
)
from experiments.runners.dataset_filter import parse_datasets_arg
from experiments.optimization.candidate_acceptance import (
    guard_candidate_summary, project_search_feedback, diagnostic_summary,
)
from experiments.optimization.selection_metrics import compute_j_mfg_for_group


# ============================================================
# Data classes
# ============================================================

@dataclass(frozen=True)
class OptimizationLoopConfig:
    seed_search_root: str
    output_root: str
    mode: str
    iterations: int
    ai4i_n: int
    pdm_n: int
    split: str = "search"
    datasets: list[str] = field(default_factory=lambda: ["AI4I", "PdM"])

    failure_ids: list[str] | None = None
    max_failure_bundles: int = 20

    candidates_per_iteration: int = 3
    max_candidates: int = 3
    max_api_calls: int | None = None
    max_cost_usd: float | None = None
    budget_ledger_path: str | None = None
    task_ids_by_dataset: dict[str, list[str]] | None = None

    candidate_prefix: str = "M4_candidate"
    proposer_mode: str = "llm"
    model_profile_name: str | None = None
    require_proxy_evidence: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    repository_root: str
    selected_failures_path: str
    proposal_patch_path: str
    variant_dir: str
    validated_dir: str
    screened_dir: str
    candidate_search_dir: str
    n_selected_failures: int
    n_generated_variants: int
    n_valid_candidates: int
    n_screened_candidates: int
    best_candidate_id: str | None
    best_candidate_j_mfg: float | None
    best_candidate_patch_path: str | None
    proposal_records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# I/O helpers
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def copy_file(src: str | Path, dst: str | Path) -> None:
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def parse_comma_list(value: str | None) -> list[str] | None:
    if value is None:
        return None

    values = [v.strip() for v in value.split(",") if v.strip()]
    return values or None


# ============================================================
# Repository merge helpers
# ============================================================

def load_csv_if_exists(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def merge_metrics_for_next_repository(
    *,
    output_root: Path,
    seed_search_root: Path,
    candidate_search_roots: list[Path],
) -> Path:
    """
    Build a cumulative repository usable by TraceRepository.

    It merges:
    - seed search metrics/failure traces/reports/traces
    - all candidate search metrics/failure traces/reports/traces so far

    The merged repository is written to:
        output_root/current_repository/
    """
    repo_dir = ensure_dir(output_root / "current_repository")

    # CSV merge
    metrics_frames = []
    failure_frames = []
    summary_frames = []

    seed_metrics = load_csv_if_exists(seed_search_root / "metrics_by_task.csv")
    if not seed_metrics.empty:
        seed_metrics["repository_source"] = "seed_search"
        metrics_frames.append(seed_metrics)

    seed_failure = load_csv_if_exists(seed_search_root / "failure_traces.csv")
    if not seed_failure.empty:
        seed_failure["repository_source"] = "seed_search"
        failure_frames.append(seed_failure)

    seed_summary = load_csv_if_exists(seed_search_root / "search_summary.csv")
    if not seed_summary.empty:
        seed_summary["repository_source"] = "seed_search"
        summary_frames.append(seed_summary)

    for candidate_root in candidate_search_roots:
        metrics = load_csv_if_exists(candidate_root / "metrics_by_task.csv")
        if not metrics.empty:
            metrics["repository_source"] = candidate_root.name
            metrics_frames.append(metrics)

        failure = load_csv_if_exists(candidate_root / "failure_traces.csv")
        if not failure.empty:
            failure["repository_source"] = candidate_root.name
            failure_frames.append(failure)

        summary = load_csv_if_exists(candidate_root / "candidate_search_summary.csv")
        if not summary.empty:
            summary["repository_source"] = candidate_root.name
            summary_frames.append(summary)

    merged_metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    merged_failures = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame()
    merged_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()

    merged_metrics.to_csv(repo_dir / "metrics_by_task.csv", index=False)
    merged_failures.to_csv(repo_dir / "failure_traces.csv", index=False)
    merged_summary.to_csv(repo_dir / "search_summary.csv", index=False)

    # JSONL merge
    for filename in ["reports.jsonl", "traces.jsonl"]:
        out_path = repo_dir / filename
        with out_path.open("w", encoding="utf-8") as fout:
            seed_file = seed_search_root / filename
            if seed_file.exists():
                with seed_file.open("r", encoding="utf-8") as fin:
                    for line in fin:
                        if line.strip():
                            fout.write(line)

            for candidate_root in candidate_search_roots:
                candidate_file = candidate_root / filename
                if candidate_file.exists():
                    with candidate_file.open("r", encoding="utf-8") as fin:
                        for line in fin:
                            if line.strip():
                                fout.write(line)

    # Config snapshot
    seed_config = seed_search_root / "config_snapshot.json"
    if seed_config.exists():
        copy_file(seed_config, repo_dir / "config_snapshot.json")
    else:
        write_json(repo_dir / "config_snapshot.json", {"source": "merged_repository"})

    return repo_dir


# ============================================================
# Summary helpers
# ============================================================

def make_loop_summary(records: list[IterationRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.to_dict() for record in records])


def make_final_candidate_ranking(candidate_search_roots: list[Path]) -> pd.DataFrame:
    frames = []

    for root in candidate_search_roots:
        summary_path = root / "candidate_search_summary.csv"
        if not summary_path.exists():
            continue

        df = pd.read_csv(summary_path)
        if df.empty:
            continue

        df["candidate_search_root"] = str(root)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    ranking = pd.concat(frames, ignore_index=True)
    if 'diagnostic_eligible' not in ranking:
        raise ValueError('Candidate ranking lacks paired M3 acceptance results')
    ranking=ranking.loc[ranking.diagnostic_eligible.astype(str).str.lower().eq('true')].copy()

    ranking = ranking.sort_values(
        by=["J_mfg", "unsafe_recommendation_rate", "unsupported_claim_rate", "cost_per_report"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    ranking["global_rank"] = range(1, len(ranking) + 1)
    return ranking


def select_final_candidate(ranking_df: pd.DataFrame) -> dict[str, Any]:
    if ranking_df.empty:
        return dict(method='M3',selection_scope='search_candidate_pool',
                    status='M3_fallback_no_eligible_M4',patch_path=None)

    row = ranking_df.iloc[0].to_dict()
    row["selection_scope"] = "search_candidate_pool"
    return row


def copy_final_selected_patch(
    *,
    selected_candidate: dict[str, Any],
    output_root: Path,
) -> Path | None:
    patch_path = selected_candidate.get("patch_path")
    if not patch_path:
        return None

    patch_path = Path(patch_path)
    if not patch_path.exists():
        return None

    out_dir = ensure_dir(output_root / "final_selected_candidate_patches")
    dst = out_dir / patch_path.name
    shutil.copy2(patch_path, dst)
    return dst


# ============================================================
# Iteration steps
# ============================================================

def run_failure_selection_step(
    *,
    repo: TraceRepository,
    output_dir: Path,
    failure_ids: list[str] | None,
    max_failure_bundles: int,
    configs: ExperimentConfigs | None = None,
    datasets: list[str] | None = None,
    task_ids: list[str] | None = None,
    incumbent_candidate_id: str | None = None,
    current_iteration: int | None = None,
) -> tuple[Path, int]:
    if incumbent_candidate_id is not None:
        from experiments.optimization.incumbent_feedback import select_incumbent_context
        selected, coverage = select_incumbent_context(repo,
            incumbent_id=incumbent_candidate_id, iteration=current_iteration,
            max_bundles=max_failure_bundles, task_ids=task_ids, datasets=datasets)
        output_dir.mkdir(parents=True, exist_ok=True)
        path=output_dir/'selected_failure_bundles.jsonl'
        path.write_text(''.join(json.dumps(b,ensure_ascii=False,default=str)+'\n' for b in selected),encoding='utf-8')
        write_json(output_dir/'selection_summary.json',coverage)
        if coverage['omitted_recent_regression_task_ids']:
            raise ValueError('Task context budget cannot cover all recent label regressions; increase the offline context budget before proposing')
        pd.DataFrame([dict(task_id=b['task_id'],method=b['method'],harness_id=b['harness_id'],
            category=b['paired_context']['category'],rank=b['rank']) for b in selected]).to_csv(output_dir/'selected_failure_bundles.csv',index=False)
        return path,len(selected)
    sampling=configs.experiment['optimization'].get('failure_trace_sampling',{}) if configs else {}
    selected = select_failure_traces(
        repo,
        FailureSelectionConfig(
            max_bundles=max_failure_bundles,
            datasets=datasets,
            task_ids=task_ids,
            include_failure_ids=failure_ids,
            include_report=True,
            max_per_task=int(sampling.get('max_per_task',2)),
            min_label_error_bundles=min(max_failure_bundles,int(sampling.get('min_label_error_bundles',4))),
            min_verified_bundles=min(max_failure_bundles,int(sampling.get('min_verified_bundles',10))),
        ),
    )

    selected_jsonl = output_dir / "selected_failure_bundles.jsonl"
    selected_csv = output_dir / "selected_failure_bundles.csv"
    summary_json = output_dir / "selection_summary.json"

    write_selected_jsonl(selected, selected_jsonl)
    write_selected_csv(selected, selected_csv)
    write_summary_json(make_selection_summary(selected), summary_json)

    return selected_jsonl, len(selected)


def run_proposer_step(
    *,
    selected_failures_path: Path,
    output_dir: Path,
    configs: ExperimentConfigs,
    iteration: int,
    proposer_mode: str,
    current_harness: dict[str, Any] | None = None,
    search_feedback: dict[str, Any] | None = None,
    candidate_id: str | None = None,
) -> Path:
    candidate_id = candidate_id or f"M4_base_proposal_iter{iteration:02d}"

    result = propose_harness_patch(
        selected_failures_path=selected_failures_path,
        output_dir=output_dir,
        configs=configs,
        candidate_id=candidate_id,
        mode=proposer_mode,
        current_harness=current_harness,
        search_feedback=search_feedback,
    )

    return output_dir / f"{result.candidate_id}_patch.json"



def run_candidate_validation_step(
    *,
    patch_dir: Path,
    output_dir: Path,
) -> None:
    validate_candidate_patches(
        patch_dir,
        output_dir=output_dir,
        strict=True,
    )


def run_candidate_screening_step(
    *,
    patch_dir: Path,
    validation_dir: Path,
    output_dir: Path,
    max_candidates: int,
) -> int:
    records = screen_candidates(
        patch_dir_or_file=patch_dir,
        validation_dir_or_file=validation_dir,
        max_candidates=max_candidates,
        allow_warnings=True,
        keep_duplicates=False,
    )

    write_screening_outputs(
        records=records,
        output_dir=output_dir,
        copy_selected_patches=True,
    )

    return sum(1 for record in records if record.selected)


def run_candidate_search_step(
    *,
    configs: ExperimentConfigs,
    mode: str,
    patch_dir: Path,
    output_dir: Path,
    split: str,
    ai4i_n: int,
    pdm_n: int,
    model_profile_name: str | None,
    require_proxy_evidence: bool,
    iteration: int,
    datasets_to_run: set[str],
    seen_signatures: set[str] | None = None,
    task_ids_by_dataset: dict[str, list[str]] | None = None,
) -> int:
    patch_files = find_patch_files(patch_dir)
    seen=seen_signatures if seen_signatures is not None else set()
    unique=[];duplicates=[]
    for file in patch_files:
        signature=patch_signature(read_json(file))
        if signature in seen:duplicates.append(dict(patch_path=str(file),signature=signature))
        else:seen.add(signature);unique.append(file)
    write_json(output_dir/'effective_duplicate_candidates.json',duplicates)
    if not unique:return 0

    run_search_candidate(
        mode=mode,
        configs=configs,
        patch_files=unique,
        split=split,
        ai4i_n=ai4i_n,
        pdm_n=pdm_n,
        output_dir=output_dir,
        model_profile_name=model_profile_name,
        require_proxy_evidence=require_proxy_evidence,
        iteration=iteration,
        datasets_to_run=datasets_to_run,
        task_ids_by_dataset=task_ids_by_dataset,
    )
    return len(unique)


# ============================================================
# Main loop
# ============================================================

def run_optimization_loop(*, loop_config: OptimizationLoopConfig, configs: ExperimentConfigs) -> None:
    from experiments.optimization.loop_execution import execute_proposer_loop
    return execute_proposer_loop(loop_config=loop_config, configs=configs)


def count_valid_candidates(validated_dir: Path) -> int:
    csv_path = validated_dir / "candidate_validation_results.csv"
    if not csv_path.exists():
        return 0

    df = pd.read_csv(csv_path)
    if df.empty or "valid" not in df.columns:
        return 0

    return int(df["valid"].astype(bool).sum())


def write_loop_outputs(
    *,
    output_root: Path,
    iteration_records: list[IterationRecord],
    candidate_search_roots: list[Path],
) -> None:
    write_proposal_cost_summary(output_root)
    loop_summary_df = make_loop_summary(iteration_records)
    loop_summary_df.to_csv(output_root / "loop_summary.csv", index=False)

    write_json(
        output_root / "loop_summary.json",
        [record.to_dict() for record in iteration_records],
    )

    # Combined metrics/failures
    metric_frames = []
    failure_frames = []

    for root in candidate_search_roots:
        metrics = load_csv_if_exists(root / "metrics_by_task.csv")
        if not metrics.empty:
            metrics["candidate_search_root"] = str(root)
            metric_frames.append(metrics)

        failures = load_csv_if_exists(root / "failure_traces.csv")
        if not failures.empty:
            failures["candidate_search_root"] = str(root)
            failure_frames.append(failures)

    combined_metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    combined_failures = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame()

    combined_metrics.to_csv(output_root / "combined_candidate_metrics.csv", index=False)
    combined_failures.to_csv(output_root / "combined_candidate_failure_traces.csv", index=False)

    ranking = make_final_candidate_ranking(candidate_search_roots)
    ranking.to_csv(output_root / "final_candidate_ranking.csv", index=False)

    selected = select_final_candidate(ranking)
    selected['selection_scope'] = 'search_only_preliminary; final harness selection requires validation'
    write_json(output_root / "final_selected_candidate.json", selected)

    copied_patch = copy_final_selected_patch(
        selected_candidate=selected,
        output_root=output_root,
    )

    if copied_patch is not None:
        selected["copied_patch_path"] = str(copied_patch)
        write_json(output_root / "final_selected_candidate.json", selected)


# ============================================================
# CLI
# ============================================================

def write_proposal_cost_summary(output_root: Path) -> None:
    """Include accepted and rejected proposals; keep deployment cost separate."""
    rows=[]
    for p in sorted(output_root.glob('iter_*/proposals/*_proposal_result.json')):
        r=json.loads(p.read_text(encoding='utf-8'))
        rows.append({k:r.get(k) for k in ['candidate_id','proposer_mode','requested_model','model_name','accepted','input_tokens','output_tokens','cost_usd','latency_sec']})
    pd.DataFrame(rows,columns=['candidate_id','proposer_mode','requested_model','model_name','accepted','input_tokens','output_tokens','cost_usd','latency_sec']).to_csv(output_root/'proposer_costs.csv',index=False)
    write_json(output_root/'proposer_cost_summary.json',{'proposals':len(rows),'llm_responses':sum(r['proposer_mode']=='llm' for r in rows),
        'input_tokens':sum(r['input_tokens'] or 0 for r in rows),'output_tokens':sum(r['output_tokens'] or 0 for r in rows),
        'cost_usd':sum(r['cost_usd'] or 0 for r in rows),'unknown_usage_proposals':sum(r['cost_usd'] is None for r in rows),
        'cost_scope':'Offline proposal cost only; candidate search/validation and final report generation are logged separately.'})

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run iterative M4 harness optimization loop on search split."
    )

    parser.add_argument(
        "--seed-search-root",
        required=True,
        help="Seed search output directory from run_search.py.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output directory for the optimization loop.",
    )
    parser.add_argument(
        "--mode",
        choices=["dummy", "api"],
        default="dummy",
        help="dummy = no API call; api = actual OpenAI API call.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of optimization iterations T.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Search split name. Default: optimization.search_split from config.",
    )
    parser.add_argument(
        "--ai4i-limit",
        type=int,
        default=None,
        help="Number of AI4I search tasks per candidate evaluation.",
    )
    parser.add_argument(
        "--pdm-limit",
        type=int,
        default=None,
        help="Number of Azure PdM search tasks per candidate evaluation.",
    )
    parser.add_argument(
        "--failure-ids",
        default="F1,F2,F3,F4,F5,F6,F7,F8,F9,F10",
        help="Comma-separated failure IDs for failure selection.",
    )
    parser.add_argument(
        "--max-failure-bundles",
        type=int,
        default=20,
        help="Max selected failure bundles per iteration.",
    )
    parser.add_argument('--candidates-per-iteration', type=int, default=None,
                        help='K complete Proposer-owned patches per iteration; config default 3.')
    parser.add_argument('--max-api-calls', type=int, default=None)
    parser.add_argument('--max-usd', type=float, default=None)
    parser.add_argument('--budget-ledger', default=None,
                        help='Shared persistent allocation ledger; required in API mode.')
    parser.add_argument('--task-manifest', default=None,
                        help='JSON optimization_task_ids mapping for a fixed search subset.')
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Maximum screened candidates evaluated per iteration; must cover K.",
    )
    parser.add_argument(
        "--candidate-prefix",
        default="M4_candidate",
        help="Prefix for generated candidate IDs.",
    )
    parser.add_argument(
        "--proposer-mode",
        choices=["rule_based", "dry_run", "llm"],
        default="llm",
        help="Harness proposer mode.",
    )
    parser.add_argument(
        "--model-profile",
        default=None,
        help="Optional model profile name.",
    )
    parser.add_argument(
        "--allow-tasks-without-evidence",
        action="store_true",
        help="If set, candidate search may include tasks without proxy evidence.",
    )
    parser.add_argument(
        "--datasets",
        default="AI4I,PdM",
        help="Comma-separated dataset filter: AI4I, PdM, or AI4I,PdM.",
    )

    args = parser.parse_args()

    datasets_to_run = parse_datasets_arg(args.datasets)

    configs = load_experiment_configs()

    split = args.split or configs.experiment.get("optimization", {}).get("search_split", "search")
    split_sizes = get_split_sizes(configs, split)

    ai4i_n = args.ai4i_limit if args.ai4i_limit is not None else int(split_sizes["ai4i_n"])
    pdm_n = args.pdm_limit if args.pdm_limit is not None else int(split_sizes["pdm_n"])

    ids = None
    if args.task_manifest:
        manifest = read_json(args.task_manifest)
        ids = manifest.get('optimization_task_ids', manifest)
        if args.ai4i_limit is None and 'AI4I' in ids: ai4i_n = len(ids['AI4I'])
        if args.pdm_limit is None and 'PdM' in ids: pdm_n = len(ids['PdM'])
    opt = configs.experiment['optimization']
    k = args.candidates_per_iteration if args.candidates_per_iteration is not None else int(opt['candidates_per_iteration'])
    loop_config = OptimizationLoopConfig(
        seed_search_root=args.seed_search_root,
        output_root=args.output_root,
        mode=args.mode,
        iterations=args.iterations if args.iterations is not None else int(opt["iterations"]),
        ai4i_n=ai4i_n,
        pdm_n=pdm_n,
        split=split,
        failure_ids=parse_comma_list(args.failure_ids),
        max_failure_bundles=args.max_failure_bundles,
        candidates_per_iteration=k,
        max_candidates=args.max_candidates if args.max_candidates is not None else k,
        max_api_calls=args.max_api_calls, max_cost_usd=args.max_usd,
        budget_ledger_path=args.budget_ledger, task_ids_by_dataset=ids,
        candidate_prefix=args.candidate_prefix,
        proposer_mode=args.proposer_mode,
        model_profile_name=args.model_profile,
        require_proxy_evidence=not args.allow_tasks_without_evidence,
        datasets=sorted(datasets_to_run),
    )

    run_optimization_loop(
        loop_config=loop_config,
        configs=configs,
    )


if __name__ == "__main__":
    main()
