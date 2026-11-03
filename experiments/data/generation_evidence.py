"""Observation-only evidence construction and explicit offline target annotation.

Generation builders never read labels or failure flags. Target annotation is a
separate offline operation. Frozen task IDs/splits are supplied by the caller.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROTOCOL_VERSION = "observation_only_v3_azure_context_twf"
GENERATION_COLUMNS = [
    "task_id", "evidence_id", "source", "field", "statistic", "direction",
    "value", "raw_value", "window", "role", "event_time", "strength",
]
HIDDEN_COLUMNS = {"gold_label", "supports", "machine_failure", "twf", "hdf", "pwf", "osf", "rnf"}
TASK_FIELDS = {
    "task_id", "dataset", "split", "udi", "product_id", "type", "air_temperature",
    "process_temperature", "rotational_speed", "torque", "tool_wear", "temperature_gap",
    "mechanical_power_w", "overstrain_score", "machineID", "failure_time", "model", "age",
    "azure_observation_context",
}


def generation_task(task: dict) -> dict:
    from experiments.data.azure_observation_context import CONTEXT_KEY, project_context
    result = {key: value for key, value in task.items() if key in TASK_FIELDS and key != CONTEXT_KEY}
    if CONTEXT_KEY in task:
        result[CONTEXT_KEY] = project_context(task[CONTEXT_KEY], result)
    return result


def visible_evidence(records) -> pd.DataFrame:
    df = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records or [])
    return df[[c for c in GENERATION_COLUMNS if c in df]].copy()


def validate_generation_evidence(evidence: pd.DataFrame, task_ids) -> None:
    if set(evidence) & HIDDEN_COLUMNS:
        raise ValueError("Hidden target columns in generation evidence")
    required = set(GENERATION_COLUMNS)
    if required - set(evidence):
        raise ValueError(f"Missing generation columns: {sorted(required - set(evidence))}")
    if evidence.task_id.isna().any() or evidence.task_id.astype(str).str.strip().eq("").any():
        raise ValueError("Evidence has a missing task_id")
    if not set(evidence.task_id).issubset(set(task_ids)):
        raise ValueError("Evidence references an unknown task_id")
    if evidence.duplicated(["task_id", "evidence_id"]).any():
        raise ValueError("Duplicate task/evidence IDs")
    if evidence.field.isin(HIDDEN_COLUMNS).any():
        raise ValueError("Target field in generation evidence")


def validate_observation_windows(evidence: pd.DataFrame, tasks: pd.DataFrame) -> None:
    """Fail closed before generation; only task IDs and observable anchors read.

    Azure uses half-open [failure_time - horizon, failure_time) intervals.
    Static AI4I rows cannot acquire invented event timestamps.
    """
    if evidence.empty:
        return
    if "failure_time" not in tasks:
        if not evidence.window.isin(["static_instance", "static", "instance"]).all() or evidence.event_time.notna().any():
            raise ValueError("Static evidence has an invented temporal window/timestamp")
        return
    anchors = tasks.set_index("task_id")["failure_time"]
    windows = {"telemetry": ("telemetry_24h_before_failure", pd.Timedelta(hours=24)),
               "errors": ("error_24h_before_failure", pd.Timedelta(hours=24)),
               "maintenance": ("maintenance_30d_before_failure", pd.Timedelta(days=30))}
    for row in evidence.to_dict("records"):
        source = row["source"]
        if source == "machines" and row["window"] == "static" and pd.isna(row["event_time"]):
            continue
        if source not in windows or row["task_id"] not in anchors:
            raise ValueError("Unknown temporal evidence source/task")
        name, delta = windows[source]
        time = pd.to_datetime(row["event_time"], utc=True, errors="coerce")
        cutoff = pd.to_datetime(anchors[row["task_id"]], utc=True, errors="coerce")
        if row["window"] != name or pd.isna(time) or pd.isna(cutoff) or not cutoff-delta <= time < cutoff:
            raise ValueError(f"Observation outside admissible pre-failure window: {row['task_id']}/{row['evidence_id']}")


def finalize_evidence(rows: list[dict], task_ids) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=GENERATION_COLUMNS)
    df = pd.DataFrame(rows)
    if "task_id" not in df or df.task_id.isna().any():
        raise ValueError("Every observation must carry its task_id before assembly")
    if "event_time" not in df:
        df["event_time"] = pd.NaT
    df["event_time"] = pd.to_datetime(df.event_time)
    if "strength" not in df:
        df["strength"] = 0.0
    # All ranking keys are observable. IDs are assigned only after this sort.
    df["_value_key"] = df.value.map(str)
    df = df.sort_values(
        ["task_id", "source", "strength", "field", "event_time", "_value_key"],
        ascending=[True, True, False, True, False, True], kind="stable", na_position="last",
    ).reset_index(drop=True)
    ids = df.groupby("task_id").cumcount().add(1).astype(int)
    df["evidence_id"] = ids.map(lambda i: f"E{i}")
    df = df[GENERATION_COLUMNS]
    validate_generation_evidence(df, task_ids)
    return df


def build_ai4i_generation_evidence(observations: pd.DataFrame) -> pd.DataFrame:
    fields = ["task_id", "type", "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
    observed = observations[fields].copy()
    rows = []
    for task in observed.to_dict("records"):
        gap = float(task["process_temperature"]) - float(task["air_temperature"])
        power = float(task["torque"]) * 2 * np.pi * float(task["rotational_speed"]) / 60
        strain = float(task["torque"]) * float(task["tool_wear"])
        threshold = {"L": 11000, "M": 12000, "H": 13000}[str(task["type"])]
        values = {k: task[k] for k in fields if k != "task_id"}
        values.update(temperature_gap=gap, mechanical_power_w=power, overstrain_score=strain)
        directions = {
            "tool_wear": "high" if float(task["tool_wear"]) >= 200 else "normal",
            "rotational_speed": "low" if float(task["rotational_speed"]) < 1380 else "normal",
            "temperature_gap": "low" if gap < 8.6 else "normal",
            "mechanical_power_w": "low" if power < 3500 else "high" if power > 9000 else "normal",
            "overstrain_score": "high" if strain > threshold else "normal",
        }
        for field, value in values.items():
            derived = field in {"temperature_gap", "mechanical_power_w", "overstrain_score"}
            rows.append({
                "task_id": task["task_id"], "source": "derived_features" if derived else "machine_features",
                "field": field, "statistic": "computed_value" if derived else "observed_value",
                "direction": directions.get(field, "observed"), "value": value, "raw_value": value,
                "window": "static_instance", "role": "contextual" if field == "type" else "symptomatic",
                "event_time": pd.NaT, "strength": 0.0,
            })
    return finalize_evidence(rows, observed.task_id)


def time_window(df, machine, start, end):
    return df[(df.machineID == machine) & (df.datetime >= start) & (df.datetime < end)].copy()


def build_azure_telemetry_evidence(tasks: pd.DataFrame, telemetry: pd.DataFrame):
    summaries, rows = [], []
    observed_tasks = tasks[["task_id", "machineID", "failure_time"]].copy()
    by_machine = {k: g for k, g in telemetry.groupby("machineID", sort=False)}
    for task in observed_tasks.to_dict("records"):
        machine, cutoff = task["machineID"], pd.Timestamp(task["failure_time"])
        data = by_machine.get(machine, telemetry.iloc[:0])
        baseline = time_window(data, machine, cutoff-pd.Timedelta(days=30), cutoff-pd.Timedelta(days=7))
        if baseline.empty:
            baseline = time_window(data, machine, cutoff-pd.Timedelta(days=30), cutoff-pd.Timedelta(hours=24))
        window = time_window(data, machine, cutoff-pd.Timedelta(hours=24), cutoff)
        candidates = []
        for field in ["volt", "rotate", "pressure", "vibration"]:
            hist, samples = baseline[field].dropna(), window[field].dropna()
            summary = {"task_id": task["task_id"], "machineID": machine, "field": field}
            if len(hist) < 2 or samples.empty:
                summary.update({k: np.nan for k in ["baseline_mean", "baseline_std", "evidence_mean", "evidence_min", "evidence_max", "max_abs_z"]})
                summary["direction"] = "unavailable"; summaries.append(summary); continue
            mean, std = float(hist.mean()), float(hist.std(ddof=0))
            # Constant baselines do not define a finite z-score.
            z = (samples-mean)/std if std > 0 else pd.Series(0.0, index=samples.index)
            ix = z.abs().idxmax(); strength = float(abs(z.loc[ix]))
            direction = "high" if float(samples.loc[ix]) >= mean else "low"
            summary.update(baseline_mean=mean, baseline_std=std, evidence_mean=float(samples.mean()), evidence_min=float(samples.min()), evidence_max=float(samples.max()), max_abs_z=strength, direction=direction)
            summaries.append(summary)
            if strength >= 2.0:
                candidates.append({
                    "task_id": task["task_id"], "source": "telemetry", "field": field,
                    "statistic": "max_abs_zscore", "direction": direction, "value": strength,
                    "raw_value": float(samples.loc[ix]), "window": "telemetry_24h_before_failure",
                    "role": "symptomatic", "event_time": window.loc[ix, "datetime"], "strength": strength,
                })
        rows.extend(sorted(candidates, key=lambda r: (-r["strength"], r["field"]))[:2])
    return pd.DataFrame(summaries), rows


def build_azure_event_evidence(tasks: pd.DataFrame, records: pd.DataFrame, source: str) -> list[dict]:
    rows = []
    field = "errorID" if source == "errors" else "comp"
    window_name = "error_24h_before_failure" if source == "errors" else "maintenance_30d_before_failure"
    delta = pd.Timedelta(hours=24) if source == "errors" else pd.Timedelta(days=30)
    for task in tasks[["task_id", "machineID", "failure_time"]].to_dict("records"):
        cutoff = pd.Timestamp(task["failure_time"])
        window = time_window(records, task["machineID"], cutoff-delta, cutoff)
        window = window.sort_values(["datetime", field], ascending=[False, True], kind="stable").drop_duplicates(field).head(3)
        for event in window.to_dict("records"):
            age = (cutoff-pd.Timestamp(event["datetime"]))/delta
            rows.append({
                "task_id": task["task_id"], "source": source, "field": field,
                "statistic": "event_occurrence" if source == "errors" else "recent_record",
                "direction": "occurred" if source == "errors" else "recent",
                "value": str(event[field]), "raw_value": str(event[field]), "window": window_name,
                "role": "symptomatic" if source == "errors" else "contextual",
                "event_time": event["datetime"], "strength": max(0.0, 1.0-float(age)),
            })
    return rows


def build_azure_generation_evidence(tasks, telemetry, errors, maintenance):
    summary, tel = build_azure_telemetry_evidence(tasks, telemetry)
    rows = tel + build_azure_event_evidence(tasks, errors, "errors") + build_azure_event_evidence(tasks, maintenance, "maintenance")
    return summary, finalize_evidence(rows, tasks.task_id)


def annotate_offline_targets(generation: pd.DataFrame, tasks: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Label-conditioned reference relevance, for post-generation scoring only.

    These are coverage proxies, not physical causal ground truth. Maintenance
    remains contextual for every Azure label. No Failure uses observable
    operating indicators, never a failure flag.
    """
    labels = tasks.set_index("task_id")["gold_label"].to_dict()
    result = generation.copy()
    relevant = {
        "TWF": {"tool_wear"}, "HDF": {"temperature_gap", "rotational_speed"},
        "PWF": {"mechanical_power_w"}, "OSF": {"overstrain_score"},
        "No Failure": {"tool_wear", "temperature_gap", "rotational_speed", "mechanical_power_w", "overstrain_score"},
    }
    supports = []
    for row in result.to_dict("records"):
        gold = labels[row["task_id"]]
        use = row["field"] in relevant[gold] if dataset == "AI4I" else row["source"] in {"telemetry", "errors"}
        supports.append(gold if use else "context")
    result["supports"] = supports
    return result
