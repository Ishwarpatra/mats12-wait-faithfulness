from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

DATASET = "open-thoughts/OpenThoughts-114k"
CONFIG = "metadata"
SPLIT = "train"
N_ROWS = 2000
PAGE_SIZE = 100
SEED = 20260815
OUT = Path("/home/ubuntu/work/openthoughts_analysis")
OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT / "metadata_rows.jsonl"
METRICS = OUT / "trace_metrics.csv"
SAMPLES = OUT / "qualitative_samples.jsonl"
SUMMARY = OUT / "summary.json"

WAIT_RE = re.compile(r"(?i)(?<![A-Za-z])wait(?:[!,:.]|\b)")
MARKERS = {
    "wait": WAIT_RE,
    "but": re.compile(r"(?i)\bbut\b"),
    "however": re.compile(r"(?i)\bhowever\b"),
    "actually": re.compile(r"(?i)\bactually\b"),
    "check": re.compile(r"(?i)\b(?:let me|need to|should) (?:check|verify|reconsider)\b"),
}
REVERSAL_RE = re.compile(
    r"(?i)\b(?:no[, ]+(?:that's|that is)|I was wrong|this is wrong|"
    r"correct(?:ion|ed)?|reconsider|let me (?:check|verify)|mistake|misread|instead|rather|"
    r"not quite|on second thought|contradict|re-evaluat|turns out|recalculate|"
    r"doesn't work|does not work|cannot be|can't be)\b"
)
ANSWER_RE = re.compile(r"(?i)(?:answer|output|therefore|thus|so)\s*(?:is|:)?\s*([A-D]|yes|no|true|false)\b")


def fetch_rows() -> list[dict]:
    rows = []
    session = requests.Session()
    total_rows = 113957
    n_pages = math.ceil(N_ROWS / PAGE_SIZE)
    offsets = [round(i * (total_rows - PAGE_SIZE) / max(1, n_pages - 1)) for i in range(n_pages)]
    for page_number, offset in enumerate(offsets):
        length = PAGE_SIZE
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={DATASET}&config={CONFIG}&split={SPLIT}"
            f"&offset={offset}&length={length}"
        )
        response = session.get(url, timeout=60)
        response.raise_for_status()
        payload = response.json()
        for item in payload["rows"]:
            row = item["row"]
            row["row_idx"] = item["row_idx"]
            rows.append(row)
        print(f"fetched {len(rows)}/{N_ROWS} (slice {page_number + 1}/{n_pages}, offset {offset})", flush=True)
        time.sleep(0.05)
    return rows


def safe_text(value) -> str:
    return value if isinstance(value, str) else ""


def trace_metrics(row: dict) -> dict:
    trace = safe_text(row.get("deepseek_reasoning"))
    trace_lower = trace.lower()
    n_chars = len(trace)
    words = trace.split()
    word_count = len(words)
    wait_matches = list(MARKERS["wait"].finditer(trace))
    marker_counts = {name: len(pattern.findall(trace)) for name, pattern in MARKERS.items()}
    positions = [m.start() / max(1, n_chars) for m in wait_matches]
    window_records = []
    for match in wait_matches:
        left = max(0, match.start() - 350)
        right = min(n_chars, match.end() + 550)
        window = trace[left:right]
        window_records.append(
            {
                "position": match.start() / max(1, n_chars),
                "window": window,
                "reversal_proxy": bool(REVERSAL_RE.search(window)),
                "answer_mentions": ANSWER_RE.findall(window),
            }
        )
    first_answer = ANSWER_RE.search(trace)
    last_answers = ANSWER_RE.findall(trace)
    return {
        "row_idx": int(row.get("row_idx", -1)),
        "domain": safe_text(row.get("domain")) or "unknown",
        "source": safe_text(row.get("source")) or "unknown",
        "trace_chars": n_chars,
        "trace_words": word_count,
        "wait_count": marker_counts["wait"],
        "but_count": marker_counts["but"],
        "however_count": marker_counts["however"],
        "actually_count": marker_counts["actually"],
        "check_count": marker_counts["check"],
        "wait_rate_per_1k_words": marker_counts["wait"] / max(1, word_count) * 1000,
        "wait_first_position": positions[0] if positions else math.nan,
        "wait_reversal_proxy_rate": (
            sum(x["reversal_proxy"] for x in window_records) / len(window_records)
            if window_records else math.nan
        ),
        "wait_windows": json.dumps(window_records, ensure_ascii=False),
        "first_answer_mention": first_answer.group(1).lower() if first_answer else "",
        "last_answer_mention": last_answers[-1].lower() if last_answers else "",
        "problem": safe_text(row.get("problem")),
        "deepseek_solution": safe_text(row.get("deepseek_solution")),
        "ground_truth_solution": safe_text(row.get("ground_truth_solution")),
        "deepseek_reasoning": trace,
    }


def matched_control_rate(df: pd.DataFrame) -> dict:
    rng = random.Random(SEED)
    controls = []
    wait_df = df[df["wait_count"] > 0]
    for _, row in wait_df.iterrows():
        trace = row["deepseek_reasoning"]
        n = len(trace)
        wait_positions = [m.start() for m in WAIT_RE.finditer(trace)]
        for pos in wait_positions:
            target = pos / max(1, n)
            candidates = [
                p for p in range(350, max(351, n - 550), 25)
                if not any(abs(p / max(1, n) - wp / max(1, n)) < 0.05 for wp in wait_positions)
            ]
            if not candidates:
                continue
            control_pos = min(candidates, key=lambda p: abs(p / max(1, n) - target))
            left = max(0, control_pos - 350)
            right = min(n, control_pos + 550)
            window = trace[left:right]
            controls.append(
                {
                    "row_idx": int(row["row_idx"]),
                    "marker": "matched_random_position",
                    "position": control_pos / max(1, n),
                    "reversal_proxy": bool(REVERSAL_RE.search(window)),
                }
            )
    if not controls:
        return {"n": 0, "reversal_rate": math.nan}
    return {
        "n": len(controls),
        "reversal_rate": sum(x["reversal_proxy"] for x in controls) / len(controls),
        "controls": controls,
    }


def save_samples(metrics: list[dict]) -> None:
    rng = random.Random(SEED)
    categories = {
        "wait_with_reversal_proxy": [m for m in metrics if m["wait_count"] > 0 and m["wait_reversal_proxy_rate"] > 0],
        "wait_without_reversal_proxy": [m for m in metrics if m["wait_count"] > 0 and m["wait_reversal_proxy_rate"] == 0],
        "no_wait": [m for m in metrics if m["wait_count"] == 0],
    }
    with SAMPLES.open("w", encoding="utf-8") as f:
        for category, items in categories.items():
            for row in rng.sample(items, min(8, len(items))):
                record = {
                    "category": category,
                    "row_idx": row["row_idx"],
                    "domain": row["domain"],
                    "problem": row["problem"],
                    "reasoning": row["deepseek_reasoning"],
                    "deepseek_solution": row["deepseek_solution"],
                    "ground_truth_solution": row["ground_truth_solution"],
                    "wait_count": row["wait_count"],
                    "wait_reversal_proxy_rate": row["wait_reversal_proxy_rate"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_plots(df: pd.DataFrame, control: dict) -> list[str]:
    plt.style.use("seaborn-v0_8-whitegrid")
    paths = []
    domain = (
        df.groupby("domain")
        .agg(traces=("row_idx", "size"), wait_prevalence=("wait_count", lambda x: (x > 0).mean()))
        .sort_values("traces", ascending=False)
        .head(10)
        .sort_values("wait_prevalence")
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=180)
    ax.barh(domain.index.astype(str), domain["wait_prevalence"] * 100, color="#2563eb")
    ax.set_xlabel("Traces containing at least one ‘Wait’ (%)")
    ax.set_ylabel("Dataset domain")
    ax.set_title("Wait-token prevalence varies by domain\n(OpenThoughts-114k metadata sample)")
    for y, value in enumerate(domain["wait_prevalence"] * 100):
        ax.text(value + 0.4, y, f"{value:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    path = OUT / "wait_prevalence_by_domain.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    wait_rows = df[df["wait_count"] > 0]
    wait_rate = float(wait_rows["wait_reversal_proxy_rate"].mean()) if len(wait_rows) else math.nan
    labels = ["Wait windows", "Matched random\nposition windows"]
    values = [wait_rate * 100 if not math.isnan(wait_rate) else 0, control.get("reversal_rate", 0) * 100]
    fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=180)
    bars = ax.bar(labels, values, color=["#dc2626", "#94a3b8"], width=0.58)
    ax.set_ylabel("Windows with reversal-language proxy (%)")
    ax.set_ylim(0, max(100, max(values) * 1.25 if values else 100))
    ax.set_title("A lexical proxy is not a causal test\nWait vs matched positions")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    path = OUT / "wait_vs_matched_control.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    positions = wait_rows["wait_first_position"].dropna()
    fig, ax = plt.subplots(figsize=(7.4, 4.5), dpi=180)
    ax.hist(positions, bins=20, color="#059669", edgecolor="white")
    ax.set_xlabel("Normalized position of first ‘Wait’ (0 = start, 1 = end)")
    ax.set_ylabel("Number of traces")
    ax.set_title("Where does the first Wait appear?")
    fig.tight_layout()
    path = OUT / "wait_first_position.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))
    return paths


def main() -> None:
    if False and RAW.exists() and sum(1 for _ in RAW.open("r", encoding="utf-8")) >= N_ROWS:
        rows = [json.loads(line) for line in RAW.open("r", encoding="utf-8")]
        print(f"reusing {len(rows)} cached rows", flush=True)
    else:
        rows = fetch_rows()
        with RAW.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = [trace_metrics(row) for row in rows]
    df = pd.DataFrame(metrics)
    df.to_csv(METRICS, index=False)
    control = matched_control_rate(df)
    save_samples(metrics)
    plot_paths = make_plots(df, control)
    domain_summary = (
        df.groupby("domain")
        .agg(
            traces=("row_idx", "size"),
            wait_prevalence=("wait_count", lambda x: float((x > 0).mean())),
            mean_wait_rate_per_1k_words=("wait_rate_per_1k_words", "mean"),
            mean_trace_words=("trace_words", "mean"),
        )
        .reset_index()
        .sort_values("traces", ascending=False)
    )
    summary = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "n_rows": len(df),
        "seed": SEED,
        "wait_trace_prevalence": float((df["wait_count"] > 0).mean()),
        "mean_wait_count_per_trace": float(df["wait_count"].mean()),
        "mean_wait_reversal_proxy_rate_among_wait_traces": float(df.loc[df["wait_count"] > 0, "wait_reversal_proxy_rate"].mean()),
        "matched_control": {k: v for k, v in control.items() if k != "controls"},
        "domain_summary": domain_summary.to_dict(orient="records"),
        "plots": plot_paths,
        "raw_data": str(RAW),
        "metrics": str(METRICS),
        "qualitative_samples": str(SAMPLES),
        "interpretation_warning": "All metrics are observational lexical/structural proxies; no causal faithfulness claim is made.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ['n_rows','wait_trace_prevalence','mean_wait_count_per_trace','mean_wait_reversal_proxy_rate_among_wait_traces','matched_control','plots']}, indent=2))


if __name__ == "__main__":
    main()
