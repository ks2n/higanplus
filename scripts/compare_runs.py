#!/usr/bin/env python3
"""Compare TensorBoard logs across the four recognizer-crop experiments.

After running the four single-experiment notebooks (baseline, left_half,
left_three_quarter, char_aligned) each writes its TensorBoard event file
to ``runs/<config-name>-<m-d-H-M>/``.  Inside each event file the
trainer logs:

* ``valid/fid``, ``valid/kid``, ``valid/is_gen``, ``valid/is_org``
  -- once per epoch when ``training.eval_fid_every`` fires.
* ``loss/<name>``  -- every ``print_iter_val`` iters during training.

This script:

1. Discovers the latest run directory for each experiment prefix.
2. Pulls the requested scalar tags out of every run's event file using
   the public ``tensorboard.backend.event_processing.event_accumulator``
   API (no internal/protobuf knowledge required).
3. Prints a Markdown comparison table to stdout (best / final / mean /
   epoch-of-best per experiment per metric).
4. Optionally writes:
   - ``--csv``  per-epoch wide CSV (one row per epoch, one column per
     "<experiment>__<metric>"), suitable for pandas / the report.
   - ``--png``  matplotlib line plots per metric, four lines per chart.

Run examples
------------
    # Print the markdown table only.
    python scripts/compare_runs.py

    # Custom runs root (e.g. after rsync'ing from Kaggle).
    python scripts/compare_runs.py --runs-root ./downloaded_runs

    # Full report artefacts.
    python scripts/compare_runs.py --csv compare.csv --png compare_plots/

The script is intentionally dependency-light: it only uses
``tensorboard``, ``numpy``, and (for ``--png``) ``matplotlib`` -- all of
which are already in ``requirements.txt``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# Order matters: report tables follow this order so columns line up.
EXPERIMENTS: List[Tuple[str, str]] = [
    ("baseline",            "gan_iam_crop_baseline"),
    ("left_half",           "gan_iam_crop_left_half"),
    ("left_three_quarter",  "gan_iam_crop_left_3q"),
    ("char_aligned",        "gan_iam_crop_char_aligned"),
]

# Per-epoch metrics worth comparing across experiments.  Tuple is
# (tag, "lower-is-better"); the table uses that flag to mark the
# winning experiment in each row.
EPOCH_METRICS: List[Tuple[str, bool]] = [
    ("valid/fid",     True),
    ("valid/kid",     True),
    ("valid/is_gen",  False),
    ("valid/is_org",  False),
]

# Per-iter loss tags that are useful to inspect after the fact (rendered
# only in the PNG plots; the table itself is epoch-resolution to stay
# readable).
ITER_LOSSES: List[str] = [
    "loss/fake_ctc_loss",
    "loss/adv_loss",
    "loss/recn_loss",
    "loss/gp_ctc",
]


@dataclass
class RunData:
    """All scalar series read from a single TensorBoard event file."""

    label: str         # human-readable experiment name
    run_dir: Path      # absolute path to the run directory
    series: Dict[str, np.ndarray]   # tag -> Nx2 array of (step, value)


def find_latest_run(runs_root: Path, prefix: str) -> Optional[Path]:
    """Return the most-recently-modified ``<prefix>-*`` directory under
    ``runs_root``, or ``None`` if no run exists.

    HiGAN+'s ``train.py`` names runs ``<config>-<MM-DD-HH-MM>`` so a
    plain glob + mtime sort is sufficient.  We could parse the
    timestamp instead, but mtime is more forgiving when the user
    rsyncs runs across machines and timestamps in the name lose their
    monotone meaning.
    """

    candidates = sorted(
        runs_root.glob(f"{prefix}-*"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def load_run(run_dir: Path, label: str, tags: Iterable[str]) -> RunData:
    """Read every requested scalar tag from a run directory.

    Returns ``RunData.series`` with at most one entry per requested tag;
    missing tags simply do not appear in the dict, which lets the
    caller distinguish "no data" from "all zeros".
    """

    # Importing here keeps ``--help`` snappy and avoids forcing users
    # who only want the markdown table to wait on tensorboard's heavy
    # protobuf import path.
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    accumulator = EventAccumulator(
        str(run_dir),
        # Default size_guidance keeps only 10k scalar events per tag; we
        # want all of them so the CSV is faithful.
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()

    available = set(accumulator.Tags().get("scalars", []))
    series: Dict[str, np.ndarray] = {}
    for tag in tags:
        if tag not in available:
            continue
        events = accumulator.Scalars(tag)
        if not events:
            continue
        arr = np.asarray([(e.step, e.value) for e in events], dtype=np.float64)
        series[tag] = arr
    return RunData(label=label, run_dir=run_dir, series=series)


def summarise_epoch_metric(
    runs: List[RunData],
    tag: str,
    lower_is_better: bool,
) -> List[Dict[str, object]]:
    """Build one summary row per run for the given epoch-resolution tag.

    Each row carries ``best``, ``best_epoch``, ``final``, ``mean``, and
    a ``winner`` flag set on whichever run has the best ``best``.  When
    a run does not log the tag (e.g. the smoke ran but FID was disabled)
    we still emit a row with NaNs so the table stays rectangular.
    """

    rows: List[Dict[str, object]] = []
    bests: List[float] = []
    for r in runs:
        s = r.series.get(tag)
        if s is None or len(s) == 0:
            row = dict(label=r.label, best=float("nan"), best_epoch=None,
                       final=float("nan"), mean=float("nan"), n=0)
            rows.append(row)
            bests.append(float("nan"))
            continue
        steps = s[:, 0].astype(int)
        vals = s[:, 1]
        if lower_is_better:
            idx = int(np.nanargmin(vals))
        else:
            idx = int(np.nanargmax(vals))
        rows.append(dict(
            label=r.label,
            best=float(vals[idx]),
            best_epoch=int(steps[idx]),
            final=float(vals[-1]),
            mean=float(np.nanmean(vals)),
            n=int(len(vals)),
        ))
        bests.append(float(vals[idx]))

    valid = np.array([b for b in bests if not np.isnan(b)])
    if len(valid) > 0:
        winner_value = valid.min() if lower_is_better else valid.max()
        for row in rows:
            row["winner"] = (
                not np.isnan(row["best"])
                and float(row["best"]) == float(winner_value)
            )
    else:
        for row in rows:
            row["winner"] = False
    return rows


def render_markdown(
    runs: List[RunData],
    metrics: List[Tuple[str, bool]],
) -> str:
    """Pretty-print the comparison as Markdown.

    One table per metric.  Each row is one run, with ``best``,
    ``best_epoch``, ``final``, ``mean``, and a star next to the best
    run for that metric.  Unknown values render as ``--``.
    """

    out: List[str] = []
    out.append(f"# HiGAN+ recognizer-crop run comparison\n")
    out.append(f"_{len(runs)} runs discovered:_\n")
    for r in runs:
        out.append(f"- **{r.label}** -> `{r.run_dir}`\n")
    out.append("")

    for tag, lower in metrics:
        rows = summarise_epoch_metric(runs, tag, lower)
        out.append(f"## `{tag}` ({'lower better' if lower else 'higher better'})\n")
        out.append("| run | best | best epoch | final | mean | n |")
        out.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            best = (
                f"**{row['best']:.4f}**" if row["winner"] and not np.isnan(row['best'])
                else (f"{row['best']:.4f}" if not np.isnan(row['best']) else "--")
            )
            be = "--" if row["best_epoch"] is None else str(row["best_epoch"])
            final = "--" if np.isnan(row["final"]) else f"{row['final']:.4f}"
            mean = "--" if np.isnan(row["mean"]) else f"{row['mean']:.4f}"
            n = row["n"] or "--"
            star = " *" if row["winner"] else ""
            out.append(f"| {row['label']}{star} | {best} | {be} | {final} | {mean} | {n} |")
        out.append("")

    return "\n".join(out)


def write_csv(runs: List[RunData], all_tags: List[str], dst: Path) -> None:
    """Dump one wide CSV with one row per epoch and one column per
    ``<experiment>__<tag>`` pair.

    Different runs may have logged at different epochs (e.g. one was
    interrupted), so we union the steps across all (run, tag) series
    and forward-fill missing cells with NaN.  Pandas-friendly.
    """

    import csv

    # 1. Union of step indices across every series we will export.
    step_set: set[int] = set()
    for r in runs:
        for tag in all_tags:
            s = r.series.get(tag)
            if s is None:
                continue
            step_set.update(int(x) for x in s[:, 0])
    if not step_set:
        print("[compare] no scalar data found; CSV not written", file=sys.stderr)
        return
    steps = sorted(step_set)

    # 2. Build a lookup: (run_label, tag) -> {step: value}.
    table: Dict[Tuple[str, str], Dict[int, float]] = {}
    for r in runs:
        for tag in all_tags:
            s = r.series.get(tag)
            if s is None:
                continue
            table[(r.label, tag)] = {int(x): float(v) for x, v in s}

    columns = [(r.label, tag) for r in runs for tag in all_tags]
    header = ["step"] + [f"{label}__{tag}" for (label, tag) in columns]

    with dst.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for step in steps:
            row: List[object] = [step]
            for col in columns:
                row.append(table.get(col, {}).get(step, ""))
            w.writerow(row)
    print(f"[compare] wrote {dst} ({len(steps)} rows, {len(columns)} series)")


def write_plots(runs: List[RunData], tags: List[str], dst_dir: Path) -> None:
    """Render one PNG per tag, four curves per chart (one per run).

    Skipped silently when matplotlib is not installed.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[compare] matplotlib unavailable ({exc}); skipping plots", file=sys.stderr)
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for r in runs:
            s = r.series.get(tag)
            if s is None or len(s) == 0:
                continue
            ax.plot(s[:, 0], s[:, 1], label=r.label, linewidth=1.6)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(tag)
        ax.set_xlabel("step")
        ax.set_ylabel(tag.split("/", 1)[-1])
        ax.grid(True, alpha=0.3)
        ax.legend()
        # Sanitize filename: TensorBoard tags contain slashes.
        slug = tag.replace("/", "_")
        out_path = dst_dir / f"{slug}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"[compare] wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs-root",
        default="HiGAN+/runs",
        help="directory that contains the <prefix>-<timestamp> run folders "
             "(default: HiGAN+/runs)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="optional path to write a wide per-epoch CSV",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="optional directory to write per-metric PNG plots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional path to write the markdown table to disk; "
             "always also printed to stdout",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    if not runs_root.is_dir():
        print(f"[compare] runs root not found: {runs_root}", file=sys.stderr)
        return 2

    # 1. Resolve runs.
    runs: List[RunData] = []
    all_tags = [t for t, _ in EPOCH_METRICS] + ITER_LOSSES
    for label, prefix in EXPERIMENTS:
        run_dir = find_latest_run(runs_root, prefix)
        if run_dir is None:
            print(f"[compare] no run yet for {label} (prefix '{prefix}-')", file=sys.stderr)
            continue
        runs.append(load_run(run_dir, label, all_tags))

    if not runs:
        print("[compare] no runs found at all; nothing to compare", file=sys.stderr)
        return 1

    # 2. Markdown summary.
    report = render_markdown(runs, EPOCH_METRICS)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"[compare] wrote {args.output}", file=sys.stderr)

    # 3. CSV + PNG artefacts.
    if args.csv:
        write_csv(runs, all_tags, args.csv)
    if args.png:
        write_plots(runs, all_tags, args.png)

    return 0


if __name__ == "__main__":
    sys.exit(main())
