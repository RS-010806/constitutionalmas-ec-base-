import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs" / "hotpotqa"

MODE_LABELS = {
    "no_comm": "No-Comm",
    "free_comm": "Free-Comm",
    "structured_protocol": "Structured-Protocol",
    "central_manager": "Central-Manager",
    "peer_constitution": "Peer-Constitution",
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_run_dir(run_dir_str: str) -> Path:
    p = Path(run_dir_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()

def resolve_benchmark_dir(benchmark_dir: str, benchmark_id: str) -> Path:
    if benchmark_dir:
        p = Path(benchmark_dir)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    if benchmark_id:
        p = DEFAULT_RUNS_ROOT / benchmark_id
        if p.exists():
            return p.resolve()
        raise SystemExit(f"benchmark_id not found: {p}")

    candidates = sorted(DEFAULT_RUNS_ROOT.glob("benchmark_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"No benchmark_* directories found under {DEFAULT_RUNS_ROOT}")
    return candidates[0].resolve()


def first_second_half_stats(values: pd.Series) -> dict:
    v = values.dropna().astype(float).reset_index(drop=True)
    if len(v) == 0:
        return {"first_mean": float("nan"), "second_mean": float("nan"), "pct_change": float("nan")}
    mid = max(1, len(v) // 2)
    first = v.iloc[:mid].mean()
    second = v.iloc[mid:].mean() if mid < len(v) else v.iloc[:mid].mean()
    pct_change = 0.0 if first == 0 else ((second - first) / first) * 100.0
    return {"first_mean": float(first), "second_mean": float(second), "pct_change": float(pct_change)}


def make_learning_curve(
    series_by_label: dict[str, pd.Series],
    ylabel: str,
    out_path: Path,
    window: int = 7,
    title: str = "",
    annotate_stats: bool = False,
    show_mean_lines: bool = False,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.25, 2.05), constrained_layout=True)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for i, (label, s) in enumerate(series_by_label.items()):
        y = s.dropna().astype(float).reset_index(drop=True)
        x = pd.Series(range(1, len(y) + 1), dtype=float)

        smooth = y.rolling(window=window, min_periods=1, center=True).mean()

        ax.plot(x, smooth, lw=1.8, color=colors[i % len(colors)], label=label)

        stats = first_second_half_stats(y)
        first_mean = stats["first_mean"]
        second_mean = stats["second_mean"]
        pct = -stats["pct_change"]

        if show_mean_lines:
            ax.axhline(first_mean, lw=0.8, ls="--", color=colors[i % len(colors)], alpha=0.18)
            ax.axhline(second_mean, lw=0.8, ls="-.", color=colors[i % len(colors)], alpha=0.18)

        if annotate_stats:
            ax.text(
                0.99,
                0.02 + 0.09 * i,
                f"{label}: {first_mean:.1f} → {second_mean:.1f} ({pct:.1f}% ↓)",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color=colors[i % len(colors)],
                fontsize=7.0,
            )

    if title:
        ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)

    ax.grid(True, which="major", axis="y", alpha=0.18)
    ax.set_axisbelow(True)

    ax.legend(frameon=False, loc="upper right", handlelength=2.2)

    try:
        fig.savefig(out_path, dpi=300)
    except PermissionError:
        alt = out_path.with_name(f"{out_path.stem}_new{out_path.suffix}")
        fig.savefig(alt, dpi=300)
        print(f"Permission denied writing {out_path}. Wrote instead: {alt}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default="",
        help="Path to runs/hotpotqa/<benchmark_id> containing runs.json. If set, script loads run dirs from there.",
    )
    parser.add_argument(
        "--benchmark_id",
        type=str,
        default="",
        help="Benchmark directory name under runs/hotpotqa (e.g., benchmark_20260123_124728). If omitted, picks latest.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Direct path to a predictions.jsonl file. Can be repeated. If provided, --benchmark_dir is optional.",
    )
    parser.add_argument("--setting", type=str, default="distractor", choices=["distractor", "fullwiki"])
    parser.add_argument(
        "--modes",
        type=str,
        default="peer_constitution,free_comm",
        help="Comma-separated modes to include when using --benchmark_dir.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="Rolling window size for smoothing (episodes).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional figure title. Prefer leaving empty and using LaTeX caption.",
    )
    parser.add_argument(
        "--annotate_stats",
        action="store_true",
        help="Overlay first-half/second-half token stats on the figure (can clutter).",
    )
    parser.add_argument(
        "--show_mean_lines",
        action="store_true",
        help="Draw faint first-half/second-half mean reference lines.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "Docs" / "figures" / "learning_curves.pdf"),
        help="Output path (.pdf recommended for LaTeX).",
    )
    args = parser.parse_args()

    series_by_label: dict[str, pd.Series] = {}

    if args.run:
        for p_str in args.run:
            p = Path(p_str)
            if p.is_dir():
                p = p / "predictions.jsonl"
            rows = load_jsonl(p)
            df = pd.DataFrame(rows)
            if "approx_tokens_final_text" not in df.columns:
                raise SystemExit(f"Missing approx_tokens_final_text in {p}")
            label = p.parent.name
            series_by_label[label] = df["approx_tokens_final_text"]
    else:
        bench = resolve_benchmark_dir(args.benchmark_dir, args.benchmark_id)
        runs_index = json.loads((bench / "runs.json").read_text(encoding="utf-8"))

        want_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        for r in runs_index:
            if r.get("setting") != args.setting:
                continue
            mode = r.get("mode")
            if mode not in want_modes:
                continue
            run_dir = resolve_run_dir(r.get("run_dir", ""))
            preds = run_dir / "predictions.jsonl"
            if not preds.exists():
                raise SystemExit(f"Missing predictions: {preds}")

            rows = load_jsonl(preds)
            df = pd.DataFrame(rows)
            label = MODE_LABELS.get(mode, mode.replace("_", "-"))
            series_by_label[label] = df["approx_tokens_final_text"]

    if not series_by_label:
        raise SystemExit("No series selected; check --setting/--modes or --run paths.")

    out_path = Path(args.out)
    ylabel = "Approx. tokens"

    make_learning_curve(
        series_by_label=series_by_label,
        ylabel=ylabel,
        out_path=out_path,
        window=max(1, int(args.window)),
        title=(args.title or ""),
        annotate_stats=bool(args.annotate_stats),
        show_mean_lines=bool(args.show_mean_lines),
    )

    print(f"Wrote: {out_path}")

    for label, s in series_by_label.items():
        stats = first_second_half_stats(s)
        first_mean = stats["first_mean"]
        second_mean = stats["second_mean"]
        pct_down = -stats["pct_change"]
        print(f"- {label}: {first_mean:.2f} → {second_mean:.2f} ({pct_down:.2f}% reduction)")


if __name__ == "__main__":
    main()
