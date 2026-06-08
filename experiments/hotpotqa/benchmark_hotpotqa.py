import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.hotpotqa.evaluate_hotpotqa import evaluate_predictions
from experiments.hotpotqa.run_hotpotqa import run_once


def setting_name(split: str) -> str:
    s = split.lower()
    if "distractor" in s:
        return "distractor"
    if "fullwiki" in s:
        return "fullwiki"
    return split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpot_dir", type=str, default="hotpot")
    parser.add_argument("--splits", type=str, default="hotpot_dev_distractor_v1.json,hotpot_dev_fullwiki_v1.json")
    parser.add_argument(
        "--modes",
        type=str,
        default="no_comm,free_comm,structured_protocol,central_manager,peer_constitution",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_titles", type=int, default=None)
    parser.add_argument("--max_titles_distractor", type=int, default=10)
    parser.add_argument("--max_titles_fullwiki", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="runs/hotpotqa")
    parser.add_argument("--include_em", action="store_true")
    parser.add_argument("--use_judge", action="store_true")
    parser.add_argument("--judge_model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    benchmark_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    benchmark_root = Path(args.out_dir) / benchmark_id
    benchmark_root.mkdir(parents=True, exist_ok=True)
    (benchmark_root / "config.json").write_text(
        json.dumps(
            {
                "benchmark_id": benchmark_id,
                "hotpot_dir": args.hotpot_dir,
                "splits": splits,
                "modes": modes,
                "limit": args.limit,
                "seed": args.seed,
                "max_titles": args.max_titles,
                "max_titles_distractor": args.max_titles_distractor,
                "max_titles_fullwiki": args.max_titles_fullwiki,
                "include_em": args.include_em,
                "use_judge": args.use_judge,
                "judge_model": args.judge_model,
                "judge_temperature": args.judge_temperature,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = []
    run_index = []

    for split in splits:
        mt = args.max_titles
        if mt is None:
            mt = args.max_titles_fullwiki if setting_name(split) == "fullwiki" else args.max_titles_distractor

        for mode in modes:
            out_root = run_once(
                split=split,
                hotpot_dir=args.hotpot_dir,
                limit=args.limit,
                seed=args.seed,
                mode=mode,
                max_titles=mt,
                out_dir=args.out_dir,
            )
            out_root = Path(out_root)
            preds = out_root / "predictions.jsonl"
            summary = evaluate_predictions(
                predictions_path=str(preds),
                use_judge=args.use_judge,
                judge_model=args.judge_model,
                judge_temperature=args.judge_temperature,
            )

            summary["setting"] = setting_name(split)
            summary["split"] = split
            summary["run_dir"] = str(out_root)
            rows.append(summary)

            run_index.append({"setting": setting_name(split), "split": split, "mode": mode, "run_dir": str(out_root)})

    all_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    columns = [
        "setting",
        "mode",
        "task_success_f1",
        "task_success_f1_strict",
        "logical_consistency_rate",
        "logical_consistency_score",
        "constraint_violation_rate",
        "avg_constraint_violations",
        "evidence_f1",
        "avg_approx_tokens",
        "split",
        "run_dir",
    ]
    if args.include_em:
        columns.insert(2, "task_success_rate_em")
        columns.insert(3, "task_success_rate_em_strict")

    all_df = all_df[columns]

    comparison_csv = benchmark_root / "comparison.csv"
    comparison_json = benchmark_root / "comparison.json"
    all_df.to_csv(comparison_csv, index=False)
    comparison_json.write_text(all_df.to_json(orient="records", indent=2), encoding="utf-8")
    (benchmark_root / "runs.json").write_text(json.dumps(run_index, indent=2), encoding="utf-8")

    print(all_df.to_string(index=False))
    print(f"Wrote {comparison_csv}")
    print(f"Wrote {comparison_json}")


if __name__ == "__main__":
    main()
