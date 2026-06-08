import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm import GeminiModel
from experiments.hotpotqa.formatting import parse_final_json
from experiments.hotpotqa.judge import judge_reasoning
from experiments.hotpotqa.scoring import exact_match, f1_score

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def constraint_violation_stats(row: dict) -> tuple[int, int]:
    constraints = row.get("constraints") or {}
    violations = constraints.get("violations") or []
    if not isinstance(violations, list):
        violations = ["violations_not_list"]
    parse_error = row.get("parse_error")
    if parse_error:
        violations = list(violations) + [f"parse_error:{parse_error}"]
    count = len(violations)
    return count, 1 if count > 0 else 0


def attach_llm_judge(rows: list[dict], judge_model: str, judge_temperature: float, cache_path: Path):
    cache = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cache[rec.get("key")] = rec

    llm = GeminiModel(model_name=judge_model, temperature=judge_temperature)

    with open(cache_path, "a", encoding="utf-8") as f:
        for r in rows:
            key = f"{r.get('mode')}::{r.get('id')}"
            if key in cache:
                r["judge"] = cache[key].get("judge")
                continue

            final_text = r.get("final_text", "")
            parsed, err = parse_final_json(final_text)
            if err or not isinstance(parsed, dict):
                judge = {"valid": False, "score": 0.0, "issues": ["invalid_prediction_json"], "rationale": "", "raw": ""}
            else:
                example = {"question": r.get("question", ""), "context": r.get("context_used", [])}
                judge = judge_reasoning(llm, example, parsed)

            rec = {"key": key, "judge": judge}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            r["judge"] = judge


def evaluate_predictions(
    predictions_path: str,
    use_judge: bool = False,
    judge_model: str = "gemini-2.0-flash",
    judge_temperature: float = 0.0,
    out_csv: str = "",
) -> pd.DataFrame:
    rows = load_jsonl(predictions_path)
    if not rows:
        raise SystemExit("No rows found")

    pred_dir = Path(predictions_path).parent
    cache_path = pred_dir / "judge_cache.jsonl"
    if use_judge:
        attach_llm_judge(rows, judge_model=judge_model, judge_temperature=judge_temperature, cache_path=cache_path)

    for r in rows:
        violation_count, violation_any = constraint_violation_stats(r)
        r["constraint_violation_count"] = violation_count
        r["constraint_violated"] = violation_any
        r["evidence_f1_f1"] = (r.get("evidence_f1") or {}).get("f1", 0.0)

        # strict vs canonicalized answer metrics
        gold_answer = r.get("gold_answer", "") or ""
        pred_answer_raw = r.get("pred_answer_raw", r.get("pred_answer", "")) or ""
        r["strict_em"] = int(bool(exact_match(pred_answer_raw, gold_answer)))
        r["strict_f1"] = float(f1_score(pred_answer_raw, gold_answer))

        if use_judge:
            j = r.get("judge") or {}
            r["logical_consistency_valid"] = int(bool(j.get("valid", False)))
            r["logical_consistency_score"] = float(j.get("score", 0.0) or 0.0)
        else:
            constraints_ok = bool((r.get("constraints") or {}).get("satisfied"))
            has_answer = bool(r.get("pred_answer"))
            r["logical_consistency_valid"] = int(constraints_ok and has_answer)
            r["logical_consistency_score"] = 1.0 if r["logical_consistency_valid"] else 0.0

    df = pd.DataFrame(rows)
    agg = {
        "exact_match": "mean",
        "f1": "mean",
        "strict_em": "mean",
        "strict_f1": "mean",
        "logical_consistency_valid": "mean",
        "logical_consistency_score": "mean",
        "constraint_violated": "mean",
        "constraint_violation_count": "mean",
        "approx_tokens_final_text": "mean",
        "evidence_f1_f1": "mean",
    }
    summary = df.groupby("mode").agg(agg).reset_index()
    summary = summary.rename(
        columns={
            "exact_match": "task_success_rate_em",
            "f1": "task_success_f1",
            "strict_em": "task_success_rate_em_strict",
            "strict_f1": "task_success_f1_strict",
            "logical_consistency_valid": "logical_consistency_rate",
            "logical_consistency_score": "logical_consistency_score",
            "constraint_violated": "constraint_violation_rate",
            "constraint_violation_count": "avg_constraint_violations",
            "approx_tokens_final_text": "avg_approx_tokens",
            "evidence_f1_f1": "evidence_f1",
        }
    )

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions.jsonl")
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--use_judge", action="store_true")
    parser.add_argument("--judge_model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    args = parser.parse_args()

    summary = evaluate_predictions(
        predictions_path=args.predictions,
        use_judge=args.use_judge,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
        out_csv=args.out_csv,
    )
    print(summary.to_string(index=False))
    if args.out_csv:
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
