import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.environment import Environment
from src.llm import GeminiModel
from src.metrics import approx_tokens, compute_metrics

from experiments.hotpotqa.formatting import (
    context_to_text,
    extract_pred_evidence,
    normalize_title,
    parse_final_json,
    select_context,
    validate_constraints,
)
from experiments.hotpotqa.scoring import canonicalize_pred_answer, exact_match, f1_score, evidence_f1
from experiments.hotpotqa.stream import reservoir_sample, stream_json_array


def load_constitution_text(constitution_path: str) -> str:
    with open(constitution_path, "r", encoding="utf-8") as f:
        constitution = yaml.safe_load(f)
    parts = ["CONSTITUTION (Core Principles you must obey):"]
    for principle, details in constitution.get("principles", {}).items():
        parts.append(f"- {principle.upper()}: {details.get('definition', '')}")
    return "\n".join(parts)


def prompt_schema() -> str:
    return (
        "Return ONLY valid JSON with keys:\n"
        "- final_answer: string\n"
        "- reasoning_steps: array of 2-5 short strings\n"
        "- supporting_facts: array of objects {title: string, sent_id: int}\n"
        "Rules:\n"
        "- Use ONLY the provided CONTEXT sentences.\n"
        "- Each supporting_facts item must reference a sentence index shown in CONTEXT.\n"
        "- Provide at least 2 supporting facts; prefer using 2 distinct titles when available.\n"
        "- final_answer MUST be the minimal answer span (no extra words).\n"
        "  * Yes/No questions: final_answer must be exactly 'yes' or 'no'.\n"
        "  * 'What year'/'When' questions: final_answer should be the year only (e.g., '1974').\n"
        "  * Numeric questions: final_answer should be the number only.\n"
        "  * Entity questions: final_answer should be just the entity/phrase, but include enough tokens to uniquely identify it (e.g., full person name).\n"
        "  * Do not add explanatory text around final_answer.\n"
        "- Keep JSON under 220 words.\n"
    )


def single_agent_solve(llm: GeminiModel, example: dict, max_titles: int) -> dict:
    context = select_context(example, max_titles=max_titles)
    ctx_text = context_to_text(context)
    question = example.get("question", "")

    prompt = (
        "You are solving a HotpotQA question using only the provided context.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{ctx_text}\n\n"
        f"{prompt_schema()}"
    )
    text = llm.generate(prompt)
    return {"final_text": text}


def free_comm_solve(llm: GeminiModel, example: dict, max_titles: int) -> dict:
    context = select_context(example, max_titles=max_titles)
    ctx_text = context_to_text(context)
    question = example.get("question", "")

    r_prompt = (
        "Role: Retriever.\n"
        "Select the most relevant context sentences (by title and sentence id) to answer the question.\n"
        "Return JSON: {facts: [{title, sent_id, quote}]}\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx_text}\n"
    )
    r_text = llm.generate(r_prompt)

    s_prompt = (
        "Role: Reasoner.\n"
        "Using only the provided context, draft an answer with short reasoning.\n"
        "You may use the retriever output but must not invent facts.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx_text}\n\n"
        f"RETRIEVER_OUTPUT:\n{r_text}\n\n"
        f"{prompt_schema()}"
    )
    s_text = llm.generate(s_prompt)

    v_prompt = (
        "Role: Verifier.\n"
        "Validate that the answer is supported by cited sentences and the JSON is valid.\n"
        "If invalid or unsupported, rewrite into compliant JSON.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx_text}\n\nDRAFT:\n{s_text}\n\n"
        f"{prompt_schema()}"
    )
    v_text = llm.generate(v_prompt)

    return {"retriever": r_text, "reasoner": s_text, "final_text": v_text}


def structured_protocol_solve(llm: GeminiModel, example: dict, max_titles: int) -> dict:
    context = select_context(example, max_titles=max_titles)
    ctx_text = context_to_text(context)
    question = example.get("question", "")

    r_prompt = (
        "Protocol: STRICT_JSON_ONLY.\n"
        "Role: Retriever.\n"
        "Return ONLY valid JSON: {facts: [{title: string, sent_id: int, quote: string}]}\n"
        "Use only CONTEXT. No extra keys.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx_text}\n"
    )
    r_text = llm.generate(r_prompt)
    r_parsed, _ = parse_final_json(r_text)
    facts = (r_parsed or {}).get("facts", [])
    if not isinstance(facts, list):
        facts = []

    s_prompt = (
        "Protocol: STRICT_JSON_ONLY.\n"
        "Role: Reasoner.\n"
        "Return ONLY valid JSON: {draft: {final_answer, reasoning_steps, supporting_facts}}\n"
        "Use only CONTEXT and RETRIEVED_FACTS.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx_text}\n\n"
        f"RETRIEVED_FACTS_JSON:\n{json.dumps({'facts': facts}, ensure_ascii=False)}\n\n"
        f"{prompt_schema()}"
    )
    s_text = llm.generate(s_prompt)
    s_parsed, _ = parse_final_json(s_text)
    draft = (s_parsed or {}).get("draft", s_parsed)

    v_prompt = (
        "Protocol: STRICT_JSON_ONLY.\n"
        "Role: Verifier.\n"
        "Return ONLY valid JSON matching the required schema. If draft is invalid, fix it.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx_text}\n\n"
        f"DRAFT_JSON:\n{json.dumps(draft, ensure_ascii=False)}\n\n"
        f"{prompt_schema()}"
    )
    v_text = llm.generate(v_prompt)
    return {"retriever": r_text, "reasoner": s_text, "final_text": v_text}


def central_manager_solve(llm: GeminiModel, constitution_text: str, example: dict, max_titles: int) -> dict:
    draft = free_comm_solve(llm, example, max_titles=max_titles)
    draft_text = draft.get("final_text", "")
    context = select_context(example, max_titles=max_titles)
    ctx_text = context_to_text(context)
    question = example.get("question", "")

    manager_prompt = (
        "You are a Constitutional Manager.\n"
        "Rewrite the DRAFT into a constitution-compliant response.\n"
        "If the draft invents facts or cites unsupported evidence, remove or correct it.\n"
        "Do not claim certainty beyond the provided context.\n\n"
        f"{constitution_text}\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx_text}\n\nDRAFT:\n{draft_text}\n\n"
        f"{prompt_schema()}"
    )
    managed = llm.generate(manager_prompt)
    draft["manager_rewrite"] = managed
    draft["final_text"] = managed
    return draft


def peer_constitution_solve(llm: GeminiModel, example: dict, max_titles: int) -> dict:
    env = Environment(llm, logger=None, max_revision_rounds=3)
    env.add_agent("Retriever", "hotpotqa_retrieval_specialist")
    env.add_agent("Reasoning", "hotpotqa_reasoning_specialist")
    env.add_agent("Verifier", "hotpotqa_verification_specialist")

    context = select_context(example, max_titles=max_titles)
    ctx_text = context_to_text(context)
    question = example.get("question", "")

    task = (
        "HOT POT QA TASK (strict JSON-only outputs).\n"
        "Contracts:\n"
        "- Retriever: return ONLY JSON {facts:[{title,sent_id},...]} (no final answer).\n"
        "- Reasoner: return ONLY JSON {draft:{final_answer,reasoning_steps,supporting_facts}}.\n"
        "- Verifier: return ONLY the FINAL JSON matching the required schema.\n"
        "Titles in supporting_facts must match CONTEXT titles exactly.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{ctx_text}\n\n"
        f"{prompt_schema()}"
    )

    history, framework_metrics = env.run_simulation(task, max_turns=3)
    final_text = ""
    for msg in reversed(history):
        if msg.get("sender") == "Verifier":
            final_text = msg.get("content", "")
            break

    pre_parsed, pre_err = parse_final_json(final_text)
    pre_constraints = validate_constraints(pre_parsed, example)
    repair_attempts = 0

    while repair_attempts < 2 and (pre_err or not pre_constraints.get("satisfied")):
        repair_attempts += 1
        repair_prompt = (
            "Repair the following draft into a valid, constraint-compliant JSON response.\n"
            "Return ONLY JSON. Do not wrap in markdown code fences.\n"
            "Rules:\n"
            "- Use ONLY the provided CONTEXT.\n"
            "- supporting_facts must reference existing (title, sent_id) pairs from CONTEXT.\n"
            "- Provide at least 2 supporting facts; prefer 2 distinct titles when available.\n"
            "- final_answer MUST be the minimal answer span (no extra words): yes/no, year-only, number-only, or entity-only.\n"
            "  Include enough tokens to disambiguate entities (e.g., full person name), but do not add explanation.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{ctx_text}\n\n"
            f"DRAFT:\n{final_text}\n\n"
            f"{prompt_schema()}"
        )
        final_text = llm.generate(repair_prompt)
        pre_parsed, pre_err = parse_final_json(final_text)
        pre_constraints = validate_constraints(pre_parsed, example)

    derived = compute_metrics(history)
    return {
        "final_text": final_text,
        "framework_metrics": framework_metrics,
        "derived_metrics": derived,
        "history": history,
        "repair_attempts": repair_attempts,
        "constraints_post": pre_constraints,
        "parse_error_post": pre_err,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="hotpot_dev_distractor_v1.json")
    parser.add_argument("--hotpot_dir", type=str, default="hotpot")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        type=str,
        default="peer_constitution",
        choices=["no_comm", "free_comm", "structured_protocol", "central_manager", "peer_constitution"],
    )
    parser.add_argument("--max_titles", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="runs/hotpotqa")
    args = parser.parse_args()

    run_once(
        split=args.split,
        hotpot_dir=args.hotpot_dir,
        limit=args.limit,
        seed=args.seed,
        mode=args.mode,
        max_titles=args.max_titles,
        out_dir=args.out_dir,
    )


def run_once(
    split: str,
    hotpot_dir: str,
    limit: int,
    seed: int,
    mode: str,
    max_titles: int,
    out_dir: str,
    model_name: str = "gemini-2.0-flash",
) -> str:
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY not found. Put it in .env")

    data_path = str(Path(hotpot_dir) / split)
    if not Path(data_path).exists():
        raise SystemExit(f"Missing dataset file: {data_path}")

    llm = GeminiModel(model_name=model_name)
    constitution_text = load_constitution_text("configs/constitution.yaml")

    stream = stream_json_array(data_path)
    examples = reservoir_sample(stream, k=limit, seed=seed)

    run_id = f"hotpotqa_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root = Path(out_dir) / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "mode": mode,
        "split": split,
        "limit": limit,
        "seed": seed,
        "max_titles": max_titles,
        "model": model_name,
    }
    (out_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    pred_path = out_root / "predictions.jsonl"
    stats = {
        "count": 0,
        "task_success_em_count": 0,
        "task_success_rate_em": 0.0,
        "task_success_f1": 0.0,
        "constraint_violation_rate": 0.0,
        "avg_constraint_violations": 0.0,
        "avg_approx_tokens": 0.0,
        "avg_evidence_f1": 0.0,
        "avg_framework_violations_detected": 0.0,
        "avg_framework_revisions_triggered": 0.0,
    }

    f1s = []
    toks = []
    ev_f1s = []
    constraint_violation_any = []
    constraint_violation_counts = []
    fw_v = []
    fw_r = []

    with open(pred_path, "w", encoding="utf-8") as out_f:
        for ex in tqdm(examples, desc=f"HotpotQA {mode}"):
            if mode == "no_comm":
                result = single_agent_solve(llm, ex, max_titles=max_titles)
            elif mode == "free_comm":
                result = free_comm_solve(llm, ex, max_titles=max_titles)
            elif mode == "structured_protocol":
                result = structured_protocol_solve(llm, ex, max_titles=max_titles)
            elif mode == "central_manager":
                result = central_manager_solve(llm, constitution_text, ex, max_titles=max_titles)
            else:
                result = peer_constitution_solve(llm, ex, max_titles=max_titles)

            final_text = result.get("final_text", "")
            parsed, parse_err = parse_final_json(final_text)
            constraints = validate_constraints(parsed, ex)
            context_used = select_context(ex, max_titles=max_titles)

            pred_answer_raw = (parsed or {}).get("final_answer", "")
            pred_answer = canonicalize_pred_answer(ex.get("question", ""), pred_answer_raw)
            gold_answer = ex.get("answer", "")
            em = exact_match(pred_answer, gold_answer)
            f1 = f1_score(pred_answer, gold_answer)

            gold_pairs = {
                (normalize_title(t), int(i))
                for t, i in (ex.get("supporting_facts") or [])
                if isinstance(t, str)
            }
            pred_pairs = extract_pred_evidence(parsed)
            ev = evidence_f1(pred_pairs, gold_pairs)

            record = {
                "id": ex.get("_id"),
                "question": ex.get("question"),
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "pred_answer_raw": pred_answer_raw,
                "exact_match": em,
                "f1": f1,
                "gold_supporting_facts": ex.get("supporting_facts"),
                "pred_supporting_facts": (parsed or {}).get("supporting_facts"),
                "evidence_f1": ev,
                "constraints": constraints,
                "parse_error": parse_err,
                "mode": mode,
                "final_text": final_text,
                "approx_tokens_final_text": approx_tokens(final_text),
                "context_used": context_used,
                "framework_metrics": result.get("framework_metrics"),
                "artifacts": {k: v for k, v in result.items() if k.endswith("_path") or k.endswith("_log") or k.endswith("_artifact")},
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            stats["count"] += 1
            stats["task_success_em_count"] += 1 if em else 0
            f1s.append(f1)
            toks.append(record["approx_tokens_final_text"])
            ev_f1s.append(ev["f1"])
            viols = constraints.get("violations") or []
            if not isinstance(viols, list):
                viols = ["violations_not_list"]
            if parse_err:
                viols = list(viols) + [f"parse_error:{parse_err}"]
            constraint_violation_counts.append(len(viols))
            constraint_violation_any.append(1 if len(viols) > 0 else 0)

            fm = record.get("framework_metrics") or {}
            if isinstance(fm, dict):
                if isinstance(fm.get("violations_detected"), int):
                    fw_v.append(fm.get("violations_detected"))
                if isinstance(fm.get("revisions_triggered"), int):
                    fw_r.append(fm.get("revisions_triggered"))

    stats["task_success_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    stats["avg_approx_tokens"] = sum(toks) / len(toks) if toks else 0.0
    stats["avg_evidence_f1"] = sum(ev_f1s) / len(ev_f1s) if ev_f1s else 0.0
    stats["task_success_rate_em"] = stats["task_success_em_count"] / stats["count"] if stats["count"] else 0.0
    stats["constraint_violation_rate"] = sum(constraint_violation_any) / len(constraint_violation_any) if constraint_violation_any else 0.0
    stats["avg_constraint_violations"] = sum(constraint_violation_counts) / len(constraint_violation_counts) if constraint_violation_counts else 0.0
    stats["avg_framework_violations_detected"] = sum(fw_v) / len(fw_v) if fw_v else 0.0
    stats["avg_framework_revisions_triggered"] = sum(fw_r) / len(fw_r) if fw_r else 0.0

    (out_root / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {pred_path}")
    print(f"Wrote {out_root / 'summary.json'}")
    return str(out_root)


if __name__ == "__main__":
    main()
