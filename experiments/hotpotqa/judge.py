import json

from experiments.hotpotqa.formatting import build_context_index, parse_final_json


def build_cited_evidence_text(example: dict, parsed: dict) -> str:
    ctx = build_context_index(example)
    items = parsed.get("supporting_facts") or []
    lines = []
    for it in items:
        if not isinstance(it, dict):
            continue
        title = it.get("title")
        sent_id = it.get("sent_id")
        if not isinstance(title, str) or not isinstance(sent_id, int):
            continue
        sents = ctx.get(title)
        if not sents or sent_id < 0 or sent_id >= len(sents):
            continue
        lines.append(f"- {title} [{sent_id}]: {sents[sent_id]}")
    return "\n".join(lines)


def judge_reasoning(llm, example: dict, parsed: dict, judge_version: str = "v1") -> dict:
    cited = build_cited_evidence_text(example, parsed)
    question = example.get("question", "")
    final_answer = parsed.get("final_answer", "")
    reasoning_steps = parsed.get("reasoning_steps", [])

    prompt = (
        "You are grading reasoning-chain validity for a HotpotQA response.\n"
        "You must judge whether the reasoning steps logically support the final answer using ONLY the cited evidence.\n"
        "Do not use outside knowledge. If evidence is insufficient, mark invalid.\n\n"
        "Return ONLY JSON with keys:\n"
        "- valid: boolean\n"
        "- score: number (0.0 to 1.0)\n"
        "- issues: array of short strings\n"
        "- rationale: short string\n\n"
        f"JUDGE_VERSION: {judge_version}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"FINAL_ANSWER:\n{final_answer}\n\n"
        f"REASONING_STEPS:\n{json.dumps(reasoning_steps, ensure_ascii=False)}\n\n"
        "CITED_EVIDENCE:\n"
        f"{cited if cited else '(none)'}\n"
    )

    raw = llm.generate(prompt)
    parsed_j, err = parse_final_json(raw)
    if err or not isinstance(parsed_j, dict):
        return {"valid": False, "score": 0.0, "issues": ["judge_invalid_json"], "rationale": "", "raw": raw}

    valid = bool(parsed_j.get("valid", False))
    score = parsed_j.get("score", 0.0)
    try:
        score = float(score)
    except Exception:
        score = 0.0
    score = max(0.0, min(1.0, score))

    issues = parsed_j.get("issues", [])
    if not isinstance(issues, list):
        issues = ["judge_issues_not_list"]
    issues = [str(x) for x in issues][:10]

    rationale = parsed_j.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = ""

    return {"valid": valid, "score": score, "issues": issues, "rationale": rationale, "raw": raw}

