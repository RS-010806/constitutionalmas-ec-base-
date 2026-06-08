import json
import html


def normalize_title(title: str) -> str:
    return html.unescape(title).strip()


def build_context_index(example: dict) -> dict[str, list[str]]:
    ctx = {}
    for title, sentences in example.get("context", []):
        ctx[title] = sentences
        ctx[normalize_title(title)] = sentences
    return ctx


def select_context(example: dict, max_titles: int = 8) -> list[tuple[str, list[str]]]:
    question = (example.get("question") or "").lower()
    q_terms = {t for t in question.split() if len(t) > 2}

    scored = []
    for title, sents in example.get("context", []):
        text = (title + " " + " ".join(sents)).lower()
        score = sum(1 for t in q_terms if t in text)
        scored.append((score, title, sents))

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [(t, s) for _, t, s in scored[:max_titles]]
    return picked


def context_to_text(context: list[tuple[str, list[str]]], max_sentences_per_title: int = 10) -> str:
    lines = []
    for title, sents in context:
        lines.append(f"TITLE: {title}")
        for i, sent in enumerate(sents[:max_sentences_per_title]):
            lines.append(f"- ({i}) {sent}")
    return "\n".join(lines)


def parse_final_json(text: str) -> tuple[dict | None, str | None]:
    if not text:
        return None, "empty"

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "no_json_object"
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet), None
    except Exception:
        return None, "invalid_json"


def extract_pred_evidence(parsed: dict | None) -> set[tuple[str, int]]:
    pairs = set()
    if not parsed:
        return pairs
    facts = parsed.get("supporting_facts") or []
    if not isinstance(facts, list):
        return pairs
    for f in facts:
        if not isinstance(f, dict):
            continue
        title = f.get("title")
        sent_id = f.get("sent_id")
        if isinstance(title, str) and isinstance(sent_id, int):
            pairs.add((normalize_title(title), sent_id))
    return pairs


def validate_constraints(parsed: dict | None, example: dict, max_words: int = 220) -> dict:
    violations = []
    if not parsed:
        return {"satisfied": False, "violations": ["invalid_or_missing_json"]}

    required = ["final_answer", "supporting_facts", "reasoning_steps"]
    for k in required:
        if k not in parsed:
            violations.append(f"missing_{k}")

    if "final_answer" in parsed and not isinstance(parsed.get("final_answer"), str):
        violations.append("final_answer_not_string")

    facts = parsed.get("supporting_facts")
    if not isinstance(facts, list):
        violations.append("supporting_facts_not_list")
        facts = []

    titles = []
    for f in facts:
        if not isinstance(f, dict):
            violations.append("supporting_fact_not_object")
            continue
        if not isinstance(f.get("title"), str):
            violations.append("supporting_fact_missing_title")
            continue
        if not isinstance(f.get("sent_id"), int):
            violations.append("supporting_fact_missing_sent_id")
            continue
        titles.append(normalize_title(f.get("title")))

    if len(facts) < 2:
        violations.append("needs_at_least_2_supporting_facts")

    gold_pairs = example.get("supporting_facts") or []
    gold_titles = {normalize_title(t) for t, _ in gold_pairs if isinstance(t, str)}
    required_distinct_titles = 2 if len(gold_titles) >= 2 else 1
    if len(set(titles)) < required_distinct_titles:
        violations.append("needs_distinct_titles")

    steps = parsed.get("reasoning_steps")
    if not isinstance(steps, list) or not all(isinstance(x, str) for x in steps):
        violations.append("reasoning_steps_not_string_list")

    word_count = len(json.dumps(parsed, ensure_ascii=False).split())
    if word_count > max_words:
        violations.append("exceeds_word_budget")

    ctx = build_context_index(example)
    for f in facts:
        if not isinstance(f, dict):
            continue
        title = f.get("title")
        sent_id = f.get("sent_id")
        if not isinstance(title, str) or not isinstance(sent_id, int):
            continue
        norm_title = normalize_title(title)
        if norm_title not in ctx:
            violations.append("supporting_fact_title_not_in_context")
            continue
        sents = ctx[norm_title]
        if sent_id < 0 or sent_id >= len(sents):
            violations.append("supporting_fact_sent_id_out_of_range")
            continue

    return {"satisfied": len(violations) == 0, "violations": violations}

