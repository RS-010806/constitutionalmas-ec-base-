import re
import string


def canonicalize_pred_answer(question: str, pred_answer: str) -> str:
    q = (question or "").strip().lower()
    a = (pred_answer or "").strip()
    if not a:
        return ""

    yn_starts = ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "has ", "have ", "had ", "will ", "would ")
    if q.startswith(yn_starts):
        low = a.lower().lstrip()
        if low.startswith("yes"):
            return "yes"
        if low.startswith("no"):
            return "no"

    if "what year" in q or q.startswith("when "):
        m = re.search(r"\b(1\d{3}|20\d{2})\b", a)
        if m:
            return m.group(1)

    if "how many" in q or "how much" in q or "how far" in q:
        m = re.search(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", a)
        if m:
            return m.group(0).replace(",", "")

    return a


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            common[t] -= 1
            num_same += 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evidence_f1(pred_pairs: set[tuple[str, int]], gold_pairs: set[tuple[str, int]]) -> dict:
    if not pred_pairs and not gold_pairs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_pairs and gold_pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if pred_pairs and not gold_pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    inter = pred_pairs.intersection(gold_pairs)
    precision = len(inter) / len(pred_pairs) if pred_pairs else 0.0
    recall = len(inter) / len(gold_pairs) if gold_pairs else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}

