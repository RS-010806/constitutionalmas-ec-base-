import math
from statistics import mean, median


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def words_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def compute_metrics(history: list[dict]) -> dict:
    messages = [m for m in history if m.get("sender") != "User"]
    by_sender: dict[str, list[dict]] = {}
    for m in messages:
        by_sender.setdefault(m.get("sender", "Unknown"), []).append(m)

    per_message = []
    for m in messages:
        content = m.get("content", "") or ""
        per_message.append(
            {
                "sender": m.get("sender", "Unknown"),
                "chars": len(content),
                "words": words_count(content),
                "approx_tokens": approx_tokens(content),
            }
        )

    def summarize(items: list[dict]) -> dict:
        if not items:
            return {
                "count": 0,
                "avg_chars": 0,
                "median_chars": 0,
                "avg_words": 0,
                "median_words": 0,
                "avg_approx_tokens": 0,
                "median_approx_tokens": 0,
            }
        chars = [i["chars"] for i in items]
        words = [i["words"] for i in items]
        toks = [i["approx_tokens"] for i in items]
        return {
            "count": len(items),
            "avg_chars": round(mean(chars), 2),
            "median_chars": median(chars),
            "avg_words": round(mean(words), 2),
            "median_words": median(words),
            "avg_approx_tokens": round(mean(toks), 2),
            "median_approx_tokens": median(toks),
        }

    overall = summarize(per_message)

    per_sender = {}
    for sender, msgs in by_sender.items():
        sender_items = [
            {
                "chars": len(m.get("content", "") or ""),
                "words": words_count(m.get("content", "") or ""),
                "approx_tokens": approx_tokens(m.get("content", "") or ""),
            }
            for m in msgs
        ]
        per_sender[sender] = summarize(sender_items)

    n = len(per_message)
    first_half = per_message[: max(1, n // 2)] if n else []
    second_half = per_message[max(1, n // 2) :] if n else []
    trend = {
        "first_half_avg_approx_tokens": summarize(first_half)["avg_approx_tokens"] if n else 0,
        "second_half_avg_approx_tokens": summarize(second_half)["avg_approx_tokens"] if n else 0,
    }

    return {
        "overall": overall,
        "per_sender": per_sender,
        "trend": trend,
    }


def format_metrics(metrics: dict) -> list[str]:
    lines = []
    overall = metrics.get("overall", {})
    lines.append("Message Efficiency (Approx):")
    lines.append(
        f"- Messages: {overall.get('count', 0)} | "
        f"Avg tokens: {overall.get('avg_approx_tokens', 0)} | "
        f"Median tokens: {overall.get('median_approx_tokens', 0)} | "
        f"Avg words: {overall.get('avg_words', 0)}"
    )

    trend = metrics.get("trend", {})
    lines.append(
        f"- Token trend (first→second half avg): "
        f"{trend.get('first_half_avg_approx_tokens', 0)} → {trend.get('second_half_avg_approx_tokens', 0)}"
    )

    per_sender = metrics.get("per_sender", {})
    if per_sender:
        lines.append("Per-Agent Efficiency:")
        for sender, s in per_sender.items():
            lines.append(
                f"- {sender}: {s.get('count', 0)} msgs | "
                f"Avg tokens {s.get('avg_approx_tokens', 0)} | "
                f"Avg words {s.get('avg_words', 0)}"
            )

    return lines

