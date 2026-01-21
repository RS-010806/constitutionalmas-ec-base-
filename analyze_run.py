import argparse
import json
from pathlib import Path

from src.metrics import format_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=str, required=True, help="Path to a run artifact .json")
    args = parser.parse_args()

    path = Path(args.artifact)
    data = json.loads(path.read_text(encoding="utf-8"))

    print(f"Artifact: {path}")
    print(f"Model: {data.get('model', {}).get('name', 'unknown')}")
    print(f"Task: {data.get('task', '')}")
    print()

    fm = data.get("framework_metrics", {})
    print("Framework Metrics:")
    for k in ["violations_detected", "revisions_triggered", "total_tokens"]:
        if k in fm:
            print(f"- {k}: {fm[k]}")
    print()

    derived = data.get("derived_metrics", {})
    for line in format_metrics(derived):
        print(line)

    history = data.get("history", [])
    print()
    print("Conversation (compact):")
    for m in history:
        sender = m.get("sender", "Unknown")
        content = (m.get("content", "") or "").replace("\n", " ").strip()
        if len(content) > 220:
            content = content[:220] + "..."
        print(f"- {sender}: {content}")


if __name__ == "__main__":
    main()

