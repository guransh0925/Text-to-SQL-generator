import argparse
import csv
import json
import re
from pathlib import Path

import torch

from text_to_sql_common import format_prompt, get_tokenizer, load_model, read_jsonl
from utilities import generate, text_to_token_ids, token_ids_to_text
from generate_sql import extract_sql


def normalize_sql(sql):
    sql = sql.strip().rstrip(";").lower()
    sql = re.sub(r"\s+", " ", sql)
    sql = re.sub(r"\s*,\s*", ", ", sql)
    sql = re.sub(r"\s*(=|>|<|>=|<=)\s*", r" \1 ", sql)
    sql = re.sub(r"\s+", " ", sql)
    return sql.strip()


def token_f1(prediction, target):
    pred_tokens = normalize_sql(prediction).split()
    target_tokens = normalize_sql(target).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0

    target_counts = {}
    for token in target_tokens:
        target_counts[token] = target_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        if target_counts.get(token, 0) > 0:
            overlap += 1
            target_counts[token] -= 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def generate_one(model, tokenizer, config, record, device, max_new_tokens):
    prompt = format_prompt(record["schema"], record["question"])
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=max_new_tokens,
        context_size=config["context_length"],
        temperature=0.0,
        eos_id=tokenizer.eot_token,
    )
    return extract_sql(token_ids_to_text(token_ids.cpu(), tokenizer))


def main():
    parser = argparse.ArgumentParser(description="Evaluate the text-to-SQL checkpoint.")
    parser.add_argument("--test-file", default="data/merged_text_sql/test.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/text_to_sql_model.pth")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out-dir", default="outputs/metrics")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model("checkpoint", checkpoint_path=args.checkpoint, device=device)
    model.eval()

    records = read_jsonl(args.test_file)[:args.limit]
    rows = []
    exact_matches = 0
    f1_scores = []

    for idx, record in enumerate(records, start=1):
        prediction = generate_one(model, tokenizer, config, record, device, args.max_new_tokens)
        target = record["sql"]
        exact = normalize_sql(prediction) == normalize_sql(target)
        f1 = token_f1(prediction, target)
        exact_matches += int(exact)
        f1_scores.append(f1)
        rows.append({
            "index": idx,
            "question": record["question"],
            "target_sql": target,
            "predicted_sql": prediction,
            "exact_match": exact,
            "token_f1": round(f1, 4),
        })
        print(f"{idx}/{len(records)} exact={exact} f1={f1:.3f}")

    summary = {
        "test_file": args.test_file,
        "checkpoint": args.checkpoint,
        "examples": len(records),
        "exact_match": round(exact_matches / max(1, len(records)), 4),
        "avg_token_f1": round(sum(f1_scores) / max(1, len(f1_scores)), 4),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
