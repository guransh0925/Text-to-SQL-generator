import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def database_to_schema(database_value):
    try:
        metadata = ast.literal_eval(str(database_value))
    except (ValueError, SyntaxError):
        return str(database_value)

    table = metadata.get("table") or metadata.get("tables") or "unknown_table"
    columns = metadata.get("columns") or []
    if isinstance(table, list):
        table = ", ".join(str(item) for item in table)
    if isinstance(columns, list):
        columns = ", ".join(str(item) for item in columns)
    return f"Table: {table}\nColumns: {columns}"


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert merged_data.csv into JSONL fine-tuning files.")
    parser.add_argument("--csv", default="merged_data.csv")
    parser.add_argument("--out-dir", default="data/merged_text_sql")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["instruction", "query", "database"])
    df = df.sample(frac=1, random_state=args.seed)

    records = []
    for row in df.itertuples(index=False):
        record = {
            "schema": database_to_schema(row.database),
            "question": str(row.instruction).strip(),
            "sql": str(row.query).strip(),
        }
        records.append(record)

    train_end = int(len(records) * 0.9)
    val_end = int(len(records) * 0.95)
    train = records[:train_end] * args.repeat
    val = records[train_end:val_end]
    test = records[val_end:]

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)
    print(
        f"Wrote {len(train)} train, {len(val)} val, "
        f"{len(test)} test records to {out_dir}"
    )


if __name__ == "__main__":
    main()
