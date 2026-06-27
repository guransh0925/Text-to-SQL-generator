import argparse
import sys

import torch

from text_to_sql_common import format_prompt, get_tokenizer, load_model
from prepare_spider_text_sql import GENERIC_SCHEMA
from utilities import generate, text_to_token_ids, token_ids_to_text


DEFAULT_SCHEMA = """CREATE TABLE customers (
  id INTEGER PRIMARY KEY,
  name TEXT,
  city TEXT,
  signup_date TEXT
);
CREATE TABLE orders (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER,
  order_date TEXT,
  total REAL,
  status TEXT
);"""


def extract_sql(generated_text):
    if "SQL:" in generated_text:
        generated_text = generated_text.split("SQL:", 1)[1]
    generated_text = generated_text.split("<|endoftext|>", 1)[0]
    if ";" in generated_text:
        generated_text = generated_text.split(";", 1)[0] + ";"
    return generated_text.strip()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Generate SQL from a schema and natural-language question.")
    parser.add_argument("question", nargs="?", default="Show completed orders over 1000.")
    parser.add_argument("--schema-file")
    parser.add_argument("--generic-schema", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/text_to_sql_model.pth")
    parser.add_argument("--init", choices=["checkpoint", "gpt2", "random"], default="checkpoint")
    parser.add_argument("--config", choices=["gpt2-small", "tiny"], default="gpt2-small")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    schema = GENERIC_SCHEMA if args.generic_schema else DEFAULT_SCHEMA
    if args.schema_file:
        with open(args.schema_file, "r", encoding="utf-8") as f:
            schema = f.read()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(args.init, args.config, args.checkpoint, device)
    model.eval()

    prompt = format_prompt(schema, args.question)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=args.max_new_tokens,
        context_size=config["context_length"],
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.eot_token,
    )
    output = token_ids_to_text(token_ids.cpu(), tokenizer)
    print(extract_sql(output))


if __name__ == "__main__":
    main()
