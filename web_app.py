import argparse
import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch

from text_to_sql_common import format_prompt, get_tokenizer, load_model
from sql_heuristics import heuristic_sql
from utilities import generate, text_to_token_ids, token_ids_to_text


ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
DEFAULT_CHECKPOINT = "checkpoints/text_to_sql_model.pth"
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


def ensure_checkpoint(checkpoint_path):
    checkpoint = Path(checkpoint_path)
    if checkpoint.exists():
        return checkpoint_path

    checkpoint_url = os.getenv("MODEL_CHECKPOINT_URL")
    if not checkpoint_url:
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Set MODEL_CHECKPOINT_URL "
            "to download it at startup, or upload the checkpoint to the host separately."
        )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model checkpoint to {checkpoint}...")
    urlretrieve(checkpoint_url, checkpoint)
    return checkpoint_path


class SQLGenerator:
    def __init__(self, checkpoint, init, config, tokenizer_dir, max_new_tokens, temperature, top_k):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = get_tokenizer(tokenizer_dir)
        self.model, self.config = load_model(init, config, checkpoint, self.device)
        self.model.eval()

    def generate_sql(self, schema, question):
        prompt = format_prompt(schema.strip(), question.strip())
        with self.lock, torch.no_grad():
            token_ids = generate(
                model=self.model,
                idx=text_to_token_ids(prompt, self.tokenizer).to(self.device),
                max_new_tokens=self.max_new_tokens,
                context_size=self.config["context_length"],
                temperature=self.temperature,
                top_k=self.top_k,
                eos_id=self.tokenizer.eot_token,
            )
        output = token_ids_to_text(token_ids.cpu(), self.tokenizer)
        return extract_sql(output)


def json_response(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(generator):
    class TextToSQLHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_DIR), **kwargs)

        def log_message(self, fmt, *args):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/defaults":
                json_response(self, 200, {"schema": DEFAULT_SCHEMA})
                return
            if parsed.path == "/health":
                json_response(self, 200, {"ok": True})
                return
            if parsed.path == "/":
                self.path = "/index.html"
            super().do_GET()

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/generate":
                json_response(self, 404, {"error": "Not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(body) if body else {}
                schema = payload.get("schema", "").strip()
                question = payload.get("question", "").strip()
                if not schema:
                    json_response(self, 400, {"error": "Add a database schema first."})
                    return
                if not question:
                    json_response(self, 400, {"error": "Ask a question to translate."})
                    return

                sql = heuristic_sql(schema, question) or generator.generate_sql(schema, question)
                json_response(self, 200, {"sql": sql})
            except Exception as exc:
                json_response(self, 500, {"error": str(exc)})

    return TextToSQLHandler


def parse_args():
    parser = argparse.ArgumentParser(description="Run the text-to-SQL website.")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--checkpoint", default=os.getenv("MODEL_CHECKPOINT_PATH", DEFAULT_CHECKPOINT))
    parser.add_argument("--tokenizer-dir", default=os.getenv("TOKENIZER_DIR", "gpt2/124M"))
    parser.add_argument("--init", choices=["checkpoint", "gpt2", "random"], default="checkpoint")
    parser.add_argument("--config", choices=["gpt2-small", "tiny"], default="gpt2-small")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    checkpoint = ensure_checkpoint(args.checkpoint) if args.init == "checkpoint" else args.checkpoint
    generator = SQLGenerator(
        checkpoint=checkpoint,
        init=args.init,
        config=args.config,
        tokenizer_dir=args.tokenizer_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(generator))
    print(f"Text-to-SQL website running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")


if __name__ == "__main__":
    main()
