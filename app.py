import os
import threading
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

from text_to_sql_common import format_prompt, get_tokenizer, load_model
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

MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "guransh0925/text-to-sql-generator")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "text_to_sql_model.pth")
MODEL_CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT_PATH")
TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", "gpt2/124M")

_generator = None
_lock = threading.Lock()


def extract_sql(generated_text):
    if "SQL:" in generated_text:
        generated_text = generated_text.split("SQL:", 1)[1]
    generated_text = generated_text.split("<|endoftext|>", 1)[0]
    if ";" in generated_text:
        generated_text = generated_text.split(";", 1)[0] + ";"
    return generated_text.strip()


def resolve_checkpoint():
    if MODEL_CHECKPOINT_PATH and Path(MODEL_CHECKPOINT_PATH).exists():
        return MODEL_CHECKPOINT_PATH

    return hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)


class SQLGenerator:
    def __init__(self):
        checkpoint = resolve_checkpoint()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = get_tokenizer(TOKENIZER_DIR)
        self.model, self.config = load_model("checkpoint", "gpt2-small", checkpoint, self.device)
        self.model.eval()

    def generate_sql(self, schema, question):
        prompt = format_prompt(schema.strip(), question.strip())
        with torch.no_grad():
            token_ids = generate(
                model=self.model,
                idx=text_to_token_ids(prompt, self.tokenizer).to(self.device),
                max_new_tokens=80,
                context_size=self.config["context_length"],
                temperature=0.0,
                top_k=None,
                eos_id=self.tokenizer.eot_token,
            )
        output = token_ids_to_text(token_ids.cpu(), self.tokenizer)
        return extract_sql(output)


def get_generator():
    global _generator
    with _lock:
        if _generator is None:
            _generator = SQLGenerator()
        return _generator


def generate_sql(schema, question):
    schema = schema.strip()
    question = question.strip()
    if not schema:
        return "Add a database schema first."
    if not question:
        return "Ask a question to translate."
    return get_generator().generate_sql(schema, question)


demo = gr.Interface(
    fn=generate_sql,
    inputs=[
        gr.Textbox(label="Database schema", value=DEFAULT_SCHEMA, lines=10),
        gr.Textbox(label="Question", value="Show completed orders over 1000.", lines=2),
    ],
    outputs=gr.Code(label="Generated SQL", language="sql"),
    title="Text to SQL Generator",
    description="Enter a database schema and a natural-language question. The app generates SQL using the fine-tuned model.",
    examples=[
        [
            "Table: employees\nColumns: employee_id, employee_name, department, salary",
            "Retrieve the names of all employees earning more than $50,000.",
        ],
        [
            DEFAULT_SCHEMA,
            "Show completed orders over 1000.",
        ],
    ],
)


if __name__ == "__main__":
    demo.launch()
