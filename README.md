# Text-to-SQL GPT Fine-Tuning

This project fine-tunes a GPT-style model for a small text-to-SQL task. The model architecture is implemented from scratch in `utilities.py`, following the learning path from Sebastian Raschka's *Build a Large Language Model From Scratch*, then adapted into a practical SQL generation demo.

## What It Does

Given database metadata and a natural-language instruction, the model generates a SQL query.

Example:

```text
Schema:
Table: employees
Columns: employee_id, employee_name, salary

Question: Retrieve the names of all employees earning more than $50,000.
```

Output:

```sql
SELECT employee_name FROM employees WHERE salary > 50000;
```

## Project Files

- `utilities.py` - GPT model, attention blocks, generation helpers, GPT-2 weight loading
- `gpt_download.py` - helper from Raschka's book for loading GPT-2 weights
- `prepare_merged_text_sql.py` - converts `merged_data.csv` into train/validation/test JSONL files
- `train_text_to_sql.py` - fine-tunes the model
- `generate_sql.py` - command-line SQL generator
- `web_app.py` - web server for the browser-based generator
- `web/` - frontend files for the hosted demo
- `evaluate_text_to_sql.py` - records exact-match and token-F1 metrics
- `merged_data.csv` - schema-aware text-to-SQL dataset

Large model files are intentionally ignored by Git.

## Setup

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Prepare the dataset:

```powershell
.\.venv\Scripts\python.exe prepare_merged_text_sql.py --repeat 8
```

Fine-tune from local GPT-2 124M weights:

```powershell
.\.venv\Scripts\python.exe train_text_to_sql.py --train-file data/merged_text_sql/train.jsonl --val-file data/merged_text_sql/val.jsonl --init gpt2 --epochs 3 --batch-size 1 --max-steps 1200 --lr 2e-5
```

Generate SQL:

```powershell
.\.venv\Scripts\python.exe generate_sql.py "Retrieve the names of all employees earning more than $50,000." --schema-file employee_schema.txt
```

Run the website locally:

```powershell
.\.venv\Scripts\python.exe web_app.py
```

Then open:

```text
http://127.0.0.1:8000
```

Evaluate:

```powershell
.\.venv\Scripts\python.exe evaluate_text_to_sql.py --test-file data/merged_text_sql/test.jsonl --checkpoint checkpoints/text_to_sql_model.pth --limit 50
```

Metrics are saved to:

```text
outputs/metrics/summary.json
outputs/metrics/predictions.csv
```

## Current Sample Result

On a quick 10-example sample from an earlier checkpoint:

```text
Exact match: 0.00
Average token F1: 0.3136
```

The model can produce correct simple schema-aware examples, but it still struggles with complex joins and nested SQL. This is expected for a small educational fine-tune and is listed as a limitation rather than hidden.

## Notes

The following files/folders are local artifacts and should not be committed:

- `checkpoints/`
- `gpt2/`
- `model.pth`
- `model_and_optimizer.pth`
- `.venv/`

If sharing the trained model, upload the checkpoint separately as a GitHub release asset or provide instructions to train it locally.

For hosting the website, see `DEPLOYMENT.md`. Use a Python web service, not a static site host, because the server needs to load the PyTorch model and respond to `/api/generate`.
