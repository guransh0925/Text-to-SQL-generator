# Text-to-SQL Generator

This project is a text-to-SQL generator built with a small GPT-style language model implemented from scratch in PyTorch. It takes a database schema plus a natural-language question and returns a SQL query.

The model architecture follows the learning path from Sebastian Raschka's *Build a Large Language Model From Scratch*, then adapts that code into a practical text-to-SQL demo with a command-line generator and a hosted web interface.

## What It Does

Input:

```text
Schema:
Table: employees
Columns: employee_id, employee_name, salary

Question:
Retrieve the names of all employees earning more than $50,000.
```

Output:

```sql
SELECT employee_name FROM employees WHERE salary > 50000;
```

The web demo also includes a lightweight schema-aware fallback for common simple queries, so examples like completed orders over a total threshold return valid SQL even when the fine-tuned model struggles.

## Web Demo

The hosted version is designed to run on Hugging Face Spaces using `app.py`.

The model weights are not committed to this repository because they are large. The Space downloads the checkpoint from a Hugging Face model repository at runtime.

Default model settings:

```text
MODEL_REPO_ID=guransh0925/text-to-sql-generator
MODEL_FILENAME=text_to_sql_model.pth
```

If the checkpoint filename changes, update `MODEL_FILENAME` in the Space settings.

## Run Locally

From the repository folder:

```powershell
python -m pip install -r requirements.txt
```

Generate SQL from the command line:

```powershell
python generate_sql.py "Retrieve the names of all employees earning more than $50,000." --schema-file employee_schema.txt --checkpoint checkpoints/text_to_sql_model.pth
```

Run the browser-based local app:

```powershell
python web_app.py --checkpoint checkpoints/text_to_sql_model.pth
```

Then open:

```text
http://127.0.0.1:8000
```

If you are running from inside a copied repo folder and your `.venv`, `checkpoints`, or `gpt2` folders are one directory above it, use paths like:

```powershell
..\.venv\Scripts\python.exe web_app.py --checkpoint ..\checkpoints\text_to_sql_model.pth --tokenizer-dir ..\gpt2\124M
```

## Example Schemas To Try

Customers and orders:

```sql
CREATE TABLE customers (
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
);
```

Questions:

```text
Show completed orders over 1000.
Show orders with total greater than 1000.
Show all customers.
Show the names of customers from Delhi.
```

Employees:

```text
Table: employees
Columns: employee_id, employee_name, department, salary
```

Questions:

```text
Show all employees.
Show employee names.
Show employees with salary greater than 50000.
Show names of employees in the sales department.
```

## Training

Prepare the dataset:

```powershell
python prepare_merged_text_sql.py --repeat 8
```

Fine-tune from local GPT-2 124M weights:

```powershell
python train_text_to_sql.py --train-file data/merged_text_sql/train.jsonl --val-file data/merged_text_sql/val.jsonl --init gpt2 --epochs 3 --batch-size 1 --max-steps 1200 --lr 2e-5
```

Evaluate:

```powershell
python evaluate_text_to_sql.py --test-file data/merged_text_sql/test.jsonl --checkpoint checkpoints/text_to_sql_model.pth --limit 50
```

Metrics are saved to:

```text
outputs/metrics/summary.json
outputs/metrics/predictions.csv
```

## Project Files

- `app.py` - Hugging Face Spaces Gradio app
- `web_app.py` - lightweight Python web server for the custom browser UI
- `web/` - HTML, CSS, and JavaScript for the browser UI
- `sql_heuristics.py` - schema-aware fallback for common simple queries
- `generate_sql.py` - command-line SQL generator
- `train_text_to_sql.py` - fine-tuning script
- `evaluate_text_to_sql.py` - evaluation script
- `text_to_sql_common.py` - prompt formatting, tokenizer loading, model loading, dataset helpers
- `utilities.py` - GPT model, attention blocks, generation helpers, GPT-2 weight loading
- `prepare_merged_text_sql.py` - converts `merged_data.csv` into train/validation/test JSONL files
- `gpt_download.py` - helper for downloading/loading GPT-2 weights
- `merged_data.csv` - schema-aware text-to-SQL dataset
- `DEPLOYMENT.md` - hosting notes

## Limitations

This is an educational fine-tuned model, not a production SQL assistant. It can handle simple examples, but it may fail on joins, nested queries, unfamiliar schemas, and exact SQL formatting. Generated SQL should always be reviewed before running it on a real database.

Large model files are intentionally ignored by Git:

- `checkpoints/`
- `gpt2/`
- `*.pth`
- `*.pt`
- `.venv/`

For hosting, use a Python/ML-capable service such as Hugging Face Spaces. Static hosts like GitHub Pages cannot run the PyTorch model.
