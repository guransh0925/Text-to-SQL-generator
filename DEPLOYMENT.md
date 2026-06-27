# Hosting the Text-to-SQL Website

The website is a Python web process that serves the UI and runs the model behind `/api/generate`.

## Keep Model Weights Out of Git

The repo intentionally ignores `.pth` files and the `checkpoints/` folder. Put the trained checkpoint somewhere your host can download it, then set:

```text
MODEL_CHECKPOINT_URL=https://your-checkpoint-download-url
```

On startup, `web_app.py` downloads that file to:

```text
checkpoints/text_to_sql_model.pth
```

You can also upload the checkpoint directly to your host and set:

```text
MODEL_CHECKPOINT_PATH=/path/to/text_to_sql_model.pth
```

## Local Run

```powershell
python -m pip install -r requirements-web.txt
python web_app.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Deploy

Use a Python web service, not a static site host. Static hosts can show the HTML, but they cannot run the PyTorch model.

Build command:

```text
pip install -r requirements-web.txt
```

Start command:

```text
python web_app.py --host 0.0.0.0
```

Required environment variable:

```text
MODEL_CHECKPOINT_URL=https://your-checkpoint-download-url
```

For a Hugging Face model repo, use the direct file URL from the repo's **Files** tab. The URL format is:

```text
https://huggingface.co/guransh0925/text-to-sql-generator/resolve/main/YOUR_CHECKPOINT_FILENAME.pth
```

For example, if the checkpoint file is named `text_to_sql_model.pth`, set:

```text
MODEL_CHECKPOINT_URL=https://huggingface.co/guransh0925/text-to-sql-generator/resolve/main/text_to_sql_model.pth
```

Optional environment variables:

```text
MODEL_CHECKPOINT_PATH=checkpoints/text_to_sql_model.pth
TOKENIZER_DIR=gpt2/124M
PORT=8000
```

Most hosts provide `PORT` automatically. The included `Procfile` works for hosts that read Procfile-style web commands. If your host cannot download the GPT-2 tokenizer files through `tiktoken`, upload `encoder.json` and `vocab.bpe` somewhere in the app and point `TOKENIZER_DIR` at that folder.
