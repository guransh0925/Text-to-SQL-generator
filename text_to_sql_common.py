import json
from pathlib import Path

import tiktoken
from tiktoken.load import data_gym_to_mergeable_bpe_ranks
from tiktoken_ext.openai_public import ENDOFTEXT, r50k_pat_str
import torch
from torch.utils.data import Dataset

from utilities import GPTModel, load_weights_into_gpt


MODEL_CONFIGS = {
    "gpt2-small": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
    "tiny": {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 128,
        "n_heads": 4,
        "n_layers": 4,
        "drop_rate": 0.1,
        "qkv_bias": False,
    },
}


def format_prompt(schema, question):
    return (
        "Translate the question to SQL.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        "SQL:"
    )


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TextToSQLDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.examples = []
        self.pad_id = tokenizer.eot_token
        for record in records:
            prompt = format_prompt(record["schema"], record["question"])
            full_text = f"{prompt} {record['sql']}<|endoftext|>"
            prompt_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
            full_ids = tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})

            if len(full_ids) < 2:
                continue
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]

            input_ids = full_ids[:-1]
            target_ids = full_ids[1:]
            labels = [
                token_id if token_position >= len(prompt_ids) else -100
                for token_position, token_id in enumerate(target_ids, start=1)
            ]
            self.examples.append((input_ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_batch(batch, pad_id):
    max_len = max(len(input_ids) for input_ids, _ in batch)
    input_batch, target_batch = [], []
    for input_ids, labels in batch:
        pad_len = max_len - len(input_ids)
        input_batch.append(input_ids + [pad_id] * pad_len)
        target_batch.append(labels + [-100] * pad_len)
    return torch.tensor(input_batch, dtype=torch.long), torch.tensor(target_batch, dtype=torch.long)


def infer_config_from_state_dict(state_dict):
    layer_numbers = []
    for key in state_dict:
        if key.startswith("trf_blocks."):
            layer_numbers.append(int(key.split(".")[1]))

    emb_dim = state_dict["token_emb.weight"].shape[1]
    context_length = state_dict["pos_emb.weight"].shape[0]
    return {
        "vocab_size": state_dict["token_emb.weight"].shape[0],
        "context_length": context_length,
        "emb_dim": emb_dim,
        "n_heads": 12 if emb_dim % 12 == 0 else 4,
        "n_layers": max(layer_numbers) + 1,
        "drop_rate": 0.1,
        "qkv_bias": "trf_blocks.0.att.W_query.bias" in state_dict,
    }


def load_model(init, config_name="gpt2-small", checkpoint_path=None, device="cpu"):
    config = MODEL_CONFIGS[config_name].copy()

    if init == "gpt2":
        import tensorflow as tf

        from gpt_download import load_gpt2_params_from_tf_ckpt

        model_dir = Path("gpt2") / "124M"
        hparams_path = model_dir / "hparams.json"
        if not hparams_path.exists():
            raise FileNotFoundError(
                "Local GPT-2 files were not found in gpt2/124M. "
                "Run gpt_download.py once with internet access, or use --init random."
            )
        settings = json.loads(hparams_path.read_text(encoding="utf-8"))
        tf_ckpt_path = tf.train.latest_checkpoint(str(model_dir))
        params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
        config.update({
            "vocab_size": settings["n_vocab"],
            "context_length": settings["n_ctx"],
            "emb_dim": settings["n_embd"],
            "n_heads": settings["n_head"],
            "n_layers": settings["n_layer"],
            "qkv_bias": True,
        })
        model = GPTModel(config)
        load_weights_into_gpt(model, params)
    elif init == "checkpoint":
        loaded = torch.load(checkpoint_path, map_location=device)
        state_dict = loaded.get("model_state_dict", loaded)
        config = loaded.get("config", infer_config_from_state_dict(state_dict))
        model = GPTModel(config)
        model.load_state_dict(state_dict)
    else:
        model = GPTModel(config)

    return model.to(device), config


def get_tokenizer(models_dir="gpt2/124M"):
    local_vocab = Path(models_dir) / "vocab.bpe"
    local_encoder = Path(models_dir) / "encoder.json"
    if local_vocab.exists() and local_encoder.exists():
        mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
            vocab_bpe_file=str(local_vocab),
            encoder_json_file=str(local_encoder),
        )
        return tiktoken.Encoding(
            name="local-gpt2",
            explicit_n_vocab=50257,
            pat_str=r50k_pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens={ENDOFTEXT: 50256},
        )
    return tiktoken.get_encoding("gpt2")
