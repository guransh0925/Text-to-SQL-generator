import argparse
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from text_to_sql_common import (
    TextToSQLDataset,
    collate_batch,
    get_tokenizer,
    load_model,
    read_jsonl,
)


def calc_loss(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
        ignore_index=-100,
    )


def evaluate(model, data_loader, device, max_batches=10):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            losses.append(calc_loss(input_batch, target_batch, model, device).item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune this GPT model for text-to-SQL.")
    parser.add_argument("--train-file", default="data/text_to_sql/train.jsonl")
    parser.add_argument("--val-file", default="data/text_to_sql/val.jsonl")
    parser.add_argument("--out", default="checkpoints/text_to_sql_model.pth")
    parser.add_argument("--init", choices=["gpt2", "checkpoint", "random"], default="gpt2")
    parser.add_argument("--checkpoint", default="model.pth")
    parser.add_argument("--config", choices=["gpt2-small", "tiny"], default="gpt2-small")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(args.init, args.config, args.checkpoint, device)
    model.train()

    train_dataset = TextToSQLDataset(read_jsonl(args.train_file), tokenizer, args.max_length)
    val_dataset = TextToSQLDataset(read_jsonl(args.val_file), tokenizer, args.max_length)
    collate = partial(collate_batch, pad_id=tokenizer.eot_token)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    global_step = 0
    for epoch in range(args.epochs):
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % args.eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"step {global_step}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")

            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "init": args.init,
        "max_length": args.max_length,
    }, out_path)
    print(f"Saved fine-tuned checkpoint to {out_path}")


if __name__ == "__main__":
    main()
