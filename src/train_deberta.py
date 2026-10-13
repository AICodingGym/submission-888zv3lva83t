"""Fine-tune DeBERTa-v3 for ordinal essay scoring on a single GPU.

The model predicts a continuous score.  It deliberately leaves conversion to
the six competition labels to the OOF threshold calibration in
``blend_gpu_cpu.py``; training a classifier and rounding a regressor are both
less faithful to the competition's quadratic-weighted-kappa metric.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cpu import qwk, round_and_clip, validate_input  # noqa: E402


class EncodedEssayDataset(Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: np.ndarray | None = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, object]:
        item: dict[str, object] = {key: value[index] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = float(self.labels[index])
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(tokenizer: AutoTokenizer, texts: np.ndarray, max_length: int) -> dict[str, list[list[int]]]:
    return tokenizer(
        texts.tolist(),
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_attention_mask=True,
    )


def make_loader(
    dataset: Dataset,
    collator: DataCollatorWithPadding,
    batch_size: int,
    shuffle: bool,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=collator,
    )


@torch.inference_mode()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        batch.pop("labels", None)
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits.squeeze(-1)
        values.append(logits.float().cpu().numpy())
    return np.concatenate(values)


def train_fold(
    args: argparse.Namespace,
    fold: int,
    tokenizer: AutoTokenizer,
    train_texts: np.ndarray,
    train_y: np.ndarray,
    valid_texts: np.ndarray,
    valid_y: np.ndarray,
    test_texts: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    device = torch.device("cuda")
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    train_dataset = EncodedEssayDataset(tokenize(tokenizer, train_texts, args.max_length), train_y)
    valid_dataset = EncodedEssayDataset(tokenize(tokenizer, valid_texts, args.max_length))
    test_dataset = EncodedEssayDataset(tokenize(tokenizer, test_texts, args.max_length))
    train_loader = make_loader(train_dataset, collator, args.train_batch_size, True, args.workers)
    valid_loader = make_loader(valid_dataset, collator, args.eval_batch_size, False, args.workers)
    test_loader = make_loader(test_dataset, collator, args.eval_batch_size, False, args.workers)

    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type="regression",
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
    ).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = (len(train_loader) + args.gradient_accumulation - 1) // args.gradient_accumulation
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )
    best_loss = float("inf")
    best_path = output_dir / f"fold_{fold}.pt"
    metrics: dict[str, object] = {"fold": fold, "epochs": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"fold {fold} epoch {epoch}")
        for step, batch in enumerate(progress, start=1):
            labels = batch.pop("labels").to(device, non_blocking=True).float()
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(**batch).logits.squeeze(-1)
                # Scaling makes gradient accumulation exactly equivalent to a larger batch.
                loss = F.smooth_l1_loss(predictions.float(), labels, beta=0.5) / args.gradient_accumulation
            loss.backward()
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.item()) * args.gradient_accumulation
            progress.set_postfix(loss=f"{running_loss / step:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        valid_predictions = predict(model, valid_loader, device)
        valid_loss = float(np.mean((valid_predictions - valid_y) ** 2))
        valid_qwk = qwk(valid_y.astype(np.int8), round_and_clip(valid_predictions))
        if valid_loss < best_loss:
            best_loss = valid_loss
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(state, best_path)
        epoch_metrics = {
            "epoch": epoch,
            "train_smooth_l1": running_loss / len(train_loader),
            "valid_mse": valid_loss,
            "valid_qwk_rounded": valid_qwk,
            "checkpoint": str(best_path),
        }
        metrics["epochs"].append(epoch_metrics)
        print(json.dumps({"fold": fold, **epoch_metrics}), flush=True)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    valid_predictions = predict(model, valid_loader, device)
    test_predictions = predict(model, test_loader, device)
    del model, optimizer, scheduler, train_loader, valid_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()
    return valid_predictions, test_predictions, metrics


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training script")
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_df = pd.read_csv(data_dir / "sample_submission.csv")
    validate_input(train_df, test_df, sample_df)
    if args.limit:
        train_df = train_df.groupby("score", group_keys=False).sample(
            n=min(args.limit // 6, train_df["score"].value_counts().min()), random_state=args.seed
        ).reset_index(drop=True)

    texts = train_df["full_text"].astype(str).to_numpy()
    test_texts = test_df["full_text"].astype(str).to_numpy()
    y = train_df["score"].to_numpy(dtype=np.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    oof = np.zeros(len(train_df), dtype=np.float64)
    test_prediction = np.zeros(len(test_df), dtype=np.float64)
    metrics: list[dict[str, object]] = []
    started = time.perf_counter()
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    for fold, (train_index, valid_index) in enumerate(splitter.split(texts, y), start=1):
        print(f"\n===== fold {fold}/{args.folds} =====", flush=True)
        fold_started = time.perf_counter()
        valid_prediction, fold_test_prediction, fold_metrics = train_fold(
            args,
            fold,
            tokenizer,
            texts[train_index],
            y[train_index],
            texts[valid_index],
            y[valid_index],
            test_texts,
            output_dir / "checkpoints",
        )
        oof[valid_index] = valid_prediction
        test_prediction += fold_test_prediction / args.folds
        rounded = round_and_clip(valid_prediction)
        fold_metrics["valid_qwk_rounded"] = qwk(y[valid_index].astype(np.int8), rounded)
        fold_metrics["seconds"] = round(time.perf_counter() - fold_started, 2)
        metrics.append(fold_metrics)
        print(json.dumps({"fold": fold, "valid_qwk_rounded": fold_metrics["valid_qwk_rounded"], "seconds": fold_metrics["seconds"]}), flush=True)

    pd.DataFrame({"essay_id": train_df["essay_id"], "score": y.astype(int), "pred_gpu": oof}).to_csv(
        output_dir / "oof_predictions.csv", index=False
    )
    pd.DataFrame({"essay_id": test_df["essay_id"], "pred_gpu": test_prediction}).to_csv(
        output_dir / "test_predictions.csv", index=False
    )
    result: dict[str, object] = {
        "model_name": args.model_name,
        "folds": args.folds,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "oof_qwk_rounded": qwk(y.astype(np.int8), round_and_clip(oof)),
        "fold_metrics": metrics,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nFinal GPU metrics")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-fold DeBERTa-v3 essay-score regressor")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/deberta_v3_large")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-large")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least two")
    return args


if __name__ == "__main__":
    run(parse_args())
