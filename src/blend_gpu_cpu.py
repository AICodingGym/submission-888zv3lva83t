"""Fit an OOF-selected CPU/GPU blend and write the competition submission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cpu import apply_thresholds, optimize_thresholds, qwk, validate_submission  # noqa: E402


def load_joined(cpu_oof: Path, gpu_oof: Path, cpu_test: Path, gpu_test: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(cpu_oof).merge(pd.read_csv(gpu_oof), on=["essay_id", "score"], validate="one_to_one")
    test = pd.read_csv(cpu_test).merge(pd.read_csv(gpu_test), on="essay_id", validate="one_to_one")
    if train[["pred_blend", "pred_gpu"]].isna().any().any() or test[["pred_blend", "pred_gpu"]].isna().any().any():
        raise ValueError("Missing predictions in the CPU/GPU merge")
    return train, test


def select_weight(y: np.ndarray, cpu: np.ndarray, gpu: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    best: tuple[float, np.ndarray, np.ndarray, float] | None = None
    for gpu_weight in np.arange(0.0, 1.001, 0.025):
        prediction = (1.0 - gpu_weight) * cpu + gpu_weight * gpu
        thresholds, score = optimize_thresholds(y, prediction)
        if best is None or score > best[3]:
            best = (float(gpu_weight), prediction, thresholds, float(score))
    assert best is not None
    return best


def main(args: argparse.Namespace) -> dict[str, object]:
    train, test = load_joined(Path(args.cpu_oof), Path(args.gpu_oof), Path(args.cpu_test), Path(args.gpu_test))
    y = train["score"].to_numpy(dtype=np.int8)
    cpu = train["pred_blend"].to_numpy(dtype=np.float64)
    gpu = train["pred_gpu"].to_numpy(dtype=np.float64)
    gpu_weight, continuous, thresholds, oof_qwk = select_weight(y, cpu, gpu)
    test_continuous = (1.0 - gpu_weight) * test["pred_blend"].to_numpy() + gpu_weight * test["pred_gpu"].to_numpy()
    final = apply_thresholds(test_continuous, thresholds)

    cross_fitted = np.zeros(len(y), dtype=np.int8)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    for train_index, valid_index in splitter.split(continuous, y):
        fold_thresholds, _ = optimize_thresholds(y[train_index], continuous[train_index])
        cross_fitted[valid_index] = apply_thresholds(continuous[valid_index], fold_thresholds)

    submission = pd.DataFrame({"essay_id": test["essay_id"], "score": final.astype(int)})
    sample = pd.read_csv(args.sample_submission)
    validate_submission(submission, sample.rename(columns={"score": "score"}))
    Path(args.submission_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_path, index=False)
    result = {
        "gpu_weight": gpu_weight,
        "cpu_weight": 1.0 - gpu_weight,
        "oof_qwk_thresholds_fit_on_all": oof_qwk,
        "oof_qwk_cross_fitted_thresholds": qwk(y, cross_fitted),
        "thresholds": [round(float(value), 8) for value in thresholds],
        "test_score_counts": {str(int(score)): int((final == score).sum()) for score in range(1, 7)},
        "submission_path": str(Path(args.submission_path).resolve()),
    }
    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOF-calibrated DeBERTa plus CPU ensemble")
    parser.add_argument("--cpu-oof", default="outputs/cpu/oof_predictions.csv")
    parser.add_argument("--gpu-oof", default="outputs/deberta_v3_large/oof_predictions.csv")
    parser.add_argument("--cpu-test", default="outputs/cpu/test_predictions.csv")
    parser.add_argument("--gpu-test", default="outputs/deberta_v3_large/test_predictions.csv")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--submission-path", default="submission_gpu_ensemble.csv")
    parser.add_argument("--metrics-path", default="outputs/gpu_ensemble/metrics.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
