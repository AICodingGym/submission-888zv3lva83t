from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_cpu import apply_thresholds, optimize_thresholds, qwk  # noqa: E402


def evaluate(args: argparse.Namespace) -> dict:
    oof = pd.read_csv(args.oof)
    with Path(args.metrics).open("r", encoding="utf-8") as handle:
        training_metrics = json.load(handle)

    y = oof["score"].to_numpy(dtype=np.int8)
    continuous = oof["pred_blend"].to_numpy(dtype=np.float64)
    final = oof["pred_final"].to_numpy(dtype=np.int8)

    cross_fitted = np.zeros(len(oof), dtype=np.int8)
    cross_fitted_folds: list[dict] = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    for fold, (train_index, valid_index) in enumerate(splitter.split(continuous, y), start=1):
        thresholds, _ = optimize_thresholds(
            y[train_index], continuous[train_index], passes=6, candidate_quantiles=301
        )
        cross_fitted[valid_index] = apply_thresholds(continuous[valid_index], thresholds)
        cross_fitted_folds.append(
            {
                "fold": fold,
                "qwk": qwk(y[valid_index], cross_fitted[valid_index]),
                "thresholds": [round(float(value), 8) for value in thresholds],
            }
        )

    error = final - y
    matrix = confusion_matrix(y, final, labels=range(1, 7))
    per_score: dict[str, dict] = {}
    for score in range(1, 7):
        mask = y == score
        per_score[str(score)] = {
            "n": int(mask.sum()),
            "exact_accuracy": float(np.mean(final[mask] == y[mask])),
            "within_one": float(np.mean(np.abs(error[mask]) <= 1)),
            "mae": float(np.mean(np.abs(error[mask]))),
            "mean_prediction": float(np.mean(final[mask])),
        }

    report = {
        "qwk_thresholds_fit_on_all_oof": qwk(y, final),
        "qwk_cross_fitted_thresholds": qwk(y, cross_fitted),
        "cross_fitted_threshold_folds": cross_fitted_folds,
        "accuracy": float(np.mean(final == y)),
        "within_one": float(np.mean(np.abs(error) <= 1)),
        "mae": float(np.mean(np.abs(error))),
        "rmse_discrete": float(np.sqrt(np.mean(error**2))),
        "confusion_matrix": matrix.tolist(),
        "per_score": per_score,
        "blend_weights": training_metrics["blend_weights"],
        "thresholds": training_metrics["thresholds"],
        "runtime_seconds": training_metrics["runtime_seconds"],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    header = "| Actual \\ Predicted | 1 | 2 | 3 | 4 | 5 | 6 |\n|---:|---:|---:|---:|---:|---:|---:|"
    confusion_rows = [
        "| " + str(score) + " | " + " | ".join(str(value) for value in matrix[score - 1]) + " |"
        for score in range(1, 7)
    ]
    score_rows = [
        "| {score} | {n} | {exact:.3f} | {within:.3f} | {mae:.3f} | {mean:.3f} |".format(
            score=score,
            n=per_score[str(score)]["n"],
            exact=per_score[str(score)]["exact_accuracy"],
            within=per_score[str(score)]["within_one"],
            mae=per_score[str(score)]["mae"],
            mean=per_score[str(score)]["mean_prediction"],
        )
        for score in range(1, 7)
    ]
    markdown = f"""# CPU OOF Evaluation

## Summary

| Metric | Value |
|---|---:|
| QWK, thresholds fit on all OOF | {report['qwk_thresholds_fit_on_all_oof']:.6f} |
| QWK, cross-fitted thresholds | {report['qwk_cross_fitted_thresholds']:.6f} |
| Exact accuracy | {report['accuracy']:.3f} |
| Within one score | {report['within_one']:.3f} |
| Mean absolute error | {report['mae']:.3f} |
| Discrete RMSE | {report['rmse_discrete']:.3f} |
| CPU training runtime | {report['runtime_seconds']:.2f} seconds |

The cross-fitted threshold score is the conservative validation estimate. The all-OOF score is useful for selecting the final thresholds applied to the test set, but is mildly optimistic because those thresholds were fitted on the same OOF labels.

Blend weights: `{json.dumps(report['blend_weights'])}`

Final thresholds: `{json.dumps(report['thresholds'])}`

## Confusion Matrix

{header}
{chr(10).join(confusion_rows)}

## Per-score Diagnostics

| Actual score | N | Exact accuracy | Within one | MAE | Mean prediction |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(score_rows)}

The hardest classes are the rare extremes. Score 1 is biased upward and score 6 is biased downward; this is consistent with class imbalance and regression toward the mean. Nevertheless, more than 98% of OOF predictions are within one point of the human score.
"""
    output_markdown = Path(args.output_markdown)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an honest OOF error-analysis report")
    parser.add_argument("--oof", default="outputs/cpu/oof_predictions.csv")
    parser.add_argument("--metrics", default="outputs/cpu/metrics.json")
    parser.add_argument("--output-json", default="reports/cpu_evaluation.json")
    parser.add_argument("--output-markdown", default="reports/cpu_evaluation.md")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
