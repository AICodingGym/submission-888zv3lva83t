from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split


SCORE_MIN = 1
SCORE_MAX = 6
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_RE = re.compile(r"[.!?]+")


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    actual = np.asarray(y_true, dtype=np.int8) - SCORE_MIN
    predicted = np.asarray(y_pred, dtype=np.int8) - SCORE_MIN
    if actual.shape != predicted.shape:
        raise ValueError("QWK arrays must have matching shapes")
    if np.any((actual < 0) | (actual > SCORE_MAX - SCORE_MIN)):
        raise ValueError("Actual QWK scores must be between 1 and 6")
    if np.any((predicted < 0) | (predicted > SCORE_MAX - SCORE_MIN)):
        raise ValueError("Predicted QWK scores must be between 1 and 6")

    class_count = SCORE_MAX - SCORE_MIN + 1
    observed = np.zeros((class_count, class_count), dtype=np.float64)
    np.add.at(observed, (actual, predicted), 1)
    actual_histogram = np.bincount(actual, minlength=class_count).astype(np.float64)
    predicted_histogram = np.bincount(predicted, minlength=class_count).astype(np.float64)
    expected = np.outer(actual_histogram, predicted_histogram) / len(actual)
    indices = np.arange(class_count, dtype=np.float64)
    weights = ((indices[:, None] - indices[None, :]) ** 2) / ((class_count - 1) ** 2)
    denominator = float(np.sum(weights * expected))
    return 1.0 - float(np.sum(weights * observed)) / denominator if denominator else 1.0


def round_and_clip(predictions: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(predictions), SCORE_MIN, SCORE_MAX).astype(np.int8)


def apply_thresholds(predictions: np.ndarray, thresholds: Iterable[float]) -> np.ndarray:
    ordered = np.asarray(list(thresholds), dtype=np.float64)
    if ordered.shape != (SCORE_MAX - SCORE_MIN,):
        raise ValueError("Exactly five thresholds are required for scores 1-6")
    if np.any(np.diff(ordered) <= 0):
        raise ValueError("Thresholds must be strictly increasing")
    return (np.digitize(predictions, ordered) + SCORE_MIN).astype(np.int8)


def optimize_thresholds(
    y_true: np.ndarray,
    predictions: np.ndarray,
    passes: int = 6,
    candidate_quantiles: int = 301,
) -> tuple[np.ndarray, float]:
    """Deterministic coordinate search for five monotonic QWK thresholds."""
    cumulative_fractions = [float(np.mean(y_true <= score)) for score in range(1, 6)]
    thresholds = np.asarray(
        [np.quantile(predictions, fraction) for fraction in cumulative_fractions],
        dtype=np.float64,
    )
    # Rare labels can make neighboring quantiles equal. Add a tiny deterministic gap.
    for index in range(1, len(thresholds)):
        thresholds[index] = max(thresholds[index], thresholds[index - 1] + 1e-5)

    quantiles = np.linspace(0.002, 0.998, candidate_quantiles)
    candidate_pool = np.unique(np.quantile(predictions, quantiles))
    best_score = qwk(y_true, apply_thresholds(predictions, thresholds))

    for _ in range(passes):
        improved = False
        for index in range(len(thresholds)):
            lower = thresholds[index - 1] + 1e-6 if index else predictions.min() - 1e-6
            upper = (
                thresholds[index + 1] - 1e-6
                if index + 1 < len(thresholds)
                else predictions.max() + 1e-6
            )
            candidates = candidate_pool[(candidate_pool > lower) & (candidate_pool < upper)]
            candidates = np.unique(np.append(candidates, thresholds[index]))
            local_threshold = thresholds[index]
            local_score = best_score
            for candidate in candidates:
                trial = thresholds.copy()
                trial[index] = float(candidate)
                score = qwk(y_true, apply_thresholds(predictions, trial))
                if score > local_score + 1e-12:
                    local_threshold = float(candidate)
                    local_score = score
            if local_score > best_score + 1e-12:
                thresholds[index] = local_threshold
                best_score = local_score
                improved = True
        if not improved:
            break
    return thresholds, best_score


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def text_statistics(texts: Iterable[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for text in texts:
        words = WORD_RE.findall(text)
        word_count = len(words)
        lowercase_words = [word.lower() for word in words]
        unique_count = len(set(lowercase_words))
        char_count = len(text)
        alpha_count = sum(char.isalpha() for char in text)
        uppercase_count = sum(char.isupper() for char in text)
        digit_count = sum(char.isdigit() for char in text)
        punctuation_count = sum(char in ".,;:!?-'\"()" for char in text)
        sentence_count = max(1, len(SENTENCE_RE.findall(text)))
        paragraph_count = max(1, len([part for part in re.split(r"\n\s*\n|\r\n\s*\r\n", text) if part.strip()]))
        long_word_count = sum(len(word) >= 7 for word in words)
        very_long_word_count = sum(len(word) >= 12 for word in words)
        repeated_punctuation = len(re.findall(r"([!?.,])\1+", text))
        repeated_whitespace = len(re.findall(r" {2,}", text))

        count_features = [
            char_count,
            word_count,
            unique_count,
            sentence_count,
            paragraph_count,
            digit_count,
            punctuation_count,
            text.count(","),
            text.count(";"),
            text.count(":"),
            text.count("!"),
            text.count("?"),
            text.count("\n"),
            long_word_count,
            very_long_word_count,
            repeated_punctuation,
            repeated_whitespace,
        ]
        ratio_features = [
            _safe_ratio(unique_count, word_count),
            _safe_ratio(sum(map(len, words)), word_count),
            _safe_ratio(word_count, sentence_count),
            _safe_ratio(word_count, paragraph_count),
            _safe_ratio(uppercase_count, alpha_count),
            _safe_ratio(digit_count, char_count),
            _safe_ratio(punctuation_count, char_count),
            _safe_ratio(long_word_count, word_count),
            _safe_ratio(very_long_word_count, word_count),
        ]
        rows.append([math.log1p(value) for value in count_features] + ratio_features)
    result = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite values found in engineered text statistics")
    return result


def build_word_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        max_features=max_features,
        strip_accents="unicode",
        sublinear_tf=True,
        dtype=np.float32,
    )


def build_char_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.995,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def dataset_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    score_counts = train["score"].value_counts().sort_index()
    word_counts = train["full_text"].str.split().str.len()
    char_counts = train["full_text"].str.len()
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "score_counts": {str(int(k)): int(v) for k, v in score_counts.items()},
        "word_count_percentiles": {
            str(percentile): float(word_counts.quantile(percentile / 100))
            for percentile in (0, 25, 50, 75, 90, 95, 99, 100)
        },
        "char_count_percentiles": {
            str(percentile): float(char_counts.quantile(percentile / 100))
            for percentile in (0, 25, 50, 75, 90, 95, 99, 100)
        },
    }


def validate_input(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(train.columns) != ["essay_id", "full_text", "score"]:
        raise ValueError(f"Unexpected train columns: {list(train.columns)}")
    if list(test.columns) != ["essay_id", "full_text"]:
        raise ValueError(f"Unexpected test columns: {list(test.columns)}")
    if list(sample.columns) != ["essay_id", "score"]:
        raise ValueError(f"Unexpected sample submission columns: {list(sample.columns)}")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Missing values found in train or test data")
    if train["essay_id"].duplicated().any() or test["essay_id"].duplicated().any():
        raise ValueError("Duplicate essay IDs found")
    if not train["score"].between(SCORE_MIN, SCORE_MAX).all():
        raise ValueError("Training scores must be between 1 and 6")
    if not sample["essay_id"].equals(test["essay_id"]):
        raise ValueError("sample_submission essay IDs do not match test.csv order")


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    if list(submission.columns) != ["essay_id", "score"]:
        raise ValueError("Submission columns must be essay_id, score")
    if not submission["essay_id"].equals(test["essay_id"]):
        raise ValueError("Submission essay IDs/order do not match test.csv")
    if not submission["score"].between(SCORE_MIN, SCORE_MAX).all():
        raise ValueError("Submission scores must be between 1 and 6")
    if not pd.api.types.is_integer_dtype(submission["score"]):
        raise ValueError("Submission scores must be integers")


def choose_blend(
    y_true: np.ndarray,
    word_predictions: np.ndarray,
    char_predictions: np.ndarray,
    stats_predictions: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, float]:
    best_score = -np.inf
    best_weights: dict[str, float] | None = None
    best_predictions: np.ndarray | None = None
    for stats_weight in np.arange(0.0, 0.401, 0.05):
        for word_weight in np.arange(0.0, 0.901, 0.05):
            char_weight = 1.0 - stats_weight - word_weight
            if char_weight < 0.0:
                continue
            predictions = (
                word_weight * word_predictions
                + char_weight * char_predictions
                + stats_weight * stats_predictions
            )
            variance = float(np.var(predictions))
            if variance < 1e-12:
                continue
            slope = float(np.cov(predictions, y_true, ddof=0)[0, 1] / variance)
            calibrated = np.mean(y_true) + slope * (predictions - np.mean(predictions))
            score = qwk(y_true, round_and_clip(calibrated))
            if score > best_score:
                best_score = score
                best_weights = {
                    "word": round(float(word_weight), 4),
                    "char": round(float(char_weight), 4),
                    "stats": round(float(stats_weight), 4),
                }
                best_predictions = predictions.copy()
    assert best_weights is not None and best_predictions is not None
    return best_weights, best_predictions, float(best_score)


def train(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_df = pd.read_csv(data_dir / "sample_submission.csv")
    validate_input(train_df, test_df, sample_df)

    if args.limit:
        # Stratified sample for smoke tests only; never used for the final run.
        train_df, _ = train_test_split(
            train_df,
            train_size=args.limit,
            stratify=train_df["score"],
            random_state=args.seed,
        )
        train_df = train_df.reset_index(drop=True)

    texts = train_df["full_text"].astype(str).to_numpy()
    test_texts = test_df["full_text"].astype(str).to_numpy()
    y = train_df["score"].to_numpy(dtype=np.int8)
    train_stats = text_statistics(texts)
    test_stats = text_statistics(test_texts)

    oof_word = np.zeros(len(train_df), dtype=np.float64)
    oof_char = np.zeros(len(train_df), dtype=np.float64)
    oof_stats = np.zeros(len(train_df), dtype=np.float64)
    test_word = np.zeros(len(test_df), dtype=np.float64)
    test_char = np.zeros(len(test_df), dtype=np.float64)
    test_stats_predictions = np.zeros(len(test_df), dtype=np.float64)
    fold_metrics: list[dict] = []

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_index, valid_index) in enumerate(splitter.split(texts, y), start=1):
        fold_started = time.perf_counter()
        print(f"\nFold {fold}/{args.folds}", flush=True)

        word_vectorizer = build_word_vectorizer(args.max_word_features)
        x_train_word = word_vectorizer.fit_transform(texts[train_index])
        x_valid_word = word_vectorizer.transform(texts[valid_index])
        x_test_word = word_vectorizer.transform(test_texts)
        word_model = Ridge(alpha=args.word_alpha, solver="lsqr", tol=1e-4)
        word_model.fit(x_train_word, y[train_index])
        oof_word[valid_index] = word_model.predict(x_valid_word)
        test_word += word_model.predict(x_test_word) / args.folds
        word_vocab = len(word_vectorizer.vocabulary_)
        del word_vectorizer, word_model, x_train_word, x_valid_word, x_test_word
        gc.collect()

        char_vectorizer = build_char_vectorizer(args.max_char_features)
        x_train_char = char_vectorizer.fit_transform(texts[train_index])
        x_valid_char = char_vectorizer.transform(texts[valid_index])
        x_test_char = char_vectorizer.transform(test_texts)
        char_model = Ridge(alpha=args.char_alpha, solver="lsqr", tol=1e-4)
        char_model.fit(x_train_char, y[train_index])
        oof_char[valid_index] = char_model.predict(x_valid_char)
        test_char += char_model.predict(x_test_char) / args.folds
        char_vocab = len(char_vectorizer.vocabulary_)
        del char_vectorizer, char_model, x_train_char, x_valid_char, x_test_char
        gc.collect()

        stats_model = HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=2.0,
            random_state=args.seed + fold,
        )
        stats_model.fit(train_stats[train_index], y[train_index])
        oof_stats[valid_index] = stats_model.predict(train_stats[valid_index])
        test_stats_predictions += stats_model.predict(test_stats) / args.folds

        metrics = {
            "fold": fold,
            "word_qwk_rounded": qwk(y[valid_index], round_and_clip(oof_word[valid_index])),
            "char_qwk_rounded": qwk(y[valid_index], round_and_clip(oof_char[valid_index])),
            "stats_qwk_rounded": qwk(y[valid_index], round_and_clip(oof_stats[valid_index])),
            "word_vocab": word_vocab,
            "char_vocab": char_vocab,
            "seconds": round(time.perf_counter() - fold_started, 2),
        }
        fold_metrics.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)

    weights, oof_blend, blend_selection_qwk = choose_blend(
        y, oof_word, oof_char, oof_stats
    )
    test_blend = (
        weights["word"] * test_word
        + weights["char"] * test_char
        + weights["stats"] * test_stats_predictions
    )
    thresholds, optimized_qwk = optimize_thresholds(y, oof_blend)
    oof_final = apply_thresholds(oof_blend, thresholds)
    test_final = apply_thresholds(test_blend, thresholds)

    oof_frame = pd.DataFrame(
        {
            "essay_id": train_df["essay_id"],
            "score": y,
            "pred_word": oof_word,
            "pred_char": oof_char,
            "pred_stats": oof_stats,
            "pred_blend": oof_blend,
            "pred_final": oof_final,
        }
    )
    oof_frame.to_csv(output_dir / "oof_predictions.csv", index=False)

    test_prediction_frame = pd.DataFrame(
        {
            "essay_id": test_df["essay_id"],
            "pred_word": test_word,
            "pred_char": test_char,
            "pred_stats": test_stats_predictions,
            "pred_blend": test_blend,
            "score": test_final,
        }
    )
    test_prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    submission = pd.DataFrame({"essay_id": test_df["essay_id"], "score": test_final.astype(int)})
    validate_submission(submission, test_df)
    submission_path = Path(args.submission_path).resolve()
    submission.to_csv(submission_path, index=False)

    metrics = {
        "dataset": dataset_summary(train_df, test_df),
        "config": {
            "folds": args.folds,
            "seed": args.seed,
            "max_word_features": args.max_word_features,
            "max_char_features": args.max_char_features,
            "word_alpha": args.word_alpha,
            "char_alpha": args.char_alpha,
            "limit": args.limit,
        },
        "fold_metrics": fold_metrics,
        "oof": {
            "word_qwk_rounded": qwk(y, round_and_clip(oof_word)),
            "char_qwk_rounded": qwk(y, round_and_clip(oof_char)),
            "stats_qwk_rounded": qwk(y, round_and_clip(oof_stats)),
            "blend_qwk_rounded": qwk(y, round_and_clip(oof_blend)),
            "blend_selection_qwk_linear_calibrated": blend_selection_qwk,
            "blend_qwk_optimized": optimized_qwk,
            "blend_rmse": float(mean_squared_error(y, oof_blend) ** 0.5),
        },
        "blend_weights": weights,
        "thresholds": [round(float(value), 8) for value in thresholds],
        "prediction_counts": {
            str(int(score)): int(count)
            for score, count in sorted(Counter(test_final.tolist()).items())
        },
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "submission_path": str(submission_path),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print("\nFinal metrics", flush=True)
    print(json.dumps(metrics["oof"], indent=2), flush=True)
    print(f"Blend weights: {weights}", flush=True)
    print(f"Thresholds: {metrics['thresholds']}", flush=True)
    print(f"Submission: {submission_path}", flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the reproducible CPU essay-scoring baseline")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/cpu")
    parser.add_argument("--submission-path", default="submission.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-word-features", type=int, default=120_000)
    parser.add_argument("--max-char-features", type=int, default=160_000)
    parser.add_argument("--word-alpha", type=float, default=2.0)
    parser.add_argument("--char-alpha", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=None, help="Stratified row limit for smoke tests")
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.limit is not None and args.limit < args.folds * 6:
        parser.error("--limit must leave at least one example per score in every fold")
    return args


if __name__ == "__main__":
    train(parse_args())
