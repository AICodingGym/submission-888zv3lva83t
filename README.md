# Automated Essay Scoring 2 — CPU and A800 Pipeline

This repository contains a reproducible CPU-first solution for predicting holistic essay scores from 1 to 6.

## Approach

The pipeline uses stratified out-of-fold validation and three complementary regressors:

1. word-level TF-IDF (unigrams and bigrams) with Ridge regression;
2. character-level TF-IDF (3-5 character `char_wb` n-grams) with Ridge regression;
3. engineered length, structure, punctuation, casing, and vocabulary features with histogram gradient boosting.

The validated Ridge regularization values are `alpha=2` for word features and `alpha=4` for character features.

The three continuous predictions are blended using weights selected on out-of-fold predictions. Five monotonic thresholds are then optimized on the blended OOF predictions to maximize quadratic weighted kappa (QWK). The same weights and thresholds are applied to averaged test-fold predictions.

This is an ordinal regression design: it preserves the ordering and distance between scores, unlike an ordinary six-class classifier.

## Setup

From this directory in PowerShell:

```powershell
& 'C:\Users\Cathy\AppData\Local\Programs\Python\Python312\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Smoke test

```powershell
.\.venv\Scripts\python.exe src\train_cpu.py --folds 2 --limit 1200 --max-word-features 20000 --max-char-features 30000 --output-dir outputs\smoke --submission-path outputs\smoke_submission.csv
```

## Full CPU training

```powershell
.\.venv\Scripts\python.exe src\train_cpu.py
```

Outputs:

- `submission.csv`: validated integer predictions in the required format;
- `outputs/cpu/metrics.json`: configuration, fold metrics, OOF metrics, weights, thresholds, and runtime;
- `outputs/cpu/oof_predictions.csv`: row-level OOF predictions for diagnostics and future ensembling;
- `outputs/cpu/test_predictions.csv`: component and blended test predictions.

Generate the conservative cross-fitted threshold evaluation and error analysis:

```powershell
.\.venv\Scripts\python.exe src\evaluate_oof.py
```

This creates `reports/cpu_evaluation.md` and `reports/cpu_evaluation.json`. Use the cross-fitted threshold QWK as the conservative validation estimate; the all-OOF threshold score is mildly optimistic and is used to choose the final test thresholds.

## Submission

From the parent AI Coding Gym workspace, submit only after reviewing `submission.csv` and `outputs/cpu/metrics.json`:

```powershell
..\.venv\bin\aicodinggym.exe mle submit learning-agency-lab-automated-essay-scoring-2 -F learning-agency-lab-automated-essay-scoring-2\submission.csv
```

## A800 training plan

The GPU model is a five-fold `microsoft/deberta-v3-large` continuous-score
regressor.  Its OOF predictions are blended with the CPU baseline, then five
monotonic score thresholds are optimized for QWK.  This keeps all blending and
threshold choices grounded in held-out predictions.

On the A800 machine, keep source, model cache, checkpoints, and outputs on the
data disk (`/root/autodl-tmp`) rather than the small system disk:

```bash
cd /root/autodl-tmp/essay-scoring
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/hf-cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_ENDPOINT=https://hf-mirror.com
pip install -r requirements-gpu.txt
python src/train_deberta.py --epochs 3 --max-length 1024 --train-batch-size 8 --gradient-accumulation 4
python src/blend_gpu_cpu.py
```

The latter command creates `submission_gpu_ensemble.csv`; validate and copy it
back locally before uploading it through the AI Coding Gym CLI.

## Completed A800 run

The completed five-fold run used three epochs per fold and finished in 4.48
hours. The DeBERTa model achieved an OOF rounded QWK of `0.82801`. The
OOF-selected ensemble uses 80% DeBERTa predictions and 20% CPU predictions;
its conservative cross-fitted threshold QWK was `0.83685`.

The validated competition file is `submission_gpu_ensemble.csv` (1,731 rows,
integer scores 1--6). It has been produced locally but is not uploaded by any
script in this repository.

For the configured remote A800 run, `scripts/watch_and_fetch_gpu_results.ps1`
waits for the remote ensemble metrics and copies the submission plus prediction
artifacts back to this project. It uses local SSH-key authentication and never
stores a password.
