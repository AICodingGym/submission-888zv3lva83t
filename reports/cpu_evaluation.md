# CPU OOF Evaluation

## Summary

| Metric | Value |
|---|---:|
| QWK, thresholds fit on all OOF | 0.804392 |
| QWK, cross-fitted thresholds | 0.803019 |
| Exact accuracy | 0.610 |
| Within one score | 0.982 |
| Mean absolute error | 0.409 |
| Discrete RMSE | 0.669 |
| CPU training runtime | 243.88 seconds |

The cross-fitted threshold score is the conservative validation estimate. The all-OOF score is useful for selecting the final thresholds applied to the test set, but is mildly optimistic because those thresholds were fitted on the same OOF labels.

Blend weights: `{"word": 0.4, "char": 0.2, "stats": 0.4}`

Final thresholds: `[1.97907043, 2.63767415, 3.33197545, 3.99166019, 4.76486588]`

## Confusion Matrix

| Actual \ Predicted | 1 | 2 | 3 | 4 | 5 | 6 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 561 | 465 | 90 | 8 | 0 | 0 |
| 2 | 589 | 2728 | 868 | 60 | 4 | 0 |
| 3 | 19 | 1125 | 3425 | 1005 | 53 | 2 |
| 4 | 0 | 26 | 836 | 2152 | 542 | 7 |
| 5 | 0 | 0 | 3 | 237 | 570 | 66 |
| 6 | 0 | 0 | 0 | 3 | 72 | 60 |

## Per-score Diagnostics

| Actual score | N | Exact accuracy | Within one | MAE | Mean prediction |
|---:|---:|---:|---:|---:|---:|
| 1 | 1124 | 0.499 | 0.913 | 0.595 | 1.595 |
| 2 | 4249 | 0.642 | 0.985 | 0.374 | 2.097 |
| 3 | 5629 | 0.608 | 0.987 | 0.405 | 2.992 |
| 4 | 3563 | 0.604 | 0.991 | 0.405 | 3.907 |
| 5 | 876 | 0.651 | 0.997 | 0.353 | 4.798 |
| 6 | 135 | 0.444 | 0.978 | 0.578 | 5.422 |

The hardest classes are the rare extremes. Score 1 is biased upward and score 6 is biased downward; this is consistent with class imbalance and regression toward the mean. Nevertheless, more than 98% of OOF predictions are within one point of the human score.
