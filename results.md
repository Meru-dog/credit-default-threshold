# Results: Credit Default Risk and Decision Threshold Design

## 1. Summary

An uncalibrated LightGBM model (`lgbm-small`) was selected. Under the base assumption that a missed default costs five times as much as a false alert, the training-set selected threshold of 0.17 achieved a test cost of 0.5617 per customer.

This was 24.8% lower than the default 0.5 threshold, 27.9% lower than reviewing everyone, and 49.2% lower than reviewing nobody. The trade-off is operational: the threshold flags 41.4% of customers for review, with 38.6% precision and 72.2% recall.

These results depend on an assumed cost ratio. This is a learning project, not a production credit model.

## 2. Data

| Item | Details |
|---|---|
| Dataset | Default of Credit Card Clients (UCI ID 350) |
| Rows | 30,000 |
| Default rate | 22.12% |
| Split | 24,000 training rows and 6,000 test rows (stratified) |
| Features | 19 credit limit, repayment, bill, and payment variables |
| Excluded | `SEX`, `MARRIAGE`, `AGE`, and `EDUCATION` |

### Data validation decisions

| Finding | Decision | Reason |
|---|---|---|
| 35 duplicate rows | Retained | No customer ID is available, so they may represent different customers |
| 1,930 rows with a negative bill amount | Retained | They are plausible but unverified, so they were kept as observed |
| `PAY_` codes `-2` and `0` | Retained as numeric values | They are outside the documented code description and were kept as observed |

## 3. Model selection

Mean and standard deviation are from stratified 5-fold cross-validation on the training set.

| Run | PR-AUC | PR-AUC std. dev. | Brier | ROC-AUC |
|---|---:|---:|---:|---:|
| `lgbm-medium` | 0.5560 | 0.0061 | 0.1344 | 0.7802 |
| `lgbm-small` | 0.5555 | 0.0097 | 0.1339 | 0.7824 |
| `lgbm-large` | 0.5471 | 0.0064 | 0.1365 | 0.7738 |
| `baseline-logistic-regression` | 0.5065 | 0.0134 | 0.1449 | 0.7252 |
| `baseline-majority` | 0.2212 | 0.0001 | 0.1723 | 0.5000 |

`lgbm-medium` had the highest PR-AUC, but its 0.0005 advantage over `lgbm-small` was within fold-to-fold variation. Under the predefined tie-breaking rule, the simpler `lgbm-small` was selected. It also had the best Brier score and ROC-AUC of the LightGBM configurations.

## 4. Calibration

| Option | PR-AUC | Brier | ROC-AUC |
|---|---:|---:|---:|
| Uncalibrated | 0.5555 | 0.1339 | 0.7824 |
| Sigmoid | 0.5559 | 0.1346 | 0.7825 |
| Isotonic | 0.5531 | 0.1340 | 0.7825 |

Calibration was not adopted because neither method improved the Brier score by more than its standard deviation.

## 5. Threshold design

Thresholds were selected using training-set out-of-fold probabilities.

| Scenario | Missed default : false alert | Theoretical threshold | Selected threshold | Selected-threshold cost |
|---|---:|---:|---:|---:|
| Low | 3:1 | 0.250 | 0.27 | 0.3980 |
| Base | 5:1 | 0.167 | 0.17 | 0.5483 |
| High | 10:1 | 0.091 | 0.09 | 0.7017 |

The primary threshold of 0.17 was fixed before test evaluation. The best base-case thresholds across folds ranged from 0.15 to 0.21.

## 6. Final test evaluation

### Probability metrics

| Metric | Cross-validation | Test set |
|---|---:|---:|
| PR-AUC | 0.5555 | 0.5541 |
| ROC-AUC | 0.7824 | 0.7783 |
| Brier score | 0.1339 | 0.1353 |

### Base-case decision results (5:1)

| Decision rule | Review rate | Precision | Recall | Cost per customer |
|---|---:|---:|---:|---:|
| Threshold 0.17 | 41.4% | 38.6% | 72.2% | 0.5617 |
| Theoretical threshold 0.167 | 42.4% | 38.0% | 72.9% | 0.5630 |
| Default threshold 0.5 | 12.0% | 66.6% | 36.1% | 0.7467 |
| Review everyone | 100.0% | 22.1% | 100.0% | 0.7788 |
| Review nobody | 0.0% | N/A | 0.0% | 1.1058 |

### Confusion matrix at threshold 0.17

| | Not reviewed | Reviewed |
|---|---:|---:|
| **Actual: no default** | 3,148 | 1,525 |
| **Actual: default** | 369 | 958 |

### Sensitivity analysis on the test set

| Scenario | Selected threshold | Cost | Review everyone | Review nobody | Threshold 0.5 |
|---|---:|---:|---:|---:|---:|
| Low (3:1) | 0.27 | 0.4050 | 0.7788 | 0.6635 | 0.4640 |
| Base (5:1) | 0.17 | 0.5617 | 0.7788 | 1.1058 | 0.7467 |
| High (10:1) | 0.09 | 0.6967 | 0.7788 | 2.2117 | 1.4533 |

In the low-cost scenario, the theoretical threshold of 0.25 had a lower test cost (0.3982) than the selected 0.27. Thresholds were not changed after test evaluation.

## 7. W&B outputs

| Output | Name |
|---|---|
| Project | `credit-default-ml` (private) |
| Calibration run | `calibration-comparison` |
| Threshold run | `threshold-design` |
| Final evaluation run | `final-lgbm-small` |
| Model Artifact | `credit-default-lgbm-small` |

## 8. Acceptance review

| Criterion | Status |
|---|---|
| Notebook runs from top to bottom in a new session | Passed |
| Demographic columns are not used as features | Passed |
| Test set is not used for model, calibration, or threshold selection | Passed |
| Selected models and thresholds are traceable in W&B | Passed |
| Final model's PR-AUC exceeds the majority baseline | Passed |
| Costs are compared across thresholds and cost ratios | Passed |
| Test cost is compared with both benchmarks | Passed |
| Final pipeline and threshold are recorded as a W&B Artifact | Passed |
| Gradio simulator updates with cost ratio and threshold | Passed |
| Assumptions and limitations are documented | Passed |
| Repository is public, in English, and free of secrets | Pending |

## 9. Limitations

- The cost ratios are assumptions, not observed financial or review costs.
- A review capacity limit was not modeled, although the selected threshold flags 41.4% of customers.
- The 2005 Taiwanese data does not establish performance for later periods or other populations.
- All rows represent a single observation window, so out-of-time validation was not possible.
- Excluding demographic columns does not establish fairness. Remaining features may act as proxies, and no fairness audit was performed.
- Treating `PAY_` codes as numeric assumes an ordering that may not be meaningful for all codes.
- Standard-deviation comparisons are simple heuristics, not statistical tests.
- W&B runs record a notebook version label but not a Git commit ID.
- This project does not satisfy the legal, regulatory, or model-risk requirements for real credit decisions.
