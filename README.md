# Credit Default Risk and Decision Threshold Design

A machine learning project that separates two questions: how well a model estimates default risk, and how those estimates should be turned into decisions when errors have different costs.

The hypothetical use case is flagging high-risk credit card customers for human review. This is a learning project, not a production credit model or a basis for real credit decisions.

## Key results

The final model is an uncalibrated LightGBM classifier. On the held-out test set, it achieved PR-AUC 0.554, ROC-AUC 0.778, and Brier score 0.135.

The table assumes that a missed default costs five times as much as a false alert.

| Decision rule | Review rate | Precision | Recall | Cost per customer |
|---|---:|---:|---:|---:|
| **Cost-based threshold (0.17)** | **41.4%** | **38.6%** | **72.2%** | **0.5617** |
| Default threshold (0.5) | 12.0% | 66.6% | 36.1% | 0.7467 |
| Review everyone | 100.0% | 22.1% | 100.0% | 0.7788 |
| Review nobody | 0.0% | N/A | 0.0% | 1.1058 |

The cost-based threshold reduced expected cost by 24.8% relative to the default threshold. It does so by flagging far more customers, so a real deployment would also need to consider review capacity. See [results.md](results.md) for full results.

## Approach

1. Validate the data and create a stratified 80/20 train-test split.
2. Exclude `SEX`, `MARRIAGE`, `AGE`, and `EDUCATION` from the features.
3. Compare baselines and three LightGBM configurations with 5-fold cross-validation.
4. Select a model using PR-AUC and a predefined tie-breaking rule.
5. Compare uncalibrated, sigmoid, and isotonic probabilities.
6. Choose thresholds from training-set out-of-fold probabilities.
7. Run sensitivity analysis for 3:1, 5:1, and 10:1 cost ratios.
8. Refit the selected model and evaluate it once on the test set.
9. Save the model and threshold as a W&B Artifact.
10. Explore the fixed model's decision trade-offs in a Gradio threshold simulator.

## Cost definition

`cost per customer = (cost_ratio × false negatives + false positives) / customers`

The dataset does not contain actual loss or review costs, so the ratios are explicit assumptions.

## Threshold simulator

The notebook includes a Gradio app for exploring how the cost ratio and threshold change cost, review rate, precision, recall, and the confusion matrix on the test set. It is for visualization after final evaluation and is not used to change the selected model or threshold.

## Tech stack

Google Colab, pandas, scikit-learn, LightGBM, Weights & Biases, joblib, Gradio, and `ucimlrepo`.

## Run the notebook

1. Open `credit_default_wandb.ipynb` in Google Colab.
2. Add your W&B API key as a Colab secret named `WANDB_API_KEY`.
3. Select **Runtime > Run all**.

The notebook downloads the dataset from UCI and creates a `credit-default-ml` project in your W&B account. Never commit your API key to the repository.

## Repository contents

| File | Description |
|---|---|
| `credit_default_wandb.ipynb` | Complete workflow |
| `requirements.md` | Requirements, selection rules, and acceptance criteria |
| `results.md` | Decisions, results, and limitations |

## Main limitations

- Cost ratios are assumptions and strongly affect the selected threshold.
- Review capacity is not modeled.
- The data covers Taiwanese card customers in 2005, and out-of-time validation is not possible.
- Excluding demographic columns does not establish fairness, and no fairness audit was performed.

## Data and citation

This project retrieves the dataset at runtime and does not redistribute it. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
