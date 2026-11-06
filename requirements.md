# Requirements: Credit Default Risk and Decision Threshold Design

## 1. Objective

Build a model that estimates the probability of credit card default and design a decision threshold based on asymmetric misclassification costs.

The hypothetical use case is identifying high-risk customers for human review, not automatically rejecting credit. This is a learning project and is not a production credit model.

## 2. Key questions

1. Which model produces the best default-risk ranking and reliable probabilities?
2. What threshold minimizes expected cost when missed defaults cost more than false alerts?
3. How much does the threshold and resulting cost change under different cost assumptions?

## 3. Scope

### In scope

- Data validation, splitting, and preprocessing
- Baseline and LightGBM model comparison
- Probability calibration comparison
- Cost-based threshold selection and sensitivity analysis
- Experiment tracking and model versioning in W&B
- Final test-set evaluation
- A Gradio threshold simulator
- Publishing the code and documentation on GitHub in English

### Out of scope

- Production deployment or automated credit decisions
- Customer-level prediction demos
- Resampling such as SMOTE and class weighting
- Automated hyperparameter search
- Fairness audit and out-of-time validation

## 4. Environment

| Item | Selection |
|---|---|
| Runtime | Google Colab |
| Data access | `ucimlrepo` |
| Data processing | pandas |
| Modeling and evaluation | scikit-learn, LightGBM |
| Experiment tracking | W&B |
| Model persistence | joblib, W&B Artifacts |
| Demo | Gradio |
| Publishing | GitHub |

## 5. Dataset

| Item | Details |
|---|---|
| Dataset | Default of Credit Card Clients (UCI ID 350) |
| Size | 30,000 rows and 23 input variables |
| Target | Default payment next month (1 = default) |
| License | CC BY 4.0 |
| Citation | Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H |

The dataset is retrieved at runtime and is not stored in the repository.

### Features

| Group | Columns | Count |
|---|---|---:|
| Credit limit | `LIMIT_BAL` | 1 |
| Repayment status | `PAY_0`, `PAY_2` to `PAY_6` | 6 |
| Bill amount | `BILL_AMT1` to `BILL_AMT6` | 6 |
| Payment amount | `PAY_AMT1` to `PAY_AMT6` | 6 |

`SEX`, `MARRIAGE`, `AGE`, and `EDUCATION` are excluded to avoid directly using demographic attributes. The remaining features may still act as proxies, so this exclusion does not establish fairness.

## 6. Functional requirements

| ID | Requirement |
|---|---|
| F-01 | Validate rows, columns, data types, missing values, duplicates, target distribution, and categorical codes |
| F-02 | Create a stratified 80/20 train-test split with a fixed random seed |
| F-03 | Keep the test set unused until final evaluation |
| F-04 | Combine required preprocessing and the model in one pipeline |
| F-05 | Evaluate majority-class prediction and logistic regression as baselines |
| F-06 | Evaluate three manually defined LightGBM configurations |
| F-07 | Use stratified 5-fold cross-validation on the training set |
| F-08 | Select a model using cross-validation results only |
| F-09 | Compare calibrated and uncalibrated probabilities using the training set |
| F-10 | Adopt calibration only if it meets the rule in section 8 |
| F-11 | Compare thresholds using out-of-fold probabilities from the training set |
| F-12 | Run threshold sensitivity analysis for 3:1, 5:1, and 10:1 cost ratios |
| F-13 | Evaluate the selected model and fixed thresholds on the test set |
| F-14 | Log the cost and calibration curves to W&B |
| F-15 | Save the final pipeline and selected threshold as a W&B Artifact |
| F-16 | Provide a Gradio simulator showing how cost ratio and threshold affect decision results after final evaluation; it must not be used to change the selected model or threshold |

## 7. Preprocessing

- Treat all 19 features as numeric, including repayment status codes.
- Standardize features for logistic regression.
- Add median imputation only if data validation finds missing values.
- Fit all preprocessing on training folds only.
- If validation finds undocumented codes, document them and retain them as observed values unless they prevent modeling.

## 8. Evaluation and selection

### Model selection

| Item | Rule |
|---|---|
| Primary metric | Mean cross-validation PR-AUC |
| Secondary metrics | Brier score, ROC-AUC, and calibration curve |
| Selection | Choose the model with the highest mean PR-AUC |
| Tie-breaking | If the difference is within fold-to-fold variation, choose the simpler model |

### Calibration

Adopt calibration only if it improves the mean Brier score by more than its fold-to-fold standard deviation without reducing mean PR-AUC by more than its standard deviation.

### Cost and threshold

Cost per customer is defined as:

`(cost_ratio × FN + 1 × FP) / number_of_customers`

| Scenario | Missed default : false alert | Theoretical threshold |
|---|---:|---:|
| Low | 3:1 | 0.250 |
| Base | 5:1 | 0.167 |
| High | 10:1 | 0.091 |

These ratios are assumptions because the dataset contains no actual loss or review costs. The theoretical threshold applies only to reasonably calibrated probabilities.

The following thresholds are compared:

- Default threshold: 0.5
- Theoretical threshold for each cost ratio
- Cost-minimizing threshold selected from training-set out-of-fold probabilities

The base-case cost-minimizing threshold is the primary threshold. The test set must not be used to select or adjust it.

### Final evaluation

Report these results on the test set:

- PR-AUC and Brier score
- Cost per customer under each scenario
- Confusion matrix, precision, and recall for each compared threshold
- Cost comparison with two benchmarks: review nobody and review everyone

## 9. Experiment tracking

| Item | Logged information |
|---|---|
| Model | Model name and hyperparameters |
| Data | Dataset ID, features, split ratio, and target rate |
| Reproducibility | Random seed and notebook version label |
| Validation | Mean and standard deviation of cross-validation metrics |
| Decision | Cost ratios, thresholds, and costs |
| Outputs | Final pipeline, threshold, and evaluation curves |

The Git commit ID is added to the results and W&B runs once the repository is available and the notebook is rerun.

## 10. Process

| Step | Work | Output |
|---|---|---|
| 1. Setup | Create the notebook and connect to W&B | Notebook, W&B project |
| 2. Data validation | Load and validate the dataset | Validation results |
| 3. Data split | Create training and test sets | Split datasets |
| 4. Preprocessing | Define model pipelines | Pipelines |
| 5. Baseline | Evaluate majority-class prediction and logistic regression | Baseline runs |
| 6. LightGBM | Evaluate three configurations | LightGBM runs |
| 7. Model selection | Compare cross-validation results | Selected model |
| 8. Calibration | Compare calibration options | Calibration decision |
| 9. Threshold design | Select thresholds and analyze sensitivity | Threshold decision, cost curves |
| 10. Final evaluation | Refit on the training set and test once | Test results |
| 11. Model saving | Save the pipeline and threshold | W&B Artifact |
| 12. Demo | Build and check the threshold simulator | Gradio demo |
| 13. Reporting | Record decisions, results, and limitations | `results.md`, `README.md` |
| 14. Publishing | Rerun from a new session and publish on GitHub | Public repository |

## 11. Non-functional requirements

| Item | Requirement |
|---|---|
| Reproducibility | Use fixed random seeds and run the notebook from top to bottom |
| Compute | Run on a standard Colab CPU |
| Security | Store the W&B API key only in Colab secrets |
| Visibility | Keep the W&B project private, avoid public Gradio links, and clear notebook outputs before publishing |
| Attribution | Cite the dataset and license in the public repository |

## 12. Acceptance criteria

- [ ] The notebook runs from top to bottom in a new session.
- [ ] The four demographic columns are not used as features.
- [ ] The test set is not used for model, calibration, or threshold selection.
- [ ] All selected models and thresholds are traceable in W&B.
- [ ] The final model's PR-AUC exceeds the majority-class baseline.
- [ ] Results compare costs across thresholds and cost ratios.
- [ ] Test-set cost is compared with review-nobody and review-everyone benchmarks.
- [ ] The final pipeline and threshold are recorded as a W&B Artifact.
- [ ] The Gradio simulator updates results when the cost ratio or threshold changes.
- [ ] Assumptions and limitations are stated in the results and README.
- [ ] The repository is public, English-language, and free of secrets.

## 13. Assumptions and limitations

- Cost ratios are assumptions, not observed financial costs.
- Data covers Taiwanese card customers in 2005, so performance on later periods or other populations is unknown.
- Excluding demographic features does not ensure fairness, and no fairness audit is performed.
- This project does not satisfy the legal, regulatory, or model-risk requirements for real credit decisions.
