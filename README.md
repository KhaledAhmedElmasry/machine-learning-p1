# Rush Hour: predicting bike-sharing demand (Machine Learning, Project 1)

A chain of five phases on two years of hourly Capital Bikeshare records. Each phase implements one
course algorithm from scratch (NumPy / pandas) and hands a numeric artifact to the next one.

| Phase | Algorithm | Notebook | Module | Artifact handed to the next phase |
|---|---|---|---|---|
| 1 | Gradient descent (linear model) | `gradient_descent.ipynb` | `gradient_descent.py` | `gradient_descent_artifacts.*` |
| 2 | Polynomial regression (warm-started from Phase 1) | `polynomial_regression.ipynb` | `polynomial_regression.py` | `polynomial_regression_artifacts.*` |
| 3 | Bias-variance tradeoff | `bias_variance.ipynb` | `bias_variance.py` | `bias_variance_artifacts.*` |
| 4 | Regularization (Ridge, Lasso, Elastic Net) | `regularization.ipynb` | `regularization.py` | `regularization_artifacts.*` |
| 5 | Logistic regression ("is this a high-demand hour?") | `logistic_regression.ipynb` | `logistic_regression.py` | `logistic_regression_artifacts.*` |

Every artifact is a `.npz` (weights, bias, split indices) plus a `.json` (parameters, metrics and the
SHA-256 of the weights). Each phase recomputes the previous phase's hash before it starts, so a phase that
was not built on the team's own previous phase fails loudly.

`make_submission.py` refits the Phase 3 model on all labelled rows with our own gradient descent, floors the
predictions at 0 and writes `sample_submission.csv` (columns `instant`, `cnt`) plus an audit record
`submission_fit.json`. `Datasets/sample_submission.csv` is the untouched template.

## Data and split

`Datasets/train.csv` (10,886 labelled hours, days 1-19 of every month), `Datasets/test.csv` (574 hours, the
20th of every month, no labels) and `Datasets/sample_submission.csv`.

The team seed is `int(sha256("_".join(sorted(ids))).hexdigest(), 16) % 100000` with the three team IDs
(16002932, 16004208, 16006616), which gives **42648**. The only seeded split is
`train_test_split(train, test_size=0.20, random_state=42648)`. Scalers, encoders, polynomial expansions and
feature selection are fitted on training rows only; `test.csv` is only ever transformed.

## Running it

Python 3.14, numpy 2.5, pandas 3.0, scikit-learn 1.9, scipy 1.18, matplotlib 3.11 (scikit-learn is used for the
seeded split, `GroupKFold` and metric cross-checks only; no sklearn model is ever fitted).
For notebooks: `pip install jupyter nbconvert` if they are not installed.

Run the notebooks in order, from this folder, so that each phase finds the previous phase's artifact:

```bash
jupyter nbconvert --to notebook --execute --inplace gradient_descent.ipynb
jupyter nbconvert --to notebook --execute --inplace polynomial_regression.ipynb
jupyter nbconvert --to notebook --execute --inplace bias_variance.ipynb
jupyter nbconvert --to notebook --execute --inplace regularization.ipynb
jupyter nbconvert --to notebook --execute --inplace logistic_regression.ipynb
python make_submission.py
```

Phases 1 and 2 take a few minutes, Phases 3 to 5 and the submission each take from several minutes to about an
hour because the models have several hundred features and are trained with plain gradient descent.
Re-running Phase 1 rewrites its weights; a different NumPy build can change the last bits and therefore the
SHA-256 that Phase 2 checks, so re-run the whole chain rather than a single phase.

## Layout

```
Datasets/                 inputs (read-only)
*.py                      reusable functions of each phase
*.ipynb                   one notebook per phase (Expectation ... work ... Outcome)
*_artifacts.npz / .json   numeric hand-off between phases
phase2_degree_table.csv   Phase 2 degree sweep
plots/                    figures written by the notebooks
make_submission.py        final refit and test predictions
sample_submission.csv     predictions for test.csv (instant, cnt)
submission_fit.json       audit record of the submission fit
```
