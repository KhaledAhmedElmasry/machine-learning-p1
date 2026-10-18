# Rush Hour: predicting bike-sharing demand (Machine Learning, Project 1)

Everything is in one notebook, **`bike_sharing_pipeline.ipynb`**: five chained phases on two years of hourly
Capital Bikeshare records, then the test predictions.

| # | Phase | What it does |
|---|---|---|
| 1 | Gradient descent | our own batch gradient descent, linear model |
| 2 | Polynomial regression | feature expansion, gradient descent warm-started from Phase 1 |
| 3 | Bias-variance tradeoff | complexity ladder; random vs day-grouped vs chronological evaluation |
| 4 | Regularization | Ridge, Lasso, Elastic Net (our own proximal gradient descent) |
| 5 | Logistic regression | "is this a high-demand hour?" on the features Phase 4 kept |
| 6 | Test predictions | refit of the recommended regression model, written to `sample_submission.csv` |

## How the phases are chained

Run the cells from the top to the bottom. Each phase starts from what the previous phase produced: the weight
vector and its hyper-parameters are kept in memory (`ARTIFACTS`) together with the SHA-256 of the weights, and every
phase first re-checks that hash. Each phase begins with a cell holding the functions it uses (written once and reused
by the later phases); two functions with the same name in two phases carry a suffix (`fit_poly_p2`, `fit_poly_p3`).
Nothing is written to disk except `sample_submission.csv`.

## Data and split

`Datasets/train.csv` (10,886 labelled hours, days 1-19 of every month), `Datasets/test.csv` (574 hours, the 20th of
every month, no labels) and `Datasets/sample_submission.csv`. No other data is used.

The team seed is `int(sha256("_".join(sorted(ids))).hexdigest(), 16) % 100000` with the three team IDs
(16002932, 16004208, 16006616), which gives **42648**. The only seeded split is
`train_test_split(train, test_size=0.20, random_state=42648)`. Scalers, encoders, polynomial expansions and feature
selection are fitted on training rows only; `test.csv` is only ever transformed. No sklearn model is ever fitted
(sklearn is used for the seeded split, `GroupKFold` and metric cross-checks).

## Running it

Python 3.14 with numpy 2.5, pandas 3.0, scipy 1.18, scikit-learn 1.9, matplotlib 3.11 and Jupyter.
Put the notebook next to the `Datasets/` folder and run all cells (about 6 minutes on a free machine, up to
about 20 on a loaded one):

```bash
jupyter nbconvert --to notebook --execute --inplace bike_sharing_pipeline.ipynb
```

The last cells print the SHA-256 of every phase's weights and of `sample_submission.csv`, so two runs can be compared.

## Files

```
bike_sharing_pipeline.ipynb   the whole project (code, results, Expectation / Outcome notes)
sample_submission.csv         predictions for test.csv (instant, cnt), written by the notebook
Datasets/                     train.csv, test.csv, sample_submission.csv (the template)
```
