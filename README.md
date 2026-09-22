# 🚗 Car Price Predictor

A Streamlit web application that predicts an estimated used-car price from vehicle details using a regression model.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project files

- `app.py` — Streamlit web application
- `car_price_model.pkl` — trained regression model
- `encoders.pkl` — categorical encoders
- `cat_cols.pkl` — categorical feature list
- `feature_columns.pkl` — model feature order
- `numeric_ranges.pkl` — valid numeric input ranges
- `train_model.py` — training script
- `Linear_Ridge_&_Lasso_Regression_(2) (1).ipynb` — cleaned analysis notebook
- `.streamlit/config.toml` — app theme/server configuration
- `requirements.txt` — pinned runtime dependencies

## Retraining the model

If you want to rerun `train_model.py`, install the training dependencies first:

```bash
pip install -r requirements-train.txt
python train_model.py
```

The web app itself does **not** need `kagglehub`; it uses the saved `.pkl` artifacts.

## Deployment

For Streamlit Community Cloud, push the project to GitHub and deploy `app.py`. Keep `requirements.txt` in the repository root so the deployment environment installs the same dependencies used by the app.

The saved model was created with scikit-learn 1.6.1, so the runtime pins that version to avoid model-version incompatibility warnings/errors.

## Important model note

The target is `Price`. The training features must exclude both `Price` and `Log_price`; including `Price` as an input would cause target leakage.
