# Diamond pricer

Interactive dashboard to estimate the value of a diamond based on its characteristics (weight, size, cut, clarity, color).
Data sourced from Kaggle: [Data Analysis on Diamonds Dataset](https://www.kaggle.com/datasets/swatikhedekar/price-prediction-of-diamond)

Quick exercise on a medium, well-populated dataset:
- Data preprocessing (exploration, imputation, transformation): 1_exploration.ipynb 
- Data modelling (split, regression model selection, test scoring): 2_model_selection.ipynb 
- Model prediction interval (error distribution analysis, quantile modelling): 3_model_error.ipynb 
- API (Flask) to call through POST to get model results: diam_price_api.py
- App / dashboard (Streamlit) to simulate user interface: diam_pricer.py

The raw data, final models used, and custom functions called can also be found in this repo:
- diamonds.csv for the raw dataset
- functions.py for the custom functions called elsewhere
- model.joblib, q1_model.joblib, q2_model.joblib for the models
