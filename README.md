# 🏠 Gurugram House Price Predictor

A machine learning web application that predicts residential property prices in Gurugram, India, based on area, locality, and number of bedrooms (BHK). Built with Python and Scikit-Learn, deployed as an interactive Streamlit app.

**Live demo:** https://gurugram-house-price-predictor-9mvfp3t3q9pc4ykwppq76h.streamlit.app/

---

## Overview

Real estate pricing in Gurugram varies significantly by locality and property size, making it hard for buyers and sellers to gauge fair value quickly. This project trains a regression model on 17,000+ real-world property listings and exposes it through a simple web interface, allowing a user to enter property details and get an instant estimated price.

## Features

- Predicts property price from three inputs: area (sq ft), locality, and BHK count
- Cleans and preprocesses a real-world, messy real estate dataset (handles inconsistent price formatting and missing values)
- Encodes categorical locality data for use in the model
- Trains a Random Forest Regression model on the cleaned dataset
- Interactive Streamlit UI with a dataset preview panel
- Deployed and publicly accessible via Streamlit Community Cloud

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas |
| Machine Learning | Scikit-Learn (Random Forest Regression, Label Encoding) |
| Web App / Deployment | Streamlit, Streamlit Community Cloud |
| Version Control | Git, GitHub |

## Dataset

- Source: [Gurgaon Real Estate Dataset (Kaggle)](https://www.kaggle.com/datasets/nikhilmehrahr26/gurgaon-real-estate-dataset)
- 17,000+ property listings across Gurugram
- Used columns: `Price`, `Area`, `Locality`, `BHK_Count`

## How It Works

1. **Data loading & cleaning** — The raw dataset is loaded with Pandas. The `Price` column (originally formatted with commas as text) is converted to numeric, and the `Area` column is coerced to numeric as well. Rows with missing values in the relevant columns are dropped.
2. **Feature encoding** — `Locality`, a categorical text field, is converted into numeric form using Scikit-Learn's `LabelEncoder` so it can be used as a model input.
3. **Model training** — A `RandomForestRegressor` (100 estimators) is trained on `Area`, `Locality`, and `BHK_Count` as features, with `Price` as the target.
4. **Prediction** — The Streamlit interface collects user input (area, BHK, locality) and feeds it to the trained model to return an estimated price in real time.

## Project Structure

```
Gurugram-House-Price-Predictor/
├── app.py              # Streamlit app: data pipeline, model training, UI
├── gurugram.csv         # Raw dataset (17,000+ listings)
├── requirements.txt     # Python dependencies
└── README.md
```

## Running Locally

```bash
git clone https://github.com/GauravSh-7/Gurugram-House-Price-Predictor.git
cd Gurugram-House-Price-Predictor
pip install -r requirements.txt
streamlit run app.py
```

## Limitations & Future Improvements

- The model currently trains on the full dataset with no train/test split, so there's no held-out accuracy metric (e.g. RMSE, R²) yet — adding this is a natural next step to validate prediction quality.
- Only three features are used (`Area`, `Locality`, `BHK_Count`); incorporating more dataset fields (e.g. property age, amenities, floor number) could improve accuracy.
- The model is retrained on every app run rather than being cached or persisted — caching the trained model would improve load time.
- No automated tests currently cover the data-cleaning or prediction logic.

## Author

[Gaurav Sh.](https://github.com/GauravSh-7)
