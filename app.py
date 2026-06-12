import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Gurugram House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Gurugram House Price Predictor")

st.write(
    "Predict property prices in Gurugram using Machine Learning."
)

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("gurugram.csv")

# Clean Price column
df["Price"] = (
    df["Price"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["Price"] = pd.to_numeric(
    df["Price"],
    errors="coerce"
)

# Clean Area column
df["Area"] = pd.to_numeric(
    df["Area"],
    errors="coerce"
)

# Keep only useful columns
df = df[
    [
        "Price",
        "Area",
        "Locality",
        "BHK_Count"
    ]
]

# Remove missing values
df = df.dropna()

# -----------------------------
# Encode Locality
# -----------------------------

le = LabelEncoder()

df["Locality"] = le.fit_transform(
    df["Locality"]
)

# -----------------------------
# Features & Target
# -----------------------------

X = df[
    [
        "Area",
        "Locality",
        "BHK_Count"
    ]
]

y = df["Price"]

# -----------------------------
# Train Model
# -----------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# -----------------------------
# User Inputs
# -----------------------------

st.subheader("Property Details")

col1, col2 = st.columns(2)

with col1:

    area = st.number_input(
        "Area (sq ft)",
        min_value=100,
        value=1500
    )

    bhk = st.number_input(
        "BHK",
        min_value=1,
        value=3
    )

with col2:

    locality_name = st.selectbox(
        "Locality",
        sorted(le.classes_)
    )

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict Price"):

    locality_encoded = le.transform(
        [locality_name]
    )[0]

    prediction = model.predict(
        [[
            area,
            locality_encoded,
            bhk
        ]]
    )

    st.success(
        f"Estimated Property Price: ₹{prediction[0]:,.0f}"
    )

# -----------------------------
# Dataset Preview
# -----------------------------

with st.expander(
    "📊 View Dataset Preview"
):

    st.dataframe(
        df.head()
    )