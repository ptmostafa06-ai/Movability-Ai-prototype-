import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title("MovAbility AI Prototype")
st.write("Adaptive Seating Risk Prediction Prototype")

# Load data
df = pd.read_csv("Prototype AI - Sheet1.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

target_column = "seating risk level"

if target_column not in df.columns:
    st.error("Column 'seating risk level' not found in CSV file.")
    st.write("Available columns are:")
    st.write(list(df.columns))
    st.stop()

# Remove empty rows
df = df.dropna(how="all")

# Fill missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

# Encode categorical columns
encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# Features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.subheader("Enter patient information")

user_input = {}

for col in X.columns:
    if col in encoders:
        selected = st.selectbox(col.title(), encoders[col].classes_)
        user_input[col] = encoders[col].transform([selected])[0]
    else:
        user_input[col] = st.number_input(col.title(), value=float(df[col].median()))

input_data = pd.DataFrame([user_input])

if st.button("Predict Seating Risk"):
    prediction = model.predict(input_data)[0]

    if target_column in encoders:
        prediction_label = encoders[target_column].inverse_transform([prediction])[0]
    else:
        prediction_label = prediction

    st.success(f"Predicted Seating Risk Level: {prediction_label}")

    prediction_text = str(prediction_label).lower()

    if prediction_text == "high":
        st.error("Clinical Interpretation: High seating/postural risk detected.")
        st.write("Recommended Action: Full adaptive seating assessment is recommended.")

    elif prediction_text == "moderate":
        st.warning("Clinical Interpretation: Moderate seating/postural risk detected.")
        st.write("Recommended Action: Monitor posture and reassess regularly.")

    else:
        st.info("Clinical Interpretation: Low seating/postural risk detected.")
        st.write("Recommended Action: Continue current management and monitor growth/posture.")
