import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Prototype AI - Sheet1.csv")

# =========================
# ENCODE DATA
# =========================
encoders = {}

for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        encoders[column] = le

# =========================
# FEATURES & TARGET
# =========================
X = df.drop('Seating Risk Level', axis=1)
y = df['Seating Risk Level']

# =========================
# TRAIN MODEL
# =========================
model = RandomForestClassifier()
model.fit(X, y)

# =========================
# STREAMLIT UI
# =========================
st.title("MovAbility AI Prototype")

st.write("Adaptive Seating Risk Prediction Prototype")

# =========================
# USER INPUTS
# =========================
age = st.number_input("Age", min_value=0, max_value=100, value=5)

gender = st.selectbox(
    "Gender",
    encoders['Gender'].classes_
)

diagnosis = st.selectbox(
    "Diagnosis",
    encoders['diagnosis'].classes_
)

gmfcs = st.selectbox(
    "GMFCS",
    encoders['GMFCS'].classes_
)

pelvic_tilt = st.selectbox(
    "Pelvic Tilt",
    encoders['pelvic tilt'].classes_
)

# =========================
# PREDICT BUTTON
# =========================
if st.button("Predict Seating Risk"):

    # ENCODE USER INPUTS
    gender_encoded = encoders['Gender'].transform([gender])[0]
    diagnosis_encoded = encoders['diagnosis'].transform([diagnosis])[0]
    gmfcs_encoded = encoders['GMFCS'].transform([gmfcs])[0]
    pelvic_encoded = encoders['pelvic tilt'].transform([pelvic_tilt])[0]

    # CREATE INPUT DATAFRAME
    input_data = pd.DataFrame([[
        age,
        gender_encoded,
        diagnosis_encoded,
        gmfcs_encoded,
        pelvic_encoded
    ]], columns=X.columns)

    # PREDICT
    prediction = model.predict(input_data)[0]

    # DECODE RESULT
    prediction_label = encoders['Seating Risk Level'].inverse_transform([prediction])[0]

    # SHOW RESULT
    st.success(f"Predicted Seating Risk Level: {prediction_label}")

    # CLINICAL INTERPRETATION
    if prediction_label.lower() == "high":
        st.error("Clinical Interpretation: High seating/postural risk detected.")
        st.write("Recommended Action: Full adaptive seating assessment is recommended.")

    elif prediction_label.lower() == "moderate":
        st.warning("Clinical Interpretation: Moderate seating/postural risk detected.")
        st.write("Recommended Action: Monitor posture and reassess regularly.")

    else:
        st.info("Clinical Interpretation: Low seating/postural risk detected.")
        st.write("Recommended Action: Continue current management and monitor growth/posture.")
