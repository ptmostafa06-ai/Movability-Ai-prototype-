import streamlit as st

st.title("MovAbility AI Prototype")
st.write("Adaptive Seating Risk Prediction Prototype")

st.subheader("Enter patient information")

age = st.number_input("Age", min_value=0, max_value=100, value=5)

gender = st.selectbox("Gender", ["Male", "Female"])

diagnosis = st.selectbox(
    "Diagnosis",
    ["Cerebral Palsy", "Spina Bifida", "Muscular Dystrophy", "Developmental Delay", "Other"]
)

gmfcs = st.selectbox("GMFCS", ["I", "II", "III", "IV", "V"])

pelvic_tilt = st.selectbox(
    "Pelvic Tilt",
    ["Neutral", "Anterior", "Posterior", "Obliquity", "Rotation"]
)

trunk_control = st.selectbox(
    "Trunk Control",
    ["Good", "Moderate", "Poor"]
)

head_control = st.selectbox(
    "Head Control",
    ["Good", "Moderate", "Poor"]
)

if st.button("Predict Seating Risk"):

    risk_score = 0

    if gmfcs in ["IV", "V"]:
        risk_score += 2
    elif gmfcs == "III":
        risk_score += 1

    if pelvic_tilt != "Neutral":
        risk_score += 1

    if trunk_control == "Poor":
        risk_score += 2
    elif trunk_control == "Moderate":
        risk_score += 1

    if head_control == "Poor":
        risk_score += 1

    if age < 3:
        risk_score += 1

    if risk_score >= 5:
        risk = "High"
    elif risk_score >= 3:
        risk = "Moderate"
    else:
        risk = "Low"

    st.success(f"Predicted Seating Risk Level: {risk}")

    if risk == "High":
        st.error("Clinical Interpretation: High seating/postural risk detected.")
        st.write("Recommended Action: Full adaptive seating assessment is recommended.")

    elif risk == "Moderate":
        st.warning("Clinical Interpretation: Moderate seating/postural risk detected.")
        st.write("Recommended Action: Monitor posture and reassess regularly.")

    else:
        st.info("Clinical Interpretation: Low seating/postural risk detected.")
        st.write("Recommended Action: Continue current management and monitor growth/posture.")
