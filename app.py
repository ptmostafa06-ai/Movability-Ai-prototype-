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

pelvic_alignment = st.multiselect(
    "Pelvic Alignment",
    [
        "Anterior Tilt",
        "Posterior Tilt",
        "Obliquity",
        "Rotation"
    ]
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
    reasons = []

    if gmfcs in ["IV", "V"]:
        risk_score += 2
        reasons.append(f"GMFCS level {gmfcs} indicates significant mobility and postural support needs.")
    elif gmfcs == "III":
        risk_score += 1
        reasons.append("GMFCS level III indicates moderate mobility limitation and possible seating support needs.")

    if "Anterior Tilt" in pelvic_alignment:
        risk_score += 1
        reasons.append("Anterior pelvic tilt may affect spinal alignment, sitting balance, and trunk control.")

    if "Posterior Tilt" in pelvic_alignment:
        risk_score += 1
        reasons.append("Posterior pelvic tilt may increase sacral sitting and reduce stable upright posture.")

    if "Obliquity" in pelvic_alignment:
        risk_score += 2
        reasons.append("Pelvic obliquity can create asymmetric loading and increase risk of postural deformity or pressure concentration.")

    if "Rotation" in pelvic_alignment:
        risk_score += 2
        reasons.append("Pelvic rotation can affect trunk alignment, sitting symmetry, and functional upper limb use.")

    if trunk_control == "Poor":
        risk_score += 2
        reasons.append("Poor trunk control increases the need for external postural support.")
    elif trunk_control == "Moderate":
        risk_score += 1
        reasons.append("Moderate trunk control suggests reduced sitting endurance and need for monitoring/support.")

    if head_control == "Poor":
        risk_score += 1
        reasons.append("Poor head control may affect visual orientation, breathing, feeding, and functional participation.")

    if age < 3:
        risk_score += 1
        reasons.append("Young age requires close monitoring because growth can quickly change seating and postural needs.")

    if risk_score >= 5:
        risk = "High"
    elif risk_score >= 3:
        risk = "Moderate"
    else:
        risk = "Low"

    st.success(f"Predicted Seating Risk Level: {risk}")

    st.subheader("Clinical Interpretation")

    if risk == "High":
        st.error("High seating/postural risk detected.")
    elif risk == "Moderate":
        st.warning("Moderate seating/postural risk detected.")
    else:
        st.info("Low seating/postural risk detected.")

    st.write("This result is based on the following clinical factors:")

    if reasons:
        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.write("- No major postural risk factors were selected.")

    st.subheader("Recommended Action")

    if risk == "High":
        st.write("A full adaptive seating assessment is recommended, including pelvic positioning, trunk support, head control, pressure distribution, and functional sitting tolerance.")
    elif risk == "Moderate":
        st.write("Monitor posture closely, review seating setup regularly, and consider reassessment if leaning, sliding, fatigue, asymmetry, or reduced function is observed.")
    else:
        st.write("Continue current management and monitor growth, posture, comfort, and daily sitting function.")
