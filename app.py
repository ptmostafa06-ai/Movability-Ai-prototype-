import streamlit as st

st.title("MovAbility AI Prototype")
st.write("Adaptive Seating Risk Prediction Prototype")

st.subheader("Enter patient information")

# AGE
age = st.number_input("Age", min_value=0, max_value=100, value=5)

# GENDER
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

# DIAGNOSIS
diagnosis = st.selectbox(
    "Diagnosis",
    [
        "Cerebral Palsy",
        "Spina Bifida",
        "Muscular Dystrophy",
        "Developmental Delay",
        "Other"
    ]
)

# GMFCS
gmfcs = st.selectbox(
    "GMFCS",
    ["I", "II", "III", "IV", "V"]
)

# PELVIC ALIGNMENT
pelvic_alignment = st.multiselect(
    "Pelvic Alignment",
    [
        "Neutral",
        "Anterior Tilt",
        "Posterior Tilt",
        "Obliquity",
        "Rotation"
    ],
    default=["Neutral"]
)

# TRUNK CONTROL
trunk_control = st.selectbox(
    "Trunk Control",
    [
        "Good",
        "Moderate",
        "Poor"
    ]
)

# HEAD CONTROL
head_control = st.selectbox(
    "Head Control",
    [
        "Good",
        "Moderate",
        "Poor"
    ]
)

# BUTTON
if st.button("Predict Seating Risk"):

    risk_score = 0
    reasons = []

    # GMFCS
    if gmfcs in ["IV", "V"]:
        risk_score += 2
        reasons.append(
            f"GMFCS level {gmfcs} indicates significant mobility and postural support needs."
        )

    elif gmfcs == "III":
        risk_score += 1
        reasons.append(
            "GMFCS level III indicates moderate mobility limitation and seating support needs."
        )

    # PELVIS
    if "Neutral" in pelvic_alignment:
        reasons.append(
            "Pelvic alignment is currently neutral without significant observable asymmetry."
        )

    if "Anterior Tilt" in pelvic_alignment and "Neutral" not in pelvic_alignment:
        risk_score += 1
        reasons.append(
            "Anterior pelvic tilt may affect spinal alignment, sitting balance, and trunk control."
        )

    if "Posterior Tilt" in pelvic_alignment and "Neutral" not in pelvic_alignment:
        risk_score += 1
        reasons.append(
            "Posterior pelvic tilt may increase sacral sitting and reduce stable upright posture."
        )

    if "Obliquity" in pelvic_alignment and "Neutral" not in pelvic_alignment:
        risk_score += 2
        reasons.append(
            "Pelvic obliquity can create asymmetric loading and increase risk of postural deformity or pressure concentration."
        )

    if "Rotation" in pelvic_alignment and "Neutral" not in pelvic_alignment:
        risk_score += 2
        reasons.append(
            "Pelvic rotation can affect trunk alignment, sitting symmetry, and functional upper limb use."
        )

    # TRUNK CONTROL
    if trunk_control == "Poor":
        risk_score += 2
        reasons.append(
            "Poor trunk control increases the need for external postural support."
        )

    elif trunk_control == "Moderate":
        risk_score += 1
        reasons.append(
            "Moderate trunk control suggests reduced sitting endurance and need for monitoring/support."
        )

    # HEAD CONTROL
    if head_control == "Poor":
        risk_score += 1
        reasons.append(
            "Poor head control may affect visual orientation, breathing, feeding, and functional participation."
        )

    # AGE
    if age < 3:
        risk_score += 1
        reasons.append(
            "Young age requires close monitoring because growth can quickly change seating and postural needs."
        )

    # FINAL RISK
    if risk_score >= 5:
        risk = "High"

    elif risk_score >= 3:
        risk = "Moderate"

    else:
        risk = "Low"

    # RESULT
    st.success(f"Predicted Seating Risk Level: {risk}")

    # CLINICAL INTERPRETATION
    st.subheader("Clinical Interpretation")

    if risk == "High":
        st.error("High seating/postural risk detected.")

    elif risk == "Moderate":
        st.warning("Moderate seating/postural risk detected.")

    else:
        st.info("Low seating/postural risk detected.")

    st.write("This prediction is based on the following clinical findings:")

    for reason in reasons:
        st.write(f"- {reason}")

    # RECOMMENDATION
    st.subheader("Recommended Action")

    if risk == "High":
        st.write(
            "A comprehensive adaptive seating and mobility assessment is strongly recommended, including posture, pressure distribution, trunk support, pelvic positioning, head control, and functional sitting tolerance."
        )

    elif risk == "Moderate":
        st.write(
            "Close monitoring and periodic reassessment are recommended to prevent progression of postural asymmetry and functional limitations."
        )

    else:
        st.write(
            "Continue current management while monitoring posture, growth, comfort, and functional sitting performance."
        )
