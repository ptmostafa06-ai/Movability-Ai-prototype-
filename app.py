import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =========================
# PAGE TITLE
# =========================

st.title("SeatMind AI Prototype")
st.write("Hybrid Predictive Seating & Positioning Intelligence")

st.caption(
    "This prototype combines rule-based clinical reasoning with a supportive machine learning layer."
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("Prototype AI - Sheet1.csv")

# Clean column names
df.columns = df.columns.str.strip()

target_col = "predicted_seating_postural_risk_level"

# =========================
# CLEAN DATA
# =========================

for col in df.columns:
    if col != "age":
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df[col] = df[col].replace(
            ["nan", "none", ""],
            pd.NA
        )

df = df.dropna()

# Convert age to integer
df["age"] = df["age"].astype(int)

# =========================
# REMOVE NONE OPTIONS
# =========================

def options_for(col):
    values = sorted(
        df[col]
        .dropna()
        .unique()
        .tolist()
    )

    return [
        v for v in values
        if v not in ["none", "nan", ""]
    ]

# =========================
# ENCODE DATA FOR ML
# =========================

encoders = {}

encoded_df = df.copy()

for column in encoded_df.columns:
    le = LabelEncoder()

    encoded_df[column] = le.fit_transform(
        encoded_df[column].astype(str)
    )

    encoders[column] = le

# =========================
# ML TRAINING
# =========================

X = encoded_df.drop(target_col, axis=1)
y = encoded_df[target_col]

model = RandomForestClassifier(
    random_state=42
)

model.fit(X, y)

# =========================
# USER INPUTS
# =========================

st.subheader("Enter Child Seating Information")

age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=5,
    step=1
)

gender = st.selectbox(
    "Gender",
    options_for("Gender")
)

current_seating_setup = st.selectbox(
    "Current Seating Setup",
    options_for("current_seating_setup")
)

mobility_level = st.selectbox(
    "Mobility Level",
    options_for("mobility_level")
)

sitting_support_level = st.selectbox(
    "Sitting Support Level",
    options_for("sitting_support_level")
)

pelvic_alignment_while_sitting = st.selectbox(
    "Pelvic Alignment While Sitting",
    options_for("pelvic_alignment_while_sitting")
)

back_trunk_position_while_sitting = st.selectbox(
    "Back / Trunk Position While Sitting",
    options_for("back_trunk_position_while_sitting")
)

head_control_while_sitting = st.selectbox(
    "Head Control While Sitting",
    options_for("head_control_while_sitting")
)

body_stiffness_movement_pattern = st.selectbox(
    "Body Stiffness / Movement Pattern",
    options_for("body_stiffness_movement_pattern")
)

sits_stable_without_position_loss = st.selectbox(
    "Sits Stable Without Position Loss",
    options_for("sits_stable_without_position_loss")
)

ability_to_adjust_position_independently = st.selectbox(
    "Ability to Adjust Position Independently",
    options_for("ability_to_adjust_position_independently")
)

sitting_endurance = st.selectbox(
    "Sitting Endurance",
    options_for("sitting_endurance")
)

pain_or_discomfort_during_sitting = st.selectbox(
    "Pain or Discomfort During Sitting",
    options_for("pain_or_discomfort_during_sitting")
)

skin_redness_pressure_history = st.selectbox(
    "Skin Redness / Pressure History",
    options_for("skin_redness_pressure_history")
)

# =========================
# RULE-BASED ENGINE
# =========================

def rule_based_prediction():

    score = 0
    reasons = []

    # AGE
    if age < 3:
        score += 1

        reasons.append(
            "Young age requires closer monitoring because growth can quickly change seating and postural needs."
        )

    # MOBILITY
    if (
        "dependent" in mobility_level
        or "full support" in mobility_level
    ):
        score += 2

        reasons.append(
            "Mobility level suggests significant dependence and increased postural support needs."
        )

    elif (
        "moderate" in mobility_level
        or "assistance" in mobility_level
    ):
        score += 1

        reasons.append(
            "Mobility level suggests moderate functional limitation and seating review needs."
        )

    # SITTING SUPPORT
    if (
        "full" in sitting_support_level
        or "dependent" in sitting_support_level
    ):
        score += 2

        reasons.append(
            "The child requires significant external sitting support."
        )

    elif (
        "moderate" in sitting_support_level
        or "partial" in sitting_support_level
    ):
        score += 1

        reasons.append(
            "The child requires partial external sitting support."
        )

    # PELVIS
    if (
        "obliquity" in pelvic_alignment_while_sitting
        or "rotation" in pelvic_alignment_while_sitting
        or "posterior" in pelvic_alignment_while_sitting
        or "anterior" in pelvic_alignment_while_sitting
        or "asymmetry" in pelvic_alignment_while_sitting
    ):
        score += 2

        reasons.append(
            "Pelvic alignment is not neutral, which may affect posture, pressure distribution, and sitting stability."
        )

    # TRUNK
    if (
        "lean" in back_trunk_position_while_sitting
        or "collapse" in back_trunk_position_while_sitting
        or "curve" in back_trunk_position_while_sitting
        or "scoliosis" in back_trunk_position_while_sitting
        or "kyphosis" in back_trunk_position_while_sitting
    ):
        score += 2

        reasons.append(
            "Back/trunk posture suggests postural asymmetry or reduced trunk stability."
        )

    # HEAD CONTROL
    if (
        "poor" in head_control_while_sitting
        or "limited" in head_control_while_sitting
    ):
        score += 1

        reasons.append(
            "Reduced head control may affect vision, feeding, breathing, communication, and participation."
        )

    # TONE
    if (
        "spastic" in body_stiffness_movement_pattern
        or "dystonia" in body_stiffness_movement_pattern
        or "mixed" in body_stiffness_movement_pattern
        or "stiff" in body_stiffness_movement_pattern
    ):
        score += 1

        reasons.append(
            "Tone or movement pattern may create changing postural needs during sitting."
        )

    # POSITION LOSS
    if (
        "loses" in sits_stable_without_position_loss
        or "no" in sits_stable_without_position_loss
    ):
        score += 2

        reasons.append(
            "The child loses sitting position, suggesting the current setup may not maintain alignment."
        )

    # POSITION ADJUSTMENT
    if (
        "unable" in ability_to_adjust_position_independently
        or "needs assistance" in ability_to_adjust_position_independently
        or "no" in ability_to_adjust_position_independently
    ):
        score += 1

        reasons.append(
            "The child may not be able to independently correct sitting position."
        )

    # ENDURANCE
    if (
        "short" in sitting_endurance
        or "limited" in sitting_endurance
        or "poor" in sitting_endurance
    ):
        score += 1

        reasons.append(
            "Reduced sitting endurance may indicate discomfort, fatigue, poor alignment, or inadequate support."
        )

    # PAIN
    if (
        "pain" in pain_or_discomfort_during_sitting
        or "discomfort" in pain_or_discomfort_during_sitting
    ):
        score += 2

        reasons.append(
            "Pain or discomfort during sitting suggests the seating setup should be reviewed."
        )

    # SKIN
    if (
        "redness" in skin_redness_pressure_history
        or "pressure" in skin_redness_pressure_history
        or "skin" in skin_redness_pressure_history
    ):
        score += 2

        reasons.append(
            "Skin redness or pressure history may indicate pressure distribution risk."
        )

    # FINAL RULE RESULT
    if score >= 6:
        rule_result = "high"

    elif score >= 3:
        rule_result = "moderate"

    else:
        rule_result = "low"

    return rule_result, score, reasons

# =========================
# MACHINE LEARNING PREDICTION
# =========================

def ml_prediction():

    input_data = pd.DataFrame([{
        "age": age,
        "Gender": gender,
        "current_seating_setup": current_seating_setup,
        "mobility_level": mobility_level,
        "sitting_support_level": sitting_support_level,
        "pelvic_alignment_while_sitting": pelvic_alignment_while_sitting,
        "back_trunk_position_while_sitting": back_trunk_position_while_sitting,
        "head_control_while_sitting": head_control_while_sitting,
        "body_stiffness_movement_pattern": body_stiffness_movement_pattern,
        "sits_stable_without_position_loss": sits_stable_without_position_loss,
        "ability_to_adjust_position_independently": ability_to_adjust_position_independently,
        "sitting_endurance": sitting_endurance,
        "pain_or_discomfort_during_sitting": pain_or_discomfort_during_sitting,
        "skin_redness_pressure_history": skin_redness_pressure_history
    }])

    for column in input_data.columns:

        if column == "age":

            input_data[column] = input_data[column].astype(int)

        else:

            input_data[column] = encoders[column].transform(
                input_data[column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

    pred_code = model.predict(input_data)[0]

    pred_label = encoders[target_col].inverse_transform(
        [pred_code]
    )[0]

    probabilities = model.predict_proba(input_data)[0]

    confidence = max(probabilities) * 100

    return pred_label.strip().lower(), confidence

# =========================
# HYBRID DECISION
# =========================

def hybrid_decision(rule_result, ml_result):

    if rule_result == "high":
        return "high"

    if (
        rule_result == "moderate"
        and ml_result == "high"
    ):
        return "high"

    if (
        rule_result == "low"
        and ml_result == "high"
    ):
        return "moderate"

    return rule_result

# =========================
# FINAL OUTPUT
# =========================

if st.button("Predict Seating Risk"):

    # RULE RESULT
    rule_result, rule_score, reasons = rule_based_prediction()

    # ML RESULT
    ml_result, ml_confidence = ml_prediction()

    # HYBRID RESULT
    final_result = hybrid_decision(
        rule_result,
        ml_result
    )

    # =========================
    # RESULT DISPLAY
    # =========================

    st.subheader("Hybrid Prediction Result")

    if final_result == "high":

        st.error(
            "Final Seating & Postural Risk Level: HIGH"
        )

        st.write(
            "Recommended Action: A comprehensive adaptive seating and postural assessment is strongly recommended."
        )

    elif final_result == "moderate":

        st.warning(
            "Final Seating & Postural Risk Level: MODERATE"
        )

        st.write(
            "Recommended Action: Seating review and close monitoring are recommended."
        )

    else:

        st.success(
            "Final Seating & Postural Risk Level: LOW"
        )

        st.write(
            "Recommended Action: Continue monitoring posture, comfort, skin condition, and sitting tolerance."
        )

    # =========================
    # DECISION BREAKDOWN
    # =========================

    st.subheader("How The Decision Was Made")

    st.write(
        f"Rule-Based Clinical Result: {rule_result.upper()}"
    )

    st.write(
        f"Rule-Based Score: {rule_score}"
    )

    st.write(
        f"Machine Learning Result: {ml_result.upper()}"
    )

    st.write(
        f"Machine Learning Confidence: {ml_confidence:.1f}%"
    )

    # =========================
    # CLINICAL EXPLANATION
    # =========================

    st.subheader("Clinical Explanation")

    if reasons:

        for reason in reasons:

            st.write(f"- {reason}")

    else:

        st.write(
            "- No major clinical risk indicators were selected."
        )

    # =========================
    # HYBRID AI EXPLANATION
    # =========================

    st.subheader("Hybrid AI Logic")

    st.write(
        "The final result prioritizes the rule-based clinical safety layer, while the machine learning model acts as a supportive predictive layer. "
        "Because the dataset is still limited, the machine learning result is currently used as supportive guidance rather than the only decision-maker."
    )

    # =========================
    # DISCLAIMER + INSTAGRAM
    # =========================

    st.markdown("---")

    st.warning(
        "This prototype is an early clinical screening and decision-support tool. "
        "It does not replace a full professional seating and mobility assessment."
    )

    st.markdown(
        """
        ### Need Clinical Assessment or Professional Guidance?

        For adaptive seating, posture, mobility, and positioning consultation:

        **Contact Mostafa Ahmed (Mostafa Physio)**

        📲 [Open Instagram Page](https://www.instagram.com/mostafaphysio?igsh=M2d3ZjMzOTFxb3M5&utm_source=qr)
        """
    )
