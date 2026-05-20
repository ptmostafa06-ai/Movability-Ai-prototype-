import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(page_title="SeatMind AI", page_icon="🧠", layout="centered")

st.title("SeatMind AI")
st.write("Hybrid Predictive Seating & Positioning Intelligence")
st.write(
    "This prototype combines rule-based clinical reasoning with a supportive machine learning layer."
)

# =========================
# LOAD DATA
# =========================

CSV_FILE = "Prototype AI - Sheet1.csv"

try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    st.error("CSV file not found. Please make sure the file name is exactly: Prototype AI - Sheet1.csv")
    st.stop()

df.columns = df.columns.str.strip()

target_col = "predicted_seating_postural_risk_level"

required_columns = [
    "age",
    "Gender",
    "current_seating_setup",
    "mobility_level",
    "sitting_support_level",
    "pelvic_alignment_while_sitting",
    "back_trunk_position_while_sitting",
    "head_control_while_sitting",
    "body_stiffness_movement_pattern",
    "sits_stable_without_position_loss",
    "ability_to_adjust_position_independently",
    "sitting_endurance",
    "pain_or_discomfort_during_sitting",
    "skin_redness_pressure_history",
    target_col
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing columns in CSV: {missing_columns}")
    st.stop()

# =========================
# CLEAN DATA
# =========================

df = df[required_columns].copy()

for col in df.columns:
    if col != "age":
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        # Important: do NOT remove "none" because it is a real clinical option
        df[col] = df[col].replace(["nan", ""], pd.NA)

df["age"] = pd.to_numeric(df["age"], errors="coerce")

# Fix CSV spelling variations
df["skin_redness_pressure_history"] = df["skin_redness_pressure_history"].replace({
    "no_readness_or_skin_issues": "no_redness_or_skin_issues"
})

df = df.dropna()
df["age"] = df["age"].astype(int)

# =========================
# FIXED OPTIONS
# =========================

FIXED_OPTIONS = {
    "Gender": [
        "male",
        "female",
    ],

    "current_seating_setup": [
        "no_adaptive_seating_system",
        "basic_chair_stroller_only",
        "adaptive_seating_system",
    ],

    "mobility_level": [
        "independent_mobility",
        "independent_wheelchair_user",
        "mobility_with_supervision",
        "mobility_with_physical_support",
        "limited_functional_mobility",
        "pushchair_caregiver_dependent",
    ],

    "sitting_support_level": [
        "independent_sitting",
        "hand_support_needed",
        "trunk_support_needed",
        "fully_supported_sitting",
    ],

    "pelvic_alignment_while_sitting": [
        "pelvis_mostly_neutral_centered",
        "anterior_pelvic_tilt",
        "posterior_pelvic_tilt",
        "pelvic_obliquity_asymmetry",
        "pelvic_rotation",
        "mixed_pelvic_asymmetry",
    ],

    "back_trunk_position_while_sitting": [
        "trunk_mostly_centered_midline",
        "mild_leaning_or_asymmetry",
        "clear_leaning_or_visible_curve",
        "severe_collapse_or_fixed_asymmetry",
    ],

    "head_control_while_sitting": [
        "good",
        "moderate",
        "poor",
    ],

    "body_stiffness_movement_pattern": [
        "normal_tone",
        "low_tone",
        "high_tone",
        "fluctuating_tone",
        "dystonic_movements,mixed_tone",
    ],

    "sits_stable_without_position_loss": [
        "sits_stable_without_position_loss",
        "loses_position_from_time_to_time",
        "loses_position_many_times_during_sitting",
        "constantly_loses_position",
    ],

    "ability_to_adjust_position_independently": [
        "independent",
        "needs_verbal_reminders_cueing",
        "needs_physical_assistance",
        "unable_to_adjust_position_independently",
    ],

    "sitting_endurance": [
        "maintains_sitting_without_fatigue",
        "gets_tired_after_prolonged_sitting",
        "gets_tired_shortly_after_sitting",
        "cannot_tolerate_sitting_for_functional_activities",
    ],

    "pain_or_discomfort_during_sitting": [
        "none",
        "mild_discomfort",
        "moderate_pain_discomfort",
        "severe_pain_discomfort",
        "unable_to_determine",
    ],

    "skin_redness_pressure_history": [
        "no_redness_or_skin_issues",
        "occasional_redness_after_sitting",
        "redness_appears_from_time_to_time",
        "previous_skin_breakdown_pressure_injury",
    ],
}

def options_for(col):
    if col in FIXED_OPTIONS:
        return FIXED_OPTIONS[col]

    return sorted(df[col].dropna().unique().tolist())

def pretty_label(value):
    return str(value).replace("_", " ").title()

# =========================
# MACHINE LEARNING MODEL
# =========================

feature_cols = [col for col in required_columns if col != target_col]

X = df[feature_cols]
y = df[target_col]

categorical_cols = [col for col in feature_cols if col != "age"]
numeric_cols = ["age"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)

model.fit(X, y)

# =========================
# USER INPUTS
# =========================

st.subheader("Enter Seating Information")

age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=5,
    step=1,
    format="%d"
)

gender = st.selectbox("Gender", options_for("Gender"), format_func=pretty_label)

current_seating_setup = st.selectbox(
    "Current Seating Setup",
    options_for("current_seating_setup"),
    format_func=pretty_label
)

mobility_level = st.selectbox(
    "Mobility Level",
    options_for("mobility_level"),
    format_func=pretty_label
)

sitting_support_level = st.selectbox(
    "Sitting Support Level",
    options_for("sitting_support_level"),
    format_func=pretty_label
)

pelvic_alignment_while_sitting = st.selectbox(
    "Pelvic Alignment While Sitting",
    options_for("pelvic_alignment_while_sitting"),
    format_func=pretty_label
)

back_trunk_position_while_sitting = st.selectbox(
    "Back / Trunk Position While Sitting",
    options_for("back_trunk_position_while_sitting"),
    format_func=pretty_label
)

head_control_while_sitting = st.selectbox(
    "Head Control While Sitting",
    options_for("head_control_while_sitting"),
    format_func=pretty_label
)

body_stiffness_movement_pattern = st.selectbox(
    "Body Stiffness / Movement Pattern",
    options_for("body_stiffness_movement_pattern"),
    format_func=pretty_label
)

sits_stable_without_position_loss = st.selectbox(
    "Sits Stable Without Position Loss",
    options_for("sits_stable_without_position_loss"),
    format_func=pretty_label
)

ability_to_adjust_position_independently = st.selectbox(
    "Ability to Adjust Position Independently",
    options_for("ability_to_adjust_position_independently"),
    format_func=pretty_label
)

sitting_endurance = st.selectbox(
    "Sitting Endurance",
    options_for("sitting_endurance"),
    format_func=pretty_label
)

pain_or_discomfort_during_sitting = st.selectbox(
    "Pain or Discomfort During Sitting",
    options_for("pain_or_discomfort_during_sitting"),
    format_func=pretty_label
)

skin_redness_pressure_history = st.selectbox(
    "Skin Redness / Pressure History",
    options_for("skin_redness_pressure_history"),
    format_func=pretty_label
)

# =========================
# RULE-BASED ENGINE
# =========================

def rule_based_prediction():
    score = 0
    reasons = []

    if age < 3:
        score += 1
        reasons.append("Young age may require closer monitoring because growth can quickly change seating and postural needs.")

    if mobility_level in ["limited_functional_mobility", "pushchair_caregiver_dependent"]:
        score += 2
        reasons.append("Mobility level suggests significant functional limitation and increased postural support needs.")

    elif mobility_level in ["mobility_with_physical_support", "mobility_with_supervision"]:
        score += 1
        reasons.append("Mobility level suggests mild to moderate functional limitation and seating review needs.")

    if sitting_support_level == "fully_supported_sitting":
        score += 2
        reasons.append("The individual requires significant external sitting support.")

    elif sitting_support_level in ["trunk_support_needed", "hand_support_needed"]:
        score += 1
        reasons.append("The individual requires external sitting support.")

    if pelvic_alignment_while_sitting in [
        "mixed_pelvic_asymmetry",
        "pelvic_obliquity_asymmetry",
        "pelvic_rotation",
        "posterior_pelvic_tilt"
    ]:
        score += 2
        reasons.append("Pelvic alignment is not neutral, which may affect posture, pressure distribution, and sitting stability.")

    elif pelvic_alignment_while_sitting == "anterior_pelvic_tilt":
        score += 1
        reasons.append("Pelvic alignment may require monitoring during sitting.")

    if back_trunk_position_while_sitting == "severe_collapse_or_fixed_asymmetry":
        score += 3
        reasons.append("Severe trunk collapse or fixed asymmetry suggests high postural support needs.")

    elif back_trunk_position_while_sitting == "clear_leaning_or_visible_curve":
        score += 2
        reasons.append("Back/trunk posture suggests visible asymmetry or reduced trunk stability.")

    elif back_trunk_position_while_sitting == "mild_leaning_or_asymmetry":
        score += 1
        reasons.append("Mild trunk asymmetry may require seating review and monitoring.")

    if head_control_while_sitting == "poor":
        score += 3
        reasons.append("Poor head control may affect vision, feeding, breathing, communication, and participation.")

    elif head_control_while_sitting == "moderate":
        score += 1
        reasons.append("Moderate head control may require additional postural support.")

    if body_stiffness_movement_pattern in ["dystonic_movements,mixed_tone", "fluctuating_tone"]:
        score += 2
        reasons.append("Fluctuating tone or dystonic movement may create changing postural needs during sitting.")

    elif body_stiffness_movement_pattern in ["high_tone", "low_tone"]:
        score += 1
        reasons.append("Tone presentation may affect postural control and sitting stability.")

    if sits_stable_without_position_loss == "constantly_loses_position":
        score += 4
        reasons.append("The individual constantly loses sitting position, suggesting high postural support needs.")

    elif sits_stable_without_position_loss == "loses_position_many_times_during_sitting":
        score += 2
        reasons.append("The individual loses position many times during sitting.")

    elif sits_stable_without_position_loss == "loses_position_from_time_to_time":
        score += 1
        reasons.append("The individual loses sitting position from time to time.")

    if ability_to_adjust_position_independently == "unable_to_adjust_position_independently":
        score += 3
        reasons.append("The individual is unable to independently correct sitting position.")

    elif ability_to_adjust_position_independently in ["needs_physical_assistance", "needs_verbal_reminders_cueing"]:
        score += 1
        reasons.append("The individual needs assistance or cueing to adjust sitting position.")

    if sitting_endurance == "cannot_tolerate_sitting_for_functional_activities":
        score += 3
        reasons.append("Very limited sitting tolerance may indicate discomfort, fatigue, poor alignment, or inadequate support.")

    elif sitting_endurance == "gets_tired_shortly_after_sitting":
        score += 2
        reasons.append("Reduced sitting endurance may indicate fatigue, discomfort, or inadequate support.")

    elif sitting_endurance == "gets_tired_after_prolonged_sitting":
        score += 1
        reasons.append("Sitting endurance may require monitoring during prolonged functional activities.")

    if pain_or_discomfort_during_sitting == "severe_pain_discomfort":
        score += 4
        reasons.append("Severe pain or discomfort during sitting is a major seating review indicator.")

    elif pain_or_discomfort_during_sitting == "moderate_pain_discomfort":
        score += 2
        reasons.append("Moderate pain or discomfort during sitting suggests the seating setup should be reviewed.")

    elif pain_or_discomfort_during_sitting in ["mild_discomfort", "unable_to_determine"]:
        score += 1
        reasons.append("Discomfort or unclear pain response may require monitoring and review.")

    if skin_redness_pressure_history == "previous_skin_breakdown_pressure_injury":
        score += 4
        reasons.append("Previous skin breakdown or pressure injury indicates increased pressure risk.")

    elif skin_redness_pressure_history in ["redness_appears_from_time_to_time", "occasional_redness_after_sitting"]:
        score += 2
        reasons.append("Skin redness after sitting may indicate pressure distribution risk.")

    critical_red_flags = [
        sits_stable_without_position_loss == "constantly_loses_position",
        skin_redness_pressure_history == "previous_skin_breakdown_pressure_injury",
        pain_or_discomfort_during_sitting == "severe_pain_discomfort",
        back_trunk_position_while_sitting == "severe_collapse_or_fixed_asymmetry",
        head_control_while_sitting == "poor",
        ability_to_adjust_position_independently == "unable_to_adjust_position_independently",
    ]

    if score >= 12 or (score >= 9 and any(critical_red_flags)):
        return "high", score, reasons

    elif score >= 6:
        return "moderate", score, reasons

    else:
        return "low", score, reasons

# =========================
# ML PREDICTION
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

    prediction = model.predict(input_data)[0]
    return str(prediction).strip().lower()

# =========================
# FINAL DECISION
# =========================

def hybrid_decision(rule_result, ml_result):
    # Final decision is based on clinical rule-based reasoning.
    # ML is supportive only at this prototype stage.
    return rule_result

# =========================
# FINAL OUTPUT
# =========================

if st.button("Predict Seating Risk"):

    rule_result, clinical_score, reasons = rule_based_prediction()
    ml_result = ml_prediction()
    final_result = hybrid_decision(rule_result, ml_result)

    st.subheader("Prediction Result")

    if final_result == "high":
        st.error("Seating & Postural Risk Level: HIGH")
        st.write("Recommended Action: A comprehensive adaptive seating and postural assessment is strongly recommended.")

    elif final_result == "moderate":
        st.warning("Seating & Postural Risk Level: MODERATE")
        st.write("Recommended Action: Seating review and close monitoring are recommended.")

    else:
        st.success("Seating & Postural Risk Level: LOW")
        st.write("Recommended Action: Continue monitoring posture, comfort, skin condition, and sitting tolerance.")

    st.subheader("Why this result?")

    st.write(f"Clinical rule-based score: {clinical_score}")
    st.write(f"Rule-based result: {rule_result.upper()}")
    st.write(f"Machine learning support result: {ml_result.upper()} — supportive only, not final decision")

    if reasons:
        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.write("- No major seating or postural risk indicators were selected based on the current input.")

    st.markdown("---")

    st.warning(
        "This is an early screening and decision-support tool. "
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
