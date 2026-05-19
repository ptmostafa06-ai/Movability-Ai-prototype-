import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title("SeatMind AI Prototype")
st.write("Predictive Seating & Positioning Intelligence")
st.caption("Prototype screening tool for seating and postural risk. This does not replace clinical assessment.")

df = pd.read_csv("Prototype AI - Sheet1.csv")
df.columns = df.columns.str.strip()

target_col = "predicted_seating_postural_risk_level"

# Clean all text values
for col in df.columns:
    if col != "age":
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df[col] = df[col].replace(["nan", "none", ""], pd.NA)

df = df.dropna()

# Make age integer
df["age"] = df["age"].astype(int)

def options_for(col):
    values = sorted(df[col].dropna().unique().tolist())
    return [v for v in values if v not in ["nan", "none", ""]]

encoders = {}
encoded_df = df.copy()

for column in encoded_df.columns:
    le = LabelEncoder()
    encoded_df[column] = le.fit_transform(encoded_df[column].astype(str))
    encoders[column] = le

X = encoded_df.drop(target_col, axis=1)
y = encoded_df[target_col]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.subheader("Enter child seating information")

age = st.number_input("Age", min_value=0, max_value=100, value=5, step=1)

gender = st.selectbox("Gender", options_for("Gender"))

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

def add_reason(condition, reason, reasons):
    if condition:
        reasons.append(reason)

if st.button("Predict Seating Risk"):

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
                input_data[column].astype(str).str.strip().str.lower()
            )

    prediction_code = model.predict(input_data)[0]
    prediction_label = encoders[target_col].inverse_transform([prediction_code])[0]

    reasons = []

    add_reason("adaptive" in current_seating_setup, "The child is already using an adaptive seating system, so the result reflects risk within the current seating setup.", reasons)
    add_reason("needs" in sitting_support_level or "full" in sitting_support_level, "The child requires external sitting support, suggesting higher postural management needs.", reasons)
    add_reason("mixed" in pelvic_alignment_while_sitting or "asymmetry" in pelvic_alignment_while_sitting or "tilt" in pelvic_alignment_while_sitting, "Pelvic alignment is not fully neutral, which can affect trunk posture, pressure distribution, and sitting stability.", reasons)
    add_reason("leaning" in back_trunk_position_while_sitting or "collapse" in back_trunk_position_while_sitting or "curve" in back_trunk_position_while_sitting, "Back/trunk position shows leaning, collapse, or visible curve, which may indicate reduced postural control or seating mismatch.", reasons)
    add_reason("poor" in head_control_while_sitting or "needs" in head_control_while_sitting, "Head control concerns may affect vision, feeding, breathing, communication, and participation.", reasons)
    add_reason("stiff" in body_stiffness_movement_pattern or "dystonia" in body_stiffness_movement_pattern or "mixed" in body_stiffness_movement_pattern, "Tone or movement pattern may create changing postural needs during sitting.", reasons)
    add_reason("loses" in sits_stable_without_position_loss or "no" in sits_stable_without_position_loss, "The child loses sitting position, suggesting the current setup may not maintain alignment during daily use.", reasons)
    add_reason("needs" in ability_to_adjust_position_independently or "unable" in ability_to_adjust_position_independently, "The child may not be able to independently correct position, increasing the need for external support and monitoring.", reasons)
    add_reason("short" in sitting_endurance or "limited" in sitting_endurance or "poor" in sitting_endurance, "Reduced sitting endurance may indicate discomfort, fatigue, poor alignment, or inadequate support.", reasons)
    add_reason("pain" in pain_or_discomfort_during_sitting or "discomfort" in pain_or_discomfort_during_sitting, "Pain or discomfort during sitting is a warning sign that seating or positioning should be reviewed.", reasons)
    add_reason("redness" in skin_redness_pressure_history or "pressure" in skin_redness_pressure_history or "skin" in skin_redness_pressure_history, "Skin redness or pressure history may indicate pressure distribution risk.", reasons)

    if age < 3:
        reasons.append("Young age requires close monitoring because growth can quickly change seating and postural needs.")

    st.subheader("Prediction Result")

    if prediction_label.strip().lower() == "high":
        st.error("Predicted Seating & Postural Risk Level: HIGH")
        st.write("Recommended Action: A full adaptive seating and postural assessment is strongly recommended.")

    elif prediction_label.strip().lower() == "moderate":
        st.warning("Predicted Seating & Postural Risk Level: MODERATE")
        st.write("Recommended Action: Seating review and close monitoring are recommended.")

    else:
        st.success("Predicted Seating & Postural Risk Level: LOW")
        st.write("Recommended Action: Continue monitoring posture, comfort, skin, and sitting tolerance.")

    st.subheader("Why this prediction?")
    if reasons:
        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.write("- No major risk indicators were selected based on the current input.")

    st.caption("This result is generated by an early prototype model and should be interpreted by a qualified clinician.")
    st.caption("This result is generated by an early prototype model and should be interpreted by a qualified clinician.")
