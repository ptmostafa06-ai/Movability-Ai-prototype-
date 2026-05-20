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
        df[col] = df[col].replace(["nan", "none", ""], pd.NA)

df["age"] = pd.to_numeric(df["age"], errors="coerce")

# Fix repeated spelling / formatting issues in CSV
df["skin_redness_pressure_history"] = df["skin_redness_pressure_history"].replace({
    "no_redness_or_skin_issues": "no_redness_or_skin_issues",
    "no_redness_or_skin_issue": "no_redness_or_skin_issues",
    "no_readness_or_skin_issues": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues.": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues,": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_________________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues__________________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues___________________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues____________________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_____________________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues______________________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues_______________________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues",
    "no_redness_or_skin_issues________________________________________________________________________________________________________________________________________________________________________________________________________________": "no_redness_or_skin_issues"
})

df = df.dropna()

df["age"] = df["age"].astype(int)

# =========================
# HELPER FUNCTIONS
# =========================

def options_for(col):
    values = sorted(df[col].dropna().unique().tolist())
    return values

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

gender = st.selectbox(
    "Gender",
    options_for("Gender"),
    format_func=pretty_label
)

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

    if mobility_level in [
        "pushchair_caregiver_dependent",
        "limited_functional_mobility"
    ]:
        score += 2
        reasons.append("Mobility level suggests significant dependence and increased postural support needs.")

    elif mobility_level in [
        "mobility_with_physical_support",
        "mobility_with_supervision"
    ]:
        score += 1
        reasons.append("Mobility level suggests moderate functional limitation and seating review needs.")

    if sitting_support_level == "fully_supported_sitting":
        score += 2
        reasons.append("The individual requires significant external sitting support.")

    elif sitting_support_level in [
        "trunk_support_needed",
        "hand_support_needed"
    ]:
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

    if back_trunk_position_while_sitting in [
        "severe_collapse_or_fixed_asymmetry",
        "clear_leaning_or_visible_curve"
    ]:
        score += 2
        reasons.append("Back/trunk posture suggests postural asymmetry or reduced trunk stability.")

    elif back_trunk_position_while_sitting == "mild_leaning_or_asymmetry":
        score += 1
        reasons.append("Mild trunk asymmetry may require seating review and monitoring.")

    if head_control_while_sitting == "poor":
        score += 2
        reasons.append("Poor head control may affect vision, feeding, breathing, communication, and participation.")

    elif head_control_while_sitting == "moderate":
        score += 1
        reasons.append("Moderate head control may require additional postural support.")

    if body_stiffness_movement_pattern in [
        "dystonic_movements,mixed_tone",
        "fluctuating_tone"
    ]:
        score += 2
        reasons.append("Fluctuating tone or dystonic movement may create changing postural needs during sitting.")

    elif body_stiffness_movement_pattern in [
        "high_tone",
        "low_tone"
    ]:
        score += 1
        reasons.append("Tone presentation may affect postural control and sitting stability.")

    if sits_stable_without_position_loss == "constantly_loses_position":
        score += 3
        reasons.append("The individual constantly loses sitting position, suggesting high postural support needs.")

    elif sits_stable_without_position_loss == "loses_position_many_times_during_sitting":
        score += 2
        reasons.append("The individual loses position many times during sitting.")

    elif sits_stable_without_position_loss == "loses_position_from_time_to_time":
        score += 1
        reasons.append("The individual loses sitting position from time to time.")

    if ability_to_adjust_position_independently == "unable_to_adjust_position_independently":
        score += 2
        reasons.append("The individual is unable to independently correct sitting position.")

    elif ability_to_adjust_position_independently in [
        "needs_physical_assistance",
        "needs_verbal_reminders_cueing"
    ]:
        score += 1
        reasons.append("The individual needs assistance or cueing to adjust sitting position.")

    if sitting_endurance == "cannot_tolerate_sitting_for_functional_activities":
        score += 2
        reasons.append("Very limited sitting tolerance may indicate discomfort, fatigue, poor alignment, or inadequate support.")

    elif sitting_endurance == "gets_tired_shortly_after_sitting":
        score += 1
        reasons.append("Reduced sitting endurance may indicate fatigue, discomfort, or inadequate support.")

    if pain_or_discomfort_during_sitting in [
        "moderate_pain_discomfort",
        "severe_pain_discomfort"
    ]:
        score += 2
        reasons.append("Pain or discomfort during sitting suggests the seating setup should be reviewed.")

    elif pain_or_discomfort_during_sitting in [
        "mild_discomfort",
        "unable_to_determine"
    ]:
        score += 1
        reasons.append("Discomfort or unclear pain response may require monitoring and review.")

    if skin_redness_pressure_history == "previous_skin_breakdown_pressure_injury":
        score += 3
        reasons.append("Previous skin breakdown or pressure injury indicates increased pressure risk.")

    elif skin_redness_pressure_history in [
        "redness_appears_from_time_to_time",
        "occasional_redness_after_sitting"
    ]:
        score += 2
        reasons.append("Skin redness after sitting may indicate pressure distribution risk.")

    if score >= 9:
        return "high", score, reasons

    elif score >= 5:
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
# HYBRID DECISION
# =========================

risk_order = {
    "low": 0,
    "moderate": 1,
    "high": 2
}

def hybrid_decision(rule_result, ml_result):
    rule_score = risk_order.get(rule_result, 0)
    ml_score = risk_order.get(ml_result, 0)

    # Rule-based clinical reasoning remains primary.
    # ML can raise low to moderate, but cannot automatically push every case to high.
    if rule_result == "high":
        return "high"

    if rule_result == "moderate" and ml_result == "high":
        return "moderate"

    if rule_result == "low" and ml_result == "high":
        return "moderate"

    if ml_score > rule_score:
        return ml_result

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
    st.write(f"Machine learning support result: {ml_result.upper()}")

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
