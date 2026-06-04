import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from datetime import datetime
import html

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(page_title="SeatMind AI", page_icon="🧠", layout="centered")

st.title("SeatMind AI")
st.write("Hybrid Predictive Seating & Positioning Intelligence")
st.write("Rule-based clinical safety engine with supportive machine learning output.")

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

df = df.rename(columns={
    "mobility_status": "mobility_level",
    "sitting_stability": "sits_stable_without_position_loss",
})

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
    target_col,
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
            .str.replace("-", "_")
        )
        df[col] = df[col].replace(["nan", "", "none_"], pd.NA)

df["age"] = pd.to_numeric(df["age"], errors="coerce")

VALUE_REPLACEMENTS = {
    "Gender": {
        "m": "male",
        "f": "female",
    },
    "skin_redness_pressure_history": {
        "no_readness_or_skin_issues": "no_redness_or_skin_issues",
        "occasional_redness_after_sitting": "redness_appears_from_time_to_time",
        "frequent_redness": "redness_appears_often",
    },
    "body_stiffness_movement_pattern": {
        "dystonic_movements,mixed_tone": "mixed_tone",
        "dystonic_movements_mixed_tone": "mixed_tone",
    },
    "pain_or_discomfort_during_sitting": {
        "no_pain": "none",
        "no_discomfort": "none",
    },
}

for col, replacements in VALUE_REPLACEMENTS.items():
    if col in df.columns:
        df[col] = df[col].replace(replacements)

df = df.dropna()
df["age"] = df["age"].astype(int)

# =========================
# FRONTEND LABELS -> BACKEND VALUES
# =========================

OPTIONS = {
    "Gender": {
        "Male": "male",
        "Female": "female",
    },

    "Current seating setup": {
        "No adaptive seating system": "no_adaptive_seating_system",
        "Basic chair, stroller, or wheelchair only": "basic_chair_stroller_only",
        "Adaptive seating system": "adaptive_seating_system",
    },

    "Mobility level": {
        "Moves independently": "independent_mobility",
        "Moves independently but needs supervision for safety": "mobility_with_supervision",
        "Moves with support from a person or device": "mobility_with_physical_support",
        "Can move only short distances or gets tired quickly": "limited_functional_mobility",
        "Usually pushed or moved by caregiver": "pushchair_caregiver_dependent",
        "Uses wheelchair independently": "independent_wheelchair_user",
    },

    "Sitting support level": {
        "Sits independently without support": "independent_sitting",
        "Uses hands for balance while sitting": "hand_support_needed",
        "Needs back or side support": "trunk_support_needed",
        "Needs full body support": "fully_supported_sitting",
    },

    "Pelvic alignment while sitting": {
        "Pelvis tilts forward": "anterior_pelvic_tilt",
        "Pelvis tilts backward / slouched sitting": "posterior_pelvic_tilt",
        "Leans more to one side": "pelvic_obliquity_asymmetry",
        "Pelvis twists to one side": "pelvic_rotation",
    },

    "Back / trunk position while sitting": {
        "Trunk is mostly upright and centered": "trunk_mostly_centered_midline",
        "Mild leaning or asymmetry": "mild_leaning_or_asymmetry",
        "Clear leaning or visible curve": "clear_leaning_or_visible_curve",
        "Severe collapse or fixed asymmetry": "severe_collapse_or_fixed_asymmetry",
    },

    "Head control while sitting": {
        "Keeps head upright most of the time": "good",
        "Sometimes loses head position": "moderate",
        "Frequently needs head support": "poor",
    },

    "Body stiffness / movement pattern": {
        "Normal movement / tone": "normal_tone",
        "Body feels floppy or weak": "low_tone",
        "Body feels stiff or tight": "high_tone",
        "Tone changes between floppy and stiff": "fluctuating_tone",
        "Involuntary twisting or uncontrolled movements": "dystonic_movements",
        "Combination of different tone patterns": "mixed_tone",
    },

    "Sitting stability": {
        "Stays in position while sitting": "sits_stable_without_position_loss",
        "Occasionally slides or leans": "loses_position_from_time_to_time",
        "Frequently loses position": "loses_position_many_times_during_sitting",
        "Cannot stay in position without help": "constantly_loses_position",
    },

    "Ability to adjust position independently": {
        "Adjusts position independently": "independent",
        "Needs reminders to adjust position": "needs_verbal_reminders_cueing",
        "Needs physical help to adjust position": "needs_physical_assistance",
        "Cannot adjust position independently": "unable_to_adjust_position_independently",
    },

    "Sitting endurance": {
        "Sits comfortably for activities": "maintains_sitting_without_fatigue",
        "Gets tired after long sitting": "gets_tired_after_prolonged_sitting",
        "Gets tired shortly after sitting": "gets_tired_shortly_after_sitting",
        "Cannot tolerate sitting for daily activities": "cannot_tolerate_sitting_for_functional_activities",
    },

    "Pain or discomfort during sitting": {
        "No discomfort": "none",
        "Mild discomfort": "mild_discomfort",
        "Discomfort affects sitting": "moderate_pain_discomfort",
        "Severe pain or distress": "severe_pain_discomfort",
        "Not sure / unable to determine": "unable_to_determine",
    },

    "Skin redness / pressure history": {
        "No redness noticed": "no_redness_or_skin_issues",
        "Redness appears sometimes": "redness_appears_from_time_to_time",
        "Redness appears frequently": "redness_appears_often",
        "Previous pressure sore or skin injury": "previous_skin_breakdown_pressure_injury",
    },
}

HELP_TEXT = {
    "Current seating setup": "Choose the seating used most of the time.",
    "Mobility level": "Choose how the person usually moves during daily activities.",
    "Sitting support level": "Choose how much support the person needs to sit safely.",
    "Pelvic alignment while sitting": "You can choose more than one if the pelvis shows combined positions, such as tilting and twisting.",
    "Back / trunk position while sitting": "Look at the trunk from the front and back if possible.",
    "Head control while sitting": "Choose what usually happens during sitting, not only for a few seconds.",
    "Body stiffness / movement pattern": "Choose how the body usually feels or moves during sitting and movement.",
    "Sitting stability": "Choose how often the person loses position while sitting.",
    "Ability to adjust position independently": "Choose whether the person can correct their own sitting position.",
    "Sitting endurance": "Choose how long sitting remains functional and comfortable.",
    "Pain or discomfort during sitting": "Choose based on report, facial expression, crying, distress, or avoidance.",
    "Skin redness / pressure history": "Check pressure areas after sitting, especially pelvis, sacrum, thighs, and bony areas.",
}

def choose(label):
    user_choice = st.selectbox(
        label,
        list(OPTIONS[label].keys()),
        help=HELP_TEXT.get(label)
    )
    return OPTIONS[label][user_choice], user_choice

# =========================
# MACHINE LEARNING MODEL
# =========================

feature_cols = [col for col in required_columns if col != target_col]

X = df[feature_cols]
y = df[target_col].astype(str).str.strip().str.lower()

categorical_cols = [col for col in feature_cols if col != "age"]
numeric_cols = ["age"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )),
    ]
)

model.fit(X, y)

# =========================
# USER INPUTS
# =========================

st.subheader("Enter Seating Information")
st.write("Please answer based on the person’s usual sitting and mobility condition.")

age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=5,
    step=1,
    format="%d",
)

gender, gender_label = choose("Gender")
current_seating_setup, current_seating_setup_label = choose("Current seating setup")
mobility_level, mobility_level_label = choose("Mobility level")
sitting_support_level, sitting_support_level_label = choose("Sitting support level")

# =========================
# PELVIS MULTI-SELECTION
# =========================

pelvis_default_normal = "Sits evenly / pelvis centered"

pelvic_selected_labels = st.multiselect(
    "Pelvic alignment while sitting",
    [pelvis_default_normal] + list(OPTIONS["Pelvic alignment while sitting"].keys()),
    default=[pelvis_default_normal],
    help=HELP_TEXT["Pelvic alignment while sitting"],
)

if not pelvic_selected_labels:
    pelvic_selected_labels = [pelvis_default_normal]

if len(pelvic_selected_labels) > 1 and pelvis_default_normal in pelvic_selected_labels:
    pelvic_selected_labels.remove(pelvis_default_normal)

if pelvic_selected_labels == [pelvis_default_normal]:
    pelvic_selected_values = ["pelvis_mostly_neutral_centered"]
    pelvic_alignment_while_sitting = "pelvis_mostly_neutral_centered"
    pelvic_label = pelvis_default_normal

elif len(pelvic_selected_labels) == 1:
    pelvic_selected_values = [
        OPTIONS["Pelvic alignment while sitting"][pelvic_selected_labels[0]]
    ]
    pelvic_alignment_while_sitting = pelvic_selected_values[0]
    pelvic_label = pelvic_selected_labels[0]

else:
    pelvic_selected_values = [
        OPTIONS["Pelvic alignment while sitting"][label]
        for label in pelvic_selected_labels
    ]
    pelvic_alignment_while_sitting = "mixed_pelvic_asymmetry"
    pelvic_label = " + ".join(pelvic_selected_labels)

back_trunk_position_while_sitting, trunk_label = choose("Back / trunk position while sitting")
head_control_while_sitting, head_label = choose("Head control while sitting")
body_stiffness_movement_pattern, tone_label = choose("Body stiffness / movement pattern")
sits_stable_without_position_loss, stability_label = choose("Sitting stability")
ability_to_adjust_position_independently, reposition_label = choose("Ability to adjust position independently")
sitting_endurance, endurance_label = choose("Sitting endurance")
pain_or_discomfort_during_sitting, pain_label = choose("Pain or discomfort during sitting")
skin_redness_pressure_history, skin_label = choose("Skin redness / pressure history")

# =========================
# RULE ENGINE: LOW / HIGH / OTHERWISE MODERATE
# =========================

def rule_based_prediction():
    reasons = []
    key_areas = []

    mild_findings = []
    moderate_findings = []
    severe_findings = []

    pelvis_problem_values = [
        v for v in pelvic_selected_values
        if v != "pelvis_mostly_neutral_centered"
    ]

    # -------------------------
    # Mild / moderate / severe findings
    # -------------------------

    if age < 3:
        mild_findings.append("young_age")
        reasons.append("Young age may require closer monitoring because growth can quickly change seating and postural needs.")
        key_areas.append("Growth monitoring")

    if mobility_level in ["mobility_with_supervision", "mobility_with_physical_support"]:
        mild_findings.append("mobility_support")
        reasons.append("Mobility level suggests mild functional limitation and possible seating review needs.")
        key_areas.append("Mobility")

    elif mobility_level in ["limited_functional_mobility", "pushchair_caregiver_dependent"]:
        moderate_findings.append("limited_mobility")
        reasons.append("Mobility level suggests significant functional limitation and increased postural support needs.")
        key_areas.append("Mobility")

    if sitting_support_level in ["hand_support_needed", "trunk_support_needed"]:
        mild_findings.append("sitting_support_needed")
        reasons.append("The person requires external support to maintain sitting balance.")
        key_areas.append("Sitting support")

    elif sitting_support_level == "fully_supported_sitting":
        moderate_findings.append("fully_supported_sitting")
        reasons.append("The person requires significant external sitting support.")
        key_areas.append("Sitting support")

    if len(pelvis_problem_values) == 1:
        if "anterior_pelvic_tilt" in pelvis_problem_values:
            mild_findings.append("anterior_pelvic_tilt")
            reasons.append("Anterior pelvic tilt was selected, which may affect sitting alignment, trunk control, and comfort.")

        elif "posterior_pelvic_tilt" in pelvis_problem_values:
            moderate_findings.append("posterior_pelvic_tilt")
            reasons.append("Posterior pelvic tilt or slouched sitting was selected, which may affect pressure distribution, trunk posture, and functional sitting.")

        elif "pelvic_obliquity_asymmetry" in pelvis_problem_values:
            moderate_findings.append("pelvic_obliquity")
            reasons.append("Pelvic obliquity or side leaning was selected, which may affect weight distribution and spinal alignment.")

        elif "pelvic_rotation" in pelvis_problem_values:
            moderate_findings.append("pelvic_rotation")
            reasons.append("Pelvic rotation was selected, which may affect sitting symmetry, trunk alignment, and functional positioning.")

        key_areas.append("Pelvis")

    elif len(pelvis_problem_values) >= 2:
        moderate_findings.append("mixed_pelvic_asymmetry")
        reasons.append("Multiple pelvic alignment concerns were selected, suggesting a mixed pelvic asymmetry pattern that may affect sitting stability, pressure distribution, and trunk alignment.")
        key_areas.append("Pelvis")

    if back_trunk_position_while_sitting == "mild_leaning_or_asymmetry":
        mild_findings.append("mild_trunk_asymmetry")
        reasons.append("Mild trunk asymmetry may require seating review and monitoring.")
        key_areas.append("Trunk")

    elif back_trunk_position_while_sitting == "clear_leaning_or_visible_curve":
        moderate_findings.append("clear_trunk_asymmetry")
        reasons.append("Back/trunk posture suggests visible asymmetry or reduced trunk stability.")
        key_areas.append("Trunk")

    elif back_trunk_position_while_sitting == "severe_collapse_or_fixed_asymmetry":
        severe_findings.append("severe_trunk_collapse")
        reasons.append("Severe trunk collapse or fixed asymmetry suggests high postural support needs.")
        key_areas.append("Trunk")

    if head_control_while_sitting == "moderate":
        mild_findings.append("moderate_head_control")
        reasons.append("Moderate head control may require additional postural support.")
        key_areas.append("Head control")

    elif head_control_while_sitting == "poor":
        severe_findings.append("poor_head_control")
        reasons.append("Poor head control may affect vision, feeding, breathing, communication, and participation.")
        key_areas.append("Head control")

    if body_stiffness_movement_pattern in ["low_tone", "high_tone"]:
        mild_findings.append("tone_affecting_posture")
        reasons.append("Tone presentation may affect postural control and sitting stability.")
        key_areas.append("Tone / movement pattern")

    elif body_stiffness_movement_pattern in ["fluctuating_tone", "dystonic_movements", "mixed_tone"]:
        moderate_findings.append("complex_tone_pattern")
        reasons.append("Fluctuating tone, dystonic movement, or mixed tone may create changing postural needs during sitting.")
        key_areas.append("Tone / movement pattern")

    if sits_stable_without_position_loss == "loses_position_from_time_to_time":
        mild_findings.append("occasional_position_loss")
        reasons.append("The person loses sitting position from time to time.")
        key_areas.append("Sitting stability")

    elif sits_stable_without_position_loss == "loses_position_many_times_during_sitting":
        moderate_findings.append("frequent_position_loss")
        reasons.append("The person loses position many times during sitting.")
        key_areas.append("Sitting stability")

    elif sits_stable_without_position_loss == "constantly_loses_position":
        severe_findings.append("constant_position_loss")
        reasons.append("The person constantly loses sitting position, suggesting high postural support needs.")
        key_areas.append("Sitting stability")

    if ability_to_adjust_position_independently == "needs_verbal_reminders_cueing":
        mild_findings.append("needs_repositioning_cues")
        reasons.append("The person needs cueing to adjust sitting position.")
        key_areas.append("Repositioning ability")

    elif ability_to_adjust_position_independently == "needs_physical_assistance":
        moderate_findings.append("needs_physical_repositioning")
        reasons.append("The person needs physical help to adjust sitting position.")
        key_areas.append("Repositioning ability")

    elif ability_to_adjust_position_independently == "unable_to_adjust_position_independently":
        severe_findings.append("unable_to_reposition")
        reasons.append("The person is unable to independently correct sitting position.")
        key_areas.append("Repositioning ability")

    if sitting_endurance == "gets_tired_after_prolonged_sitting":
        mild_findings.append("fatigue_after_prolonged_sitting")
        reasons.append("Sitting endurance may require monitoring during prolonged functional activities.")
        key_areas.append("Sitting endurance")

    elif sitting_endurance == "gets_tired_shortly_after_sitting":
        moderate_findings.append("early_sitting_fatigue")
        reasons.append("Reduced sitting endurance may indicate fatigue, discomfort, or inadequate support.")
        key_areas.append("Sitting endurance")

    elif sitting_endurance == "cannot_tolerate_sitting_for_functional_activities":
        severe_findings.append("cannot_tolerate_functional_sitting")
        reasons.append("Very limited sitting tolerance may indicate discomfort, fatigue, poor alignment, or inadequate support.")
        key_areas.append("Sitting endurance")

    if pain_or_discomfort_during_sitting in ["mild_discomfort", "unable_to_determine"]:
        mild_findings.append("mild_or_unclear_discomfort")
        reasons.append("Discomfort or unclear pain response may require monitoring and review.")
        key_areas.append("Pain / discomfort")

    elif pain_or_discomfort_during_sitting == "moderate_pain_discomfort":
        moderate_findings.append("moderate_pain")
        reasons.append("Moderate pain or discomfort during sitting suggests the seating setup should be reviewed.")
        key_areas.append("Pain / discomfort")

    elif pain_or_discomfort_during_sitting == "severe_pain_discomfort":
        severe_findings.append("severe_pain")
        reasons.append("Severe pain or discomfort during sitting is a major seating review indicator.")
        key_areas.append("Pain / discomfort")

    if skin_redness_pressure_history == "redness_appears_from_time_to_time":
        moderate_findings.append("occasional_redness")
        reasons.append("Skin redness after sitting may indicate pressure distribution risk.")
        key_areas.append("Pressure management")

    elif skin_redness_pressure_history == "redness_appears_often":
        severe_findings.append("frequent_redness")
        reasons.append("Frequent skin redness after sitting may indicate increased pressure distribution risk.")
        key_areas.append("Pressure management")

    elif skin_redness_pressure_history == "previous_skin_breakdown_pressure_injury":
        severe_findings.append("previous_pressure_injury")
        reasons.append("Previous skin breakdown or pressure injury indicates increased pressure risk.")
        key_areas.append("Pressure management")

    # -------------------------
    # Seating setup logic
    # Basic chair alone is not a problem.
    # It only appears when there are actual clinical concerns.
    # -------------------------

    concern_count = len(mild_findings) + len(moderate_findings) + len(severe_findings)

    if current_seating_setup == "no_adaptive_seating_system" and concern_count >= 2:
        mild_findings.append("no_adaptive_seating_with_concerns")
        reasons.append("No adaptive seating system is used despite selected postural or functional concerns.")
        key_areas.append("Seating setup")

    elif current_seating_setup == "basic_chair_stroller_only" and concern_count >= 3:
        reasons.append("Current basic seating may need review because other postural or functional concerns were selected.")
        key_areas.append("Seating setup")

    # -------------------------
    # FINAL SAFETY CHECK LOGIC
    # -------------------------

    severe_count = len(severe_findings)
    moderate_count = len(moderate_findings)
    mild_count = len(mild_findings)
    total_concerns = severe_count + moderate_count + mild_count

    clearly_low = (
        severe_count == 0
        and moderate_count == 0
        and mild_count <= 1
    )

    high_red_flag_single = any(flag in severe_findings for flag in [
        "previous_pressure_injury",
        "severe_pain",
        "constant_position_loss",
        "cannot_tolerate_functional_sitting",
    ])

    high_red_flag_combination = (
        severe_count >= 2
        or ("severe_trunk_collapse" in severe_findings and moderate_count >= 1)
        or ("poor_head_control" in severe_findings and (moderate_count >= 1 or mild_count >= 2))
        or ("unable_to_reposition" in severe_findings and moderate_count >= 2)
        or ("frequent_redness" in severe_findings and (moderate_count >= 1 or mild_count >= 2))
    )

    clearly_high = high_red_flag_single or high_red_flag_combination

    if clearly_low:
        final_result = "low"
    elif clearly_high:
        final_result = "high"
    else:
        final_result = "moderate"

    clinical_score = (mild_count * 1) + (moderate_count * 2) + (severe_count * 4)

    return (
        final_result,
        clinical_score,
        reasons,
        list(dict.fromkeys(key_areas)),
        mild_findings,
        moderate_findings,
        severe_findings,
    )

# =========================
# DYNAMIC POTENTIAL COMPLICATIONS
# =========================

def generate_complications(final_result):
    if final_result == "low":
        return ["No major complications were identified based on the current screening answers."]

    complications = []

    pelvis_problem_values = [
        v for v in pelvic_selected_values
        if v != "pelvis_mostly_neutral_centered"
    ]

    if current_seating_setup == "no_adaptive_seating_system":
        complications.append("Lack of adaptive seating may contribute to positioning difficulty when postural or functional concerns are present.")

    elif current_seating_setup == "basic_chair_stroller_only":
        complications.append("Basic seating may require review if it does not adequately support the person’s current postural needs.")

    if "anterior_pelvic_tilt" in pelvis_problem_values:
        complications.append("Anterior pelvic tilt may contribute to increased lumbar extension, fatigue, discomfort, or inefficient sitting posture.")

    if "posterior_pelvic_tilt" in pelvis_problem_values:
        complications.append("Posterior pelvic tilt may increase sacral sitting, pressure concentration, sliding, and reduced functional reach.")

    if "pelvic_obliquity_asymmetry" in pelvis_problem_values:
        complications.append("Pelvic obliquity may contribute to uneven pressure distribution, trunk asymmetry, and spinal alignment concerns.")

    if "pelvic_rotation" in pelvis_problem_values:
        complications.append("Pelvic rotation may affect sitting symmetry, lower limb positioning, and functional sitting alignment.")

    if len(pelvis_problem_values) >= 2:
        complications.append("Combined pelvic asymmetry may increase the risk of postural deterioration and pressure management problems.")

    if back_trunk_position_while_sitting in ["clear_leaning_or_visible_curve", "severe_collapse_or_fixed_asymmetry"]:
        complications.append("Trunk asymmetry may contribute to scoliosis-related concerns, reduced upright sitting tolerance, and reduced function.")

    if head_control_while_sitting in ["moderate", "poor"]:
        complications.append("Reduced head control may affect visual engagement, communication, feeding efficiency, breathing comfort, or participation.")

    if body_stiffness_movement_pattern in ["high_tone", "fluctuating_tone", "dystonic_movements", "mixed_tone"]:
        complications.append("Changing tone or involuntary movements may increase positioning difficulty and risk of postural breakdown.")

    if sits_stable_without_position_loss in ["loses_position_many_times_during_sitting", "constantly_loses_position"]:
        complications.append("Reduced sitting stability may affect upper limb use, participation, safety, and caregiver handling.")

    if ability_to_adjust_position_independently in ["needs_physical_assistance", "unable_to_adjust_position_independently"]:
        complications.append("Reduced ability to self-adjust position may increase dependence and risk of prolonged poor posture.")

    if sitting_endurance in ["gets_tired_shortly_after_sitting", "cannot_tolerate_sitting_for_functional_activities"]:
        complications.append("Reduced sitting endurance may limit school, work, meals, therapy, play, or daily functional activities.")

    if pain_or_discomfort_during_sitting in ["moderate_pain_discomfort", "severe_pain_discomfort"]:
        complications.append("Pain or discomfort may reduce sitting tolerance, participation, mood, and daily function.")

    if skin_redness_pressure_history in ["redness_appears_from_time_to_time", "redness_appears_often", "previous_skin_breakdown_pressure_injury"]:
        complications.append("Skin redness or previous pressure injury may indicate increased risk of pressure injury or skin breakdown.")

    if mobility_level in ["limited_functional_mobility", "pushchair_caregiver_dependent"]:
        complications.append("Reduced independent mobility may increase sitting time and pressure/postural management needs.")

    if not complications:
        complications.append("No specific complication indicators were selected, but monitoring is still recommended based on the screening result.")

    return list(dict.fromkeys(complications))

# =========================
# DYNAMIC RECOMMENDATIONS
# =========================

def get_recommendations(final_result, key_areas):
    recommendations = []

    if final_result == "high":
        recommendations.extend([
            "Comprehensive adaptive seating and postural assessment is strongly recommended.",
            "Review pelvis, trunk, head support, pressure distribution, and sitting tolerance.",
            "Consider pressure mapping and equipment reassessment where available.",
            "Seek professional seating and mobility guidance as soon as possible.",
        ])

    elif final_result == "moderate":
        recommendations.extend([
            "Professional seating review is recommended.",
            "Monitor posture, comfort, skin condition, and sitting endurance.",
            "Review current seating setup and consider adjustment if function or posture is affected.",
            "Reassess if symptoms increase, posture changes, or sitting tolerance decreases.",
        ])

    else:
        recommendations.extend([
            "Continue monitoring posture, comfort, skin condition, and sitting tolerance.",
            "Repeat screening if there is growth, change in function, new pain, skin redness, or change in equipment.",
            "Routine seating review may still be useful as part of long-term postural care.",
        ])

    if final_result in ["moderate", "high"]:
        if "Pressure management" in key_areas:
            recommendations.append("Check skin after sitting and review pressure distribution with a qualified clinician.")

        if "Pelvis" in key_areas:
            recommendations.append("Pelvic positioning should be reviewed because pelvic alignment can influence trunk posture and pressure distribution.")

        if "Trunk" in key_areas:
            recommendations.append("Trunk support and midline alignment should be reviewed during sitting.")

        if "Head control" in key_areas:
            recommendations.append("Head and upper trunk support should be considered during seating review.")

    return list(dict.fromkeys(recommendations))

# =========================
# ML PREDICTION
# =========================

def ml_prediction():
    input_data = pd.DataFrame(
        [
            {
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
                "skin_redness_pressure_history": skin_redness_pressure_history,
            }
        ]
    )

    prediction = model.predict(input_data)[0]
    return str(prediction).strip().lower()

def hybrid_decision(rule_result, ml_result):
    return rule_result

# =========================
# REPORT HELPERS
# =========================

def make_list_html(items):
    if not items:
        return "<li>No major risk indicators were selected based on the current input.</li>"
    return "".join([f"<li>{html.escape(str(item))}</li>" for item in items])

def selected_answers_dict():
    return {
        "Age": age,
        "Gender": gender_label,
        "Current seating setup": current_seating_setup_label,
        "Mobility level": mobility_level_label,
        "Sitting support level": sitting_support_level_label,
        "Pelvic alignment while sitting": pelvic_label,
        "Back / trunk position while sitting": trunk_label,
        "Head control while sitting": head_label,
        "Body stiffness / movement pattern": tone_label,
        "Sitting stability": stability_label,
        "Ability to adjust position independently": reposition_label,
        "Sitting endurance": endurance_label,
        "Pain or discomfort during sitting": pain_label,
        "Skin redness / pressure history": skin_label,
    }

def make_answers_html():
    rows = ""
    for key, value in selected_answers_dict().items():
        rows += (
            "<tr>"
            f"<td><strong>{html.escape(str(key))}</strong></td>"
            f"<td>{html.escape(str(value))}</td>"
            "</tr>"
        )
    return rows

def generate_html_report(
    final_result,
    clinical_score,
    rule_result,
    ml_result,
    reasons,
    complications,
    recommendations,
    key_areas,
):
    risk_color = {
        "low": "#198754",
        "moderate": "#d89b00",
        "high": "#dc3545",
    }.get(final_result, "#333333")

    report_date = datetime.now().strftime("%d %B %Y - %I:%M:%S %p")
    instagram_url = "https://www.instagram.com/mostafaphysio?igsh=M2d3ZjMzOTFxb3M5&utm_source=qr"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>SeatMind AI Screening Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 30px;
                color: #222;
                line-height: 1.5;
            }}
            .container {{
                max-width: 850px;
                margin: auto;
                border: 1px solid #ddd;
                border-radius: 14px;
                padding: 28px;
            }}
            h1 {{
                margin-bottom: 4px;
            }}
            h2 {{
                margin-top: 28px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 6px;
            }}
            .subtitle {{
                color: #555;
                margin-top: 0;
            }}
            .risk-box {{
                border-radius: 12px;
                padding: 18px;
                background: #f7f7f7;
                border-left: 8px solid {risk_color};
                margin: 20px 0;
            }}
            .risk {{
                font-size: 28px;
                font-weight: bold;
                color: {risk_color};
                text-transform: uppercase;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            td {{
                border: 1px solid #ddd;
                padding: 10px;
                vertical-align: top;
            }}
            ul {{
                padding-left: 22px;
            }}
            .disclaimer {{
                background: #fff3cd;
                border: 1px solid #ffeeba;
                padding: 14px;
                border-radius: 10px;
                margin-top: 20px;
            }}
            .contact {{
                background: #f2f6ff;
                padding: 14px;
                border-radius: 10px;
                margin-top: 20px;
            }}
            @media print {{
                body {{
                    margin: 0;
                }}
                .container {{
                    border: none;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SeatMind AI Screening Report</h1>
            <p class="subtitle">Seating & Postural Risk Screening Tool</p>
            <p><strong>Date:</strong> {html.escape(report_date)}</p>

            <div class="risk-box">
                <p><strong>Final Screening Result:</strong></p>
                <div class="risk">{html.escape(final_result.upper())} RISK</div>
                <p><strong>Clinical safety score:</strong> {html.escape(str(clinical_score))}</p>
                <p><strong>Rule-based safety result:</strong> {html.escape(rule_result.upper())}</p>
                <p><strong>Machine learning support result:</strong> {html.escape(ml_result.upper())} — supportive only, not final decision</p>
            </div>

            <h2>Person Information & Selected Answers</h2>
            <table>
                {make_answers_html()}
            </table>

            <h2>Key Areas of Concern</h2>
            <ul>
                {make_list_html(key_areas)}
            </ul>

            <h2>Clinical Reasoning</h2>
            <ul>
                {make_list_html(reasons)}
            </ul>

            <h2>Potential Complications</h2>
            <ul>
                {make_list_html(complications)}
            </ul>

            <h2>Recommended Next Step</h2>
            <ul>
                {make_list_html(recommendations)}
            </ul>

            <div class="disclaimer">
                <strong>Disclaimer:</strong><br>
                This is an early screening and decision-support tool. It does not replace a full professional seating and mobility assessment.
            </div>

            <div class="contact">
                <strong>Professional Guidance:</strong><br>
                Mostafa Ahmed<br>
                Adaptive Seating Specialist & Senior Physiotherapist<br>
                Instagram: <a href="{html.escape(instagram_url)}">@mostafaphysio</a>
            </div>
        </div>
    </body>
    </html>
    """

# =========================
# FINAL OUTPUT
# =========================

if st.button("Generate Screening Report"):
    (
        rule_result,
        clinical_score,
        reasons,
        key_areas,
        mild_findings,
        moderate_findings,
        severe_findings,
    ) = rule_based_prediction()

    ml_result = ml_prediction()
    final_result = hybrid_decision(rule_result, ml_result)

    if not reasons:
        reasons = ["No major seating or postural risk indicators were selected based on the current input."]

    if not key_areas:
        key_areas = ["No major concern areas identified"]

    complications = generate_complications(final_result)
    recommendations = get_recommendations(final_result, key_areas)

    st.markdown("---")
    st.subheader("SeatMind AI Screening Report")

    if final_result == "high":
        st.error("Final Screening Result: HIGH RISK")
    elif final_result == "moderate":
        st.warning("Final Screening Result: MODERATE RISK")
    else:
        st.success("Final Screening Result: LOW RISK")

    st.markdown("### Screening Summary")
    st.write(f"Clinical safety score: {clinical_score}")
    st.write(f"Rule-based safety result: {rule_result.upper()}")
    st.write(f"Machine learning support result: {ml_result.upper()} — supportive only, not final decision")

    st.markdown("### Person Information & Selected Answers")
    for key, value in selected_answers_dict().items():
        st.write(f"**{key}:** {value}")

    st.markdown("### Key Areas of Concern")
    for area in key_areas:
        st.write(f"- {area}")

    st.markdown("### Clinical Reasoning")
    for reason in reasons:
        st.write(f"- {reason}")

    st.markdown("### Potential Complications")
    for item in complications:
        st.write(f"- {item}")

    st.markdown("### Recommended Next Step")
    for item in recommendations:
        st.write(f"- {item}")

    st.warning(
        "This is an early screening and decision-support tool. "
        "It does not replace a full professional seating and mobility assessment."
    )

    st.markdown(
        """
        ### Need Clinical Assessment or Professional Guidance?

        **Mostafa Ahmed**  
        Adaptive Seating Specialist & Senior Physiotherapist  

        📲 Instagram: [@mostafaphysio](https://www.instagram.com/mostafaphysio?igsh=M2d3ZjMzOTFxb3M5&utm_source=qr)
        """
    )

    report_html = generate_html_report(
        final_result=final_result,
        clinical_score=clinical_score,
        rule_result=rule_result,
        ml_result=ml_result,
        reasons=reasons,
        complications=complications,
        recommendations=recommendations,
        key_areas=key_areas,
    )

    st.download_button(
        label="Download Screening Report",
        data=report_html,
        file_name="SeatMind_AI_Screening_Report.html",
        mime="text/html",
    )
