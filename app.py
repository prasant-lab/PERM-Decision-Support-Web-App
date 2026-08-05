import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.impute import SimpleImputer

import sklearn.compose._column_transformer as _column_transformer


class _RemainderColsList(list):
    def __init__(self, columns=None, *args, **kwargs):
        if columns is None:
            columns = []
        super().__init__(columns)
        for key, value in kwargs.items():
            setattr(self, key, value)


_RemainderColsList.__module__ = "sklearn.compose._column_transformer"
_column_transformer._RemainderColsList = _RemainderColsList


PROJECT_DIR = Path.cwd()

CLASSIFICATION_MODEL_PATH = PROJECT_DIR / "best_classification_model.joblib"
REGRESSION_MODEL_PATH = PROJECT_DIR / "best_regression_model.joblib"
METADATA_PATH = PROJECT_DIR / "model_metadata.json"
CLEAN_DATA_PATH = PROJECT_DIR / "perm_cleaned_fy2025_q4.csv"

IMPORTANCE_PATH = PROJECT_DIR / "permutation_importance.csv"
CLASSIFICATION_RESULTS_PATH = PROJECT_DIR / "classification_results.csv"
REGRESSION_RESULTS_PATH = PROJECT_DIR / "regression_results.csv"
FAIRNESS_RESULTS_PATH = PROJECT_DIR / "fairness_group_metrics.csv"
TAIL_RESULTS_PATH = PROJECT_DIR / "tail_error_analysis.csv"


st.set_page_config(
    page_title="PERM Decision Support Web Application",
    page_icon="📊",
    layout="wide"
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }
    .main-title {
        font-size: 38px;
        font-weight: 900;
        color: #17324D;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 18px;
        color: #555555;
        margin-bottom: 20px;
    }
    .warning-box {
        background-color: #FFF4CC;
        padding: 16px;
        border-radius: 14px;
        border-left: 7px solid #F0AD4E;
        color: #3B3100;
        margin-bottom: 15px;
    }
    .info-box {
        background-color: #EAF3FF;
        padding: 16px;
        border-radius: 14px;
        border-left: 7px solid #1F77B4;
        color: #17324D;
        margin-bottom: 15px;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 16px;
        border-radius: 14px;
        border-left: 7px solid #2E7D32;
        color: #143D19;
        margin-bottom: 15px;
    }
    .section-card {
        background-color: #F7F9FC;
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        margin-bottom: 12px;
    }
    .small-text {
        font-size: 14px;
        color: #555555;
    }
    </style>
    """,
    unsafe_allow_html=True
)



def patch_simple_imputers(model):
    visited = set()

    def walk(obj):
        if obj is None:
            return

        object_id = id(obj)

        if object_id in visited:
            return

        visited.add(object_id)

        if isinstance(obj, SimpleImputer):
            if not hasattr(obj, "_fill_dtype"):
                try:
                    obj._fill_dtype = np.asarray(obj.statistics_).dtype
                except Exception:
                    obj._fill_dtype = object

        if hasattr(obj, "steps"):
            for _, step in obj.steps:
                walk(step)

        if hasattr(obj, "named_steps"):
            for step in obj.named_steps.values():
                walk(step)

        if hasattr(obj, "transformers"):
            for _, transformer, _ in obj.transformers:
                if transformer not in ["drop", "passthrough"]:
                    walk(transformer)

        if hasattr(obj, "transformers_"):
            for _, transformer, _ in obj.transformers_:
                if transformer not in ["drop", "passthrough"]:
                    walk(transformer)

    walk(model)
    return model


@st.cache_resource
def load_models():
    missing_files = []

    for file_path in [CLASSIFICATION_MODEL_PATH, REGRESSION_MODEL_PATH, METADATA_PATH]:
        if not file_path.exists():
            missing_files.append(str(file_path))

    if missing_files:
        st.error("Required model files are missing.")
        st.write(missing_files)
        st.stop()

    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    regression_model = joblib.load(REGRESSION_MODEL_PATH)

    classification_model = patch_simple_imputers(classification_model)
    regression_model = patch_simple_imputers(regression_model)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return classification_model, regression_model, metadata


@st.cache_data
def load_clean_data():
    if not CLEAN_DATA_PATH.exists():
        st.error("Clean dataset file is missing.")
        st.stop()

    return pd.read_csv(CLEAN_DATA_PATH, low_memory=False)


@st.cache_data
def load_csv(path):
    path = Path(path)

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


classification_model, regression_model, metadata = load_models()
df = load_clean_data()

importance_df = load_csv(IMPORTANCE_PATH)
classification_results_df = load_csv(CLASSIFICATION_RESULTS_PATH)
regression_results_df = load_csv(REGRESSION_RESULTS_PATH)
fairness_results_df = load_csv(FAIRNESS_RESULTS_PATH)
tail_results_df = load_csv(TAIL_RESULTS_PATH)

classification_features = metadata.get("classification_features", [])
regression_features = metadata.get("regression_features", [])
all_features = sorted(list(set(classification_features + regression_features)))

numeric_features = []
categorical_features = []

for feature in all_features:
    if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
        numeric_features.append(feature)
    else:
        categorical_features.append(feature)


def clean_label(name):
    return str(name).replace("_", " ").title()


def get_numeric_defaults(feature):
    if feature not in df.columns:
        return 0.0, 0.0, 1.0

    values = pd.to_numeric(df[feature], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()

    if len(values) == 0:
        return 0.0, 0.0, 1.0

    minimum = float(values.quantile(0.01))
    maximum = float(values.quantile(0.99))
    default = float(values.median())

    if minimum == maximum:
        maximum = minimum + 1.0

    default = min(max(default, minimum), maximum)

    return default, minimum, maximum


def get_category_options(feature):
    if feature not in df.columns:
        return ["MISSING", "OTHER"]

    values = df[feature].dropna().astype(str)
    values = values[values.str.lower() != "nan"]
    options = values.value_counts().head(60).index.tolist()

    if not options:
        options = ["MISSING"]

    if "MISSING" not in options:
        options.append("MISSING")

    if "OTHER" not in options:
        options.append("OTHER")

    return options


def baseline_row(features):
    row = {}

    for feature in features:
        if feature not in df.columns:
            row[feature] = np.nan

        elif feature in numeric_features:
            values = pd.to_numeric(df[feature], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            row[feature] = float(values.median()) if len(values) else 0.0

        else:
            values = df[feature].dropna().astype(str)
            mode_values = values.mode()
            row[feature] = mode_values.iloc[0] if len(mode_values) else "MISSING"

    return pd.DataFrame([row])


def selected_form_features():
    selected = []

    priority_keywords = [
        "WORKSITE_STATE", "EMPLOYER_STATE", "STATE", "SOC_TITLE", "SOC",
        "JOB_TITLE", "TITLE", "WAGE", "WAGE_BAND", "PW", "NAICS",
        "FULL_TIME", "EDUCATION", "EXPERIENCE", "LEVEL"
    ]

    for keyword in priority_keywords:
        for feature in all_features:
            if keyword in feature.upper() and feature not in selected:
                selected.append(feature)

    if not importance_df.empty and "Feature" in importance_df.columns:
        for feature in importance_df["Feature"].astype(str).head(25).tolist():
            if feature in all_features and feature not in selected:
                selected.append(feature)

    for feature in all_features:
        if feature not in selected:
            selected.append(feature)

        if len(selected) >= 18:
            break

    return selected[:18]


def build_model_input(user_values, features):
    row = baseline_row(features)

    for feature, value in user_values.items():
        if feature in row.columns:
            row.loc[0, feature] = value

    return row


def run_prediction(user_values):
    x_class = build_model_input(user_values, classification_features)
    x_reg = build_model_input(user_values, regression_features)

    predicted_outcome = classification_model.predict(x_class)[0]
    predicted_days = float(regression_model.predict(x_reg)[0])
    predicted_days = max(predicted_days, 0)

    if hasattr(classification_model, "predict_proba"):
        probabilities = classification_model.predict_proba(x_class)[0]
        classes = classification_model.classes_

        probability_table = pd.DataFrame({
            "Outcome": classes,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

    else:
        probability_table = pd.DataFrame({
            "Outcome": [predicted_outcome],
            "Probability": [1.0]
        })

    return predicted_outcome, predicted_days, probability_table, x_class, x_reg


def main_probability(probability_table):
    preferred_labels = ["CERTIFIED_GROUP", "CERTIFIED", "CERTIFIED-EXPIRED", "CERTIFIED EXPIRED"]

    for label in preferred_labels:
        match = probability_table[probability_table["Outcome"].astype(str).str.upper() == label]

        if len(match):
            return str(match["Outcome"].iloc[0]), float(match["Probability"].iloc[0])

    top = probability_table.iloc[0]

    return str(top["Outcome"]), float(top["Probability"])


def effect_table(model, input_row, features, task):
    if not features:
        return pd.DataFrame()

    base_row = baseline_row(features)

    if not importance_df.empty and "Feature" in importance_df.columns:
        candidates = [
            feature for feature in importance_df["Feature"].astype(str).tolist()
            if feature in features and feature in input_row.columns
        ]
    else:
        candidates = [feature for feature in features if feature in input_row.columns]

    candidates = candidates[:12]

    if task == "classification":
        if not hasattr(model, "predict_proba"):
            return pd.DataFrame()

        classes = list(model.classes_)
        base_prediction = model.predict(input_row)[0]
        target_class = "CERTIFIED_GROUP" if "CERTIFIED_GROUP" in classes else base_prediction
        target_index = classes.index(target_class)
        base_value = float(model.predict_proba(input_row)[0][target_index])
        effect_name = f"Effect on {target_class} probability"

    else:
        base_value = float(model.predict(input_row)[0])
        effect_name = "Effect on estimated days"

    rows = []

    for feature in candidates:
        changed_row = input_row.copy()
        changed_row.loc[0, feature] = base_row.loc[0, feature]

        if task == "classification":
            changed_value = float(model.predict_proba(changed_row)[0][target_index])
        else:
            changed_value = float(model.predict(changed_row)[0])

        effect = base_value - changed_value

        rows.append({
            "Feature": clean_label(feature),
            "Current Value": str(input_row.loc[0, feature]),
            "Baseline Value": str(base_row.loc[0, feature]),
            effect_name: effect
        })

    output = pd.DataFrame(rows)

    if not output.empty:
        effect_col = output.columns[-1]
        output["Absolute Effect"] = output[effect_col].abs()
        output = output.sort_values("Absolute Effect", ascending=False)
        output = output.drop(columns=["Absolute Effect"]).head(8)

    return output


def probability_chart(probability_table):
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_data = probability_table.sort_values("Probability", ascending=True)
    colours = ["#2E86AB", "#A23B72", "#F18F01", "#6A994E", "#C73E1D"][:len(plot_data)]

    ax.barh(plot_data["Outcome"].astype(str), plot_data["Probability"], color=colours)
    ax.set_title("Outcome Probability", fontweight="bold", pad=14)
    ax.set_xlabel("Probability", fontweight="bold")
    ax.set_xlim(0, 1)

    for index, value in enumerate(plot_data["Probability"]):
        ax.text(min(value + 0.01, 0.96), index, f"{value:.1%}", va="center", fontweight="bold")

    plt.tight_layout()
    return fig


def effect_chart(data, title):
    if data.empty:
        return None

    effect_col = data.columns[-1]
    plot_data = data.sort_values(effect_col)

    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["#C73E1D" if value < 0 else "#2E86AB" for value in plot_data[effect_col]]

    ax.barh(plot_data["Feature"], plot_data[effect_col], color=colours)
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xlabel("Model Effect", fontweight="bold")

    for index, value in enumerate(plot_data[effect_col]):
        ax.text(value, index, f" {value:.3f}", va="center", fontweight="bold")

    plt.tight_layout()
    return fig


st.markdown(
    '<div class="main-title">PERM Labour Certification Decision Support Web Application</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Explainable AI prototype for outcome probability and processing time planning.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="warning-box">
    This system supports planning only. It is not legal advice, does not guarantee an official decision, and does not replace professional judgement.
    </div>
    """,
    unsafe_allow_html=True
)

tab_prediction, tab_dataset, tab_results, tab_testing, tab_limitations = st.tabs(
    ["Prediction Tool", "Dataset Overview", "Model Results", "User Testing", "Limitations"]
)


with tab_prediction:
    st.subheader("Case Input Form")
    st.markdown(
        '<div class="small-text">Enter case details. Fields not shown use safe baseline values from the dataset.</div>',
        unsafe_allow_html=True
    )

    form_features = selected_form_features()
    user_values = {}

    col1, col2 = st.columns(2)

    for index, feature in enumerate(form_features):
        active_col = col1 if index % 2 == 0 else col2

        with active_col:
            if feature in numeric_features:
                default, minimum, maximum = get_numeric_defaults(feature)
                step = max((maximum - minimum) / 100, 1.0)

                user_values[feature] = st.number_input(
                    clean_label(feature),
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=step,
                    key=f"num_{feature}"
                )

            else:
                user_values[feature] = st.selectbox(
                    clean_label(feature),
                    options=get_category_options(feature),
                    key=f"cat_{feature}"
                )

    st.divider()

    if st.button("Generate Prediction", type="primary", use_container_width=True):
        predicted_outcome, predicted_days, probability_table, x_class, x_reg = run_prediction(user_values)
        probability_label, probability_value = main_probability(probability_table)

        st.subheader("Prediction Results")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Most Likely Outcome", str(predicted_outcome))
        metric2.metric(f"{probability_label} Probability", f"{probability_value:.1%}")
        metric3.metric("Estimated Processing Time", f"{predicted_days:,.0f} days")

        st.progress(min(max(probability_value, 0), 1))

        chart_col, table_col = st.columns([1.2, 1])

        with chart_col:
            st.pyplot(probability_chart(probability_table), use_container_width=True)

        with table_col:
            show_probability = probability_table.copy()
            show_probability["Probability"] = show_probability["Probability"].map(lambda value: f"{value:.2%}")
            st.dataframe(show_probability, use_container_width=True, hide_index=True)

        st.subheader("Explanation")

        class_effects = effect_table(classification_model, x_class, classification_features, "classification")
        reg_effects = effect_table(regression_model, x_reg, regression_features, "regression")

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("#### Outcome Explanation")

            if class_effects.empty:
                st.info("Outcome explanation is not available for this model.")
            else:
                st.pyplot(effect_chart(class_effects, "Feature Effect on Outcome Probability"), use_container_width=True)
                st.dataframe(class_effects, use_container_width=True, hide_index=True)

        with exp_col2:
            st.markdown("#### Processing Time Explanation")

            if reg_effects.empty:
                st.info("Processing time explanation is not available for this model.")
            else:
                st.pyplot(effect_chart(reg_effects, "Feature Effect on Estimated Days"), use_container_width=True)
                st.dataframe(reg_effects, use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div class="info-box">
            Explanations show model behaviour compared with baseline values. They do not prove legal cause and effect.
            </div>
            """,
            unsafe_allow_html=True
        )


with tab_dataset:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Classification Features", f"{len(classification_features):,}")
    c4.metric("Regression Features", f"{len(regression_features):,}")

    if "CASE_STATUS_CLEAN" in df.columns:
        counts = df["CASE_STATUS_CLEAN"].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(counts.index.astype(str), counts.values, color="#2E86AB")
        ax.set_title("Top Case Status Values", fontweight="bold", pad=15)
        ax.set_xlabel("Case Status", fontweight="bold")
        ax.set_ylabel("Number of Cases", fontweight="bold")
        plt.xticks(rotation=35, ha="right")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:,.0f}",
                ha="center",
                va="bottom",
                fontweight="bold"
            )

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    if "DAYS_TO_DECISION" in df.columns:
        days = pd.to_numeric(df["DAYS_TO_DECISION"], errors="coerce").dropna()
        days = days[(days >= 0) & (days <= days.quantile(0.99))]

        if len(days):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(days, bins=30, color="#A23B72", edgecolor="white")
            ax.axvline(
                days.median(),
                color="#F18F01",
                linewidth=2,
                label=f"Median: {days.median():,.0f} days"
            )
            ax.set_title("Processing Time Distribution", fontweight="bold", pad=15)
            ax.set_xlabel("Days to Decision", fontweight="bold")
            ax.set_ylabel("Number of Cases", fontweight="bold")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    st.subheader("Sample Data")
    st.dataframe(df.head(25), use_container_width=True)


with tab_results:
    st.subheader("Model Results")

    st.markdown(
        f"""
        <div class="success-box">
        Classification model: <b>{metadata.get("best_classification_model", "Not recorded")}</b><br>
        Regression model: <b>{metadata.get("best_regression_model", "Not recorded")}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    r1, r2 = st.columns(2)

    with r1:
        st.markdown("#### Classification Results")

        if classification_results_df.empty:
            st.info("Classification results file is not available.")
        else:
            st.dataframe(classification_results_df, use_container_width=True, hide_index=True)

    with r2:
        st.markdown("#### Regression Results")

        if regression_results_df.empty:
            st.info("Regression results file is not available.")
        else:
            st.dataframe(regression_results_df, use_container_width=True, hide_index=True)

    st.markdown("#### Global Feature Importance")

    if not importance_df.empty:
        show_importance = importance_df.head(15).copy()

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_data = show_importance.sort_values("Importance")
        ax.barh(plot_data["Feature"], plot_data["Importance"], color="#2E86AB")
        ax.set_title("Top Global Features", fontweight="bold", pad=15)
        ax.set_xlabel("Importance", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.dataframe(show_importance, use_container_width=True, hide_index=True)

    else:
        st.info("Feature importance file is not available.")

    st.markdown("#### Fairness Diagnostics")

    if fairness_results_df.empty:
        st.info("Fairness results file is not available.")
    else:
        st.dataframe(fairness_results_df, use_container_width=True, hide_index=True)

    st.markdown("#### Tail Error Analysis")

    if tail_results_df.empty:
        st.info("Tail error analysis file is not available.")
    else:
        st.dataframe(tail_results_df, use_container_width=True, hide_index=True)


with tab_testing:
    st.subheader("User Testing Plan")

    st.markdown(
        """
        <div class="section-card">
        The app should be tested with a small group of users or proxy users. Each participant should complete a simple task and answer the questionnaire.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Suggested Test Tasks")
    st.write("1. Open the app and read the disclaimer.")
    st.write("2. Enter or adjust the case details.")
    st.write("3. Generate a prediction.")
    st.write("4. Read the outcome probability and estimated processing time.")
    st.write("5. Review the explanation chart.")
    st.write("6. Answer the questionnaire.")

    st.markdown("#### Questionnaire Items")

    questionnaire = pd.DataFrame({
        "Question": [
            "The app was easy to use.",
            "The input fields were clear.",
            "The prediction result was easy to understand.",
            "The processing time estimate was useful.",
            "The explanation chart helped me understand the output.",
            "The disclaimer made the system limits clear.",
            "I would trust this as a planning support tool.",
            "What should be improved?"
        ],
        "Answer Type": [
            "1 to 5 scale",
            "1 to 5 scale",
            "1 to 5 scale",
            "1 to 5 scale",
            "1 to 5 scale",
            "1 to 5 scale",
            "1 to 5 scale",
            "Open text"
        ]
    })

    st.dataframe(questionnaire, use_container_width=True, hide_index=True)

    st.markdown("#### Screenshots to Save")
    st.write(
        "Input form, prediction results, probability chart, explanation chart, dataset tab, model results tab, user testing tab, and limitations tab."
    )


with tab_limitations:
    st.subheader("Purpose")
    st.write("This application is a dissertation prototype for PERM labour certification planning.")

    st.subheader("What It Does")
    st.write("It estimates likely case outcome probabilities and processing time using trained machine learning models.")

    st.subheader("What It Does Not Do")
    st.write(
        "It does not provide legal advice, does not guarantee official decisions, and does not replace immigration professionals."
    )

    st.subheader("Main Limitations")
    st.write(
        "The model is based on historical disclosure data and may not capture policy changes, missing context, exceptional cases, or future data changes."
    )

    st.subheader("Responsible Use")
    st.write(
        "Predictions should be used as planning signals only. Users should not treat the system as an official decision-maker."
    )
