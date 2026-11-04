"""Interactive Virtual Lab for Linear Regression using the Diabetes dataset."""

from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
HISTORY_FILE = ROOT / "experiment_history.json"
RANDOM_STATE = 42


@st.cache_data
def load_data():
    dataset = load_diabetes(as_frame=True)
    return dataset, dataset.data.copy(), dataset.target.copy()


def train_model(features, target, train_size, random_state, standardize, fit_intercept):
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, train_size=train_size, random_state=random_state
    )
    scaler = StandardScaler() if standardize else None
    x_train_model = scaler.fit_transform(x_train) if scaler else x_train
    x_test_model = scaler.transform(x_test) if scaler else x_test
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(x_train_model, y_train)
    predictions = model.predict(x_test_model)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R² Score": r2_score(y_test, predictions),
    }
    return x_train, x_test, y_train, y_test, scaler, model, predictions, metrics


def history_records():
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def save_history(record):
    records = history_records()
    records.append(record)
    HISTORY_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")


def style_app():
    st.set_page_config(page_title="Linear Regression", layout="wide")
    st.markdown(
        """
        <style>
        :root { --ink:#17243b; --muted:#667085; --line:#dfe5ec; --blue:#2f6f9f; --green:#22a06b; }
        .stApp { background:#f3f6fa; color:var(--ink); }
        [data-testid="stAppViewContainer"] > .main { padding-top:0; }
        .block-container { max-width:1400px; padding:2rem 2.25rem 3rem; }
        [data-testid="stSidebar"] { background:#101c35; border-right:1px solid #263452; }
        [data-testid="stSidebar"] * { color:#eef3f8; }
        [data-testid="stSidebar"] .stRadio label { border-radius:3px; padding:.18rem .4rem; margin:.05rem 0; }
        [data-testid="stSidebar"] .stRadio label:hover { background:#1b2b4a; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#aeb9c8; }
        .sidebar-brand { border-bottom:1px solid #33425d; padding-bottom:1rem; margin-bottom:1rem; }
        .sidebar-brand strong { color:#ffffff; font-size:1.1rem; font-weight:600; }
        .card { background:#ffffff; padding:1.25rem 1.4rem; border:1px solid var(--line);
                border-radius:8px; box-shadow:0 1px 5px rgba(31,45,79,.04); margin-bottom:1rem; }
        .card h3 { margin-top:0; color:var(--ink); font-size:1.05rem; }
        .eyebrow, .section-kicker { color:var(--green); text-transform:uppercase; letter-spacing:.08em;
                   font-size:.68rem; font-weight:700; }
        .section-kicker { margin-bottom:.15rem; }
        .breadcrumb { color:#718096; font-size:.78rem; margin-bottom:.55rem; }
        .status-dot { display:inline-block; width:7px; height:7px; background:var(--green);
                      border-radius:50%; margin-right:6px; }
        .step { border:1px solid var(--line); border-left:3px solid var(--green); padding:.7rem .9rem;
                background:#ffffff; border-radius:5px; margin:.5rem 0; }
        .metric-card { background:#ffffff; border:1px solid var(--line); border-radius:8px;
                       padding:1rem 1.1rem; min-height:104px; }
        .metric-label { color:#667085; font-size:.82rem; margin-bottom:.45rem; }
        .metric-value { color:#17243b; font-size:1.7rem; font-weight:700; line-height:1.1; }
        .status-line { color:#667085; font-size:.86rem; }
        div[data-testid="stMetric"] { background:#ffffff; border:1px solid var(--line);
                                      border-radius:7px; padding:.7rem .85rem; box-shadow:none; }
        .stButton > button { border-radius:4px; font-weight:600; }
        h1,h2,h3 { color:var(--ink); letter-spacing:-.01em; }
        h1 { font-size:1.75rem; }
        h2 { font-size:1.35rem; }
        @media (max-width:700px) {
            .block-container { padding:1.25rem .85rem 2rem; }
            h1 { font-size:1.5rem; }
            .card { padding:.9rem; }
            .metric-card { min-height:88px; }
            .metric-value { font-size:1.35rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(title, body):
    st.markdown(f'<div class="card"><h3>{title}</h3>{body}</div>', unsafe_allow_html=True)


def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def chart_card(title):
    st.markdown(f'<div class="card"><h3>{title}</h3>', unsafe_allow_html=True)


def close_card():
    st.markdown("</div>", unsafe_allow_html=True)


def page_header(title, description):
    st.markdown(
        f'<div class="breadcrumb">Home &nbsp;/&nbsp; {title}</div>'
        f'<div class="section-kicker"><span class="status-dot"></span>Ready</div>'
        f'<h1>{title}</h1><p style="color:#667085;margin-top:-.6rem;margin-bottom:1.25rem">{description}</p>',
        unsafe_allow_html=True,
    )


def navigate(page):
    st.session_state.page = page
    st.rerun()


def page_aim():
    page_header("Aim", "Understand how Linear Regression models relationships and predicts continuous values.")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Aim")
    st.write("To implement Linear Regression on a suitable dataset and analyse regression coefficients, predictions and model performance.")
    st.subheader("Learning outcomes")
    outcomes = [
        "Understand the fundamentals of Linear Regression.",
        "Identify independent and dependent variables.",
        "Understand slope and intercept.",
        "Train a Linear Regression model.",
        "Generate predictions.",
        "Evaluate regression performance.",
    ]
    for outcome in outcomes:
        st.markdown(f'<p><span class="status-dot"></span>{outcome}</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("Read the Theory →", type="primary"):
        navigate("Theory")


def page_theory():
    page_header("Theory", "Understand the equation, optimization, errors, and evaluation behind regression.")
    sections = [
        ("What is Linear Regression?", "Linear Regression predicts a continuous dependent variable from one or more independent variables by fitting the best line or hyperplane through the observations."),
        ("Simple and multiple regression", "Simple regression uses one feature: y = mx + c. Multiple regression uses several features: y = b₀ + b₁x₁ + ... + bₙxₙ."),
        ("Independent and dependent variables", "Independent variables (X) are inputs or predictors. The dependent variable (y) is the numerical outcome the model learns to predict."),
        ("Ordinary Least Squares", "OLS chooses coefficients that minimize the sum of squared residuals. A residual is the difference between an observed value and its prediction."),
        ("Overfitting and underfitting", "Underfitting is too simple to capture the relationship. Overfitting memorizes training noise and performs poorly on unseen data. A held-out test set helps detect this."),
    ]
    for title, text in sections:
        with st.expander(title, expanded=title == "What is Linear Regression?"):
            st.markdown(text)
    st.subheader("Core formulas")
    formulas = pd.DataFrame([
        ["Simple regression", r"$y = mx + c$", "m is slope; c is intercept."],
        ["Multiple regression", r"$y = b_0 + b_1x_1 + \cdots + b_nx_n$", "Each b coefficient changes the prediction for one feature."],
        ["Mean Squared Error", r"$MSE = \frac{1}{n}\sum(y_i-\hat{y}_i)^2$", "Average squared prediction error."],
        ["Mean Absolute Error", r"$MAE = \frac{1}{n}\sum|y_i-\hat{y}_i|$", "Average absolute distance from actual values."],
        ["Root MSE", r"$RMSE = \sqrt{MSE}$", "Error in the target's original units."],
        ["R-squared", r"$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$", "Proportion of target variation explained by the model."],
    ], columns=["Concept", "Formula", "Meaning"])
    st.table(formulas)
    st.subheader("Regression-line illustration")
    figure, axis = plt.subplots(figsize=(8, 3.2))
    x = pd.Series(range(1, 11))
    axis.scatter(x, x * 2 + [1, -1, 2, 0, -2, 1, 0, 2, -1, 1], color="#4f46e5", label="Observed")
    axis.plot(x, x * 2, color="#ef4444", label="Fitted relationship")
    axis.set_xlabel("Independent variable"); axis.set_ylabel("Dependent variable"); axis.legend()
    st.pyplot(figure, use_container_width=True)
    st.caption("In the actual Simulation page, the chart is generated from the selected Diabetes features and trained model.")


def page_procedure():
    page_header("Procedure", "Follow the complete practical workflow in twelve reproducible steps.")
    steps = [
        ("Import libraries", "Load Pandas, Matplotlib, Seaborn, and Scikit-learn tools."),
        ("Import dataset", "Load the built-in Diabetes dataset without a manual download."),
        ("Explore dataset", "Inspect shape, columns, data types, and descriptive statistics."),
        ("Data preprocessing", "Check missing values and duplicates; select features and target."),
        ("Feature selection", "Choose one feature for simple or several for multiple regression."),
        ("Train-test split", "Reserve a configurable percentage for unbiased evaluation."),
        ("Model initialization", "Create LinearRegression with the fit-intercept option."),
        ("Model training", "Fit coefficients using Ordinary Least Squares on training data."),
        ("Prediction", "Generate numerical predictions for unseen test records."),
        ("Model evaluation", "Calculate MAE, MSE, RMSE, and R²."),
        ("Visualization", "Plot actual-vs-predicted values, residuals, and feature effects."),
        ("Result interpretation", "Explain coefficients, errors, and the practical conclusion."),
    ]
    for index, (title, explanation) in enumerate(steps, 1):
        st.markdown(f'<div class="step"><b>{index:02d} · {title}</b><br><span style="color:#667085">{explanation}</span></div>', unsafe_allow_html=True)
    if st.button("Proceed to Data Preparation", type="primary"):
        navigate("Data Preparation")


def page_data():
    dataset, features, target = load_data()
    page_header("Data Preparation", "Inspect the Diabetes dataset and configure a leakage-safe split.")
    st.subheader("Dataset overview")
    metric_columns = st.columns(4)
    with metric_columns[0]:
        metric_card("Total records", f"{len(features):,}")
    with metric_columns[1]:
        metric_card("Total features", f"{features.shape[1]}")
    with metric_columns[2]:
        metric_card("Missing values", f"{int(features.isna().sum().sum() + target.isna().sum())}")
    with metric_columns[3]:
        metric_card("Duplicate records", f"{int(features.duplicated().sum())}")
    overview, preparation = st.columns([1.5, 1], gap="large")
    with overview:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Dataset information")
        st.write("**Dataset:** Diabetes (Scikit-learn)")
        st.write("**Description:** A regression dataset containing physiological measurements and a quantitative disease-progression target.")
        st.write(f"**Target variable:** target &nbsp; **Shape:** {features.shape[0]} rows × {features.shape[1]} features")
        with st.expander("Column names and data types"):
            st.dataframe(pd.DataFrame({"Column": features.columns, "Type": features.dtypes.astype(str).values}), hide_index=True, use_container_width=True)
        st.markdown("**First five records**")
        st.dataframe(features.head().assign(target=target.head()), hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with preparation:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Train-test configuration")
        st.caption("Configure the split used before modelling. Preprocessing is fitted on training data only.")
        prep_train = st.slider("Training percentage", 60, 90, 80, 5, key="prep_train_percent")
        prep_random = st.number_input("Random state", 0, 999, 42, key="prep_random_state")
        st.write(f"Testing percentage: **{100 - prep_train}%**")
        selected_prep = st.multiselect("Features", list(features.columns), default=list(features.columns), key="prep_features")
        prep_scale = st.checkbox("Standardize features", True, key="prep_standardize")
        execute = st.button("Execute preprocessing", type="primary", use_container_width=True)
        st.write(f"Training records: **{int(len(features) * prep_train / 100)}**")
        st.write(f"Testing records: **{len(features) - int(len(features) * prep_train / 100)}**")
        st.markdown("</div>", unsafe_allow_html=True)
    train_count = int(len(features) * prep_train / 100)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Preprocessing results")
    result_columns = st.columns(4)
    result_columns[0].metric("Training samples", train_count)
    result_columns[1].metric("Testing samples", len(features) - train_count)
    result_columns[2].metric("Selected features", len(selected_prep))
    result_columns[3].metric("Scaling", "Enabled" if prep_scale else "Disabled")
    if execute:
        st.success(f"Preprocessing executed for {len(selected_prep)} selected features using a {prep_train}:{100 - prep_train} split.")
    st.markdown('<div class="status-line"><span class="status-dot"></span>Scaler leakage prevention: fitted only on training data.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.subheader("Data preview and visualizations")
    query = st.text_input("Search feature columns", placeholder="For example: bmi")
    selected = [column for column in features.columns if query.lower() in column.lower()] if query else list(features.columns)
    preview, statistics = st.columns([1.5, 1], gap="large")
    with preview:
        chart_card("Data preview")
        st.dataframe(features[selected].assign(target=target), use_container_width=True, height=280)
        close_card()
    with statistics:
        chart_card("Statistical summary")
        st.dataframe(features.describe().T.round(4), use_container_width=True, height=280)
        close_card()
    charts_left, charts_right = st.columns(2, gap="large")
    with charts_left:
        chart_card("Target distribution")
        figure, axis = plt.subplots(figsize=(7, 3.1))
        axis.hist(target, bins=20, color="#2563eb", alpha=.82)
        axis.set_xlabel("Disease progression"); axis.set_ylabel("Records")
        st.pyplot(figure, use_container_width=True)
        close_card()
    with charts_right:
        chart_card("Feature distribution")
        feature_to_plot = selected[0] if selected else features.columns[0]
        figure, axis = plt.subplots(figsize=(7, 3.1))
        axis.hist(features[feature_to_plot], bins=20, color="#22a06b", alpha=.82)
        axis.set_xlabel(feature_to_plot); axis.set_ylabel("Records")
        st.pyplot(figure, use_container_width=True)
        close_card()
    charts_left, charts_right = st.columns(2, gap="large")
    with charts_left:
        chart_card("Correlation heatmap")
        figure, axis = plt.subplots(figsize=(7, 4.5))
        sns.heatmap(pd.concat([features, target], axis=1).corr(), cmap="Blues", center=0, ax=axis)
        st.pyplot(figure, use_container_width=True)
        close_card()
    with charts_right:
        chart_card("Feature versus target")
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.scatter(features[feature_to_plot], target, color="#2563eb", alpha=.65, s=18)
        axis.set_xlabel(feature_to_plot); axis.set_ylabel("Target")
        st.pyplot(figure, use_container_width=True)
        close_card()
    if st.button("Continue to Simulation", type="primary"):
        navigate("Simulation")


def equation_text(feature_names, coefficients, intercept):
    parts = [f"{intercept:.3f}"]
    for name, coefficient in zip(feature_names, coefficients):
        sign = "+" if coefficient >= 0 else "-"
        parts.append(f" {sign} {abs(coefficient):.3f}×{name}")
    return "ŷ = " + "".join(parts)


def page_simulation():
    dataset, features, target = load_data()
    page_header("Simulation", "Train a real model, enter a prediction, and interpret the measured relationship.")
    controls, dataset_card = st.columns([1, 1.25], gap="large")
    with controls:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Model configuration")
        st.caption("Select predictors and configure the training process.")
        selected_features = st.multiselect("Independent features", list(features.columns), default=["bmi", "s5"], key="sim_features")
        train_percent = st.slider("Training percentage", 60, 90, 80, 5, key="sim_train_percent")
        random_state = st.number_input("Random state", 0, 999, 42, key="sim_random_state")
        standardize = st.checkbox("Standardize features", True, key="sim_standardize")
        fit_intercept = st.checkbox("Fit intercept", True, key="sim_intercept")
        st.markdown("</div>", unsafe_allow_html=True)
    with dataset_card:
        card("Dataset selection", "<p><b>Diabetes dataset</b><br>442 records · 10 numerical predictors · continuous target</p>")
        st.dataframe(features[selected_features or list(features.columns)[:3]].head(), hide_index=True, use_container_width=True)
    columns = st.columns(3)
    columns[0].metric("Dataset", "Diabetes")
    columns[1].metric("Selected features", len(selected_features))
    columns[2].metric("Target", "disease progression")
    actions = st.columns(4)
    if actions[0].button("Train Model", type="primary"):
        if not selected_features:
            st.error("Select at least one independent feature before training.")
        else:
            with st.spinner("Training Linear Regression..."):
                result = train_model(features[selected_features], target, train_percent / 100, random_state, standardize, fit_intercept)
            st.session_state.regression_result = result
            st.session_state.regression_features = selected_features
            st.session_state.regression_config = {
                "train_percent": train_percent, "random_state": random_state,
                "standardize": standardize, "fit_intercept": fit_intercept,
            }
            st.session_state.regression_trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("Model trained successfully.")
    if actions[3].button("Reset Simulation"):
        for key in ["regression_result", "regression_features", "regression_config", "regression_trained_at", "show_regression_evaluation"]:
            st.session_state.pop(key, None)
        st.rerun()
    result = st.session_state.get("regression_result")
    if result is None:
        st.info("Choose features in the sidebar and click **Train Model** to start.")
        return
    x_train, x_test, y_train, y_test, scaler, model, predictions, metrics = result
    selected_features = st.session_state.regression_features
    st.caption(f"Last trained: {st.session_state.regression_trained_at}")
    st.subheader("Model performance")
    metric_columns = st.columns(4)
    for column, (name, value) in zip(metric_columns, metrics.items()):
        column.metric(name, f"{value:.4f}")
    equation_columns = st.columns([1.5, 1], gap="large")
    with equation_columns[0]:
        chart_card("Regression equation")
        st.code(equation_text(selected_features, model.coef_, model.intercept_))
        close_card()
    with equation_columns[1]:
        chart_card("Training status")
        st.markdown('<div class="status-line"><span class="status-dot"></span>Model trained successfully</div>', unsafe_allow_html=True)
        st.write(f"Training samples: **{len(x_train)}**")
        st.write(f"Testing samples: **{len(x_test)}**")
        close_card()
    coefficient_frame = pd.DataFrame({"Feature": selected_features, "Coefficient": model.coef_})
    coefficient_frame["Magnitude"] = coefficient_frame["Coefficient"].abs()
    left, right = st.columns(2)
    with left:
        st.subheader("Feature coefficients")
        st.dataframe(coefficient_frame.sort_values("Magnitude", ascending=False), hide_index=True, use_container_width=True)
        st.caption("Coefficient magnitude is directly comparable when standardization is enabled; otherwise features retain their original units.")
    with right:
        st.subheader("Coefficient visualization")
        figure, axis = plt.subplots(figsize=(6, 3.8))
        ordered = coefficient_frame.sort_values("Coefficient")
        axis.barh(ordered["Feature"], ordered["Coefficient"], color=["#ef4444" if value < 0 else "#4f46e5" for value in ordered["Coefficient"]])
        axis.set_xlabel("Coefficient"); axis.axvline(0, color="#667085", linewidth=.8)
        st.pyplot(figure, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Actual vs predicted")
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.scatter(y_test, predictions, color="#4f46e5", alpha=.75)
        low, high = min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())
        axis.plot([low, high], [low, high], "--", color="#ef4444")
        axis.set_xlabel("Actual"); axis.set_ylabel("Predicted")
        st.pyplot(figure, use_container_width=True)
    with right:
        st.subheader("Residual plot")
        residuals = y_test - predictions
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.scatter(predictions, residuals, color="#5b50d6", alpha=.75)
        axis.axhline(0, color="#ef4444", linestyle="--")
        axis.set_xlabel("Predicted"); axis.set_ylabel("Residual")
        st.pyplot(figure, use_container_width=True)
    st.subheader("Prediction")
    record_index = st.number_input("Test record index", 0, len(x_test) - 1, 0, 1, key="regression_record_index")
    values = {}
    input_columns = st.columns(min(len(selected_features), 4))
    for index, feature in enumerate(selected_features):
        values[feature] = input_columns[index % len(input_columns)].number_input(feature, value=float(x_test.iloc[record_index][feature]), format="%.4f", key=f"prediction_{feature}")
    if actions[1].button("Run Prediction"):
        input_frame = pd.DataFrame([values], columns=selected_features)
        transformed = scaler.transform(input_frame) if scaler else input_frame
        prediction = float(model.predict(transformed)[0])
        actual = float(y_test.iloc[record_index])
        st.session_state.latest_prediction = (prediction, actual)
    if "latest_prediction" in st.session_state:
        prediction, actual = st.session_state.latest_prediction
        p1, p2, p3 = st.columns(3)
        p1.metric("Predicted value", f"{prediction:.2f}")
        p2.metric("Actual value", f"{actual:.2f}")
        p3.metric("Prediction error", f"{prediction - actual:+.2f}")
    if actions[2].button("Evaluate Model"):
        st.session_state.show_regression_evaluation = True
    if st.session_state.get("show_regression_evaluation"):
        st.subheader("Save this actual experiment")
        if st.button("Save result to experiment history"):
            save_history({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": "Diabetes",
                "features": ", ".join(selected_features),
                "target": "target",
                "split": f"{st.session_state.regression_config['train_percent']}:{100 - st.session_state.regression_config['train_percent']}",
                "model": "LinearRegression",
                **{name: round(value, 6) for name, value in metrics.items()},
            })
            st.success("Experiment result saved.")


QUESTIONS = [
    ("What does Linear Regression predict?", ["A continuous value", "Only a class label", "A cluster", "An image"], 0, "Regression predicts numerical continuous outcomes."),
    ("What is the dependent variable?", ["The outcome being predicted", "A random state", "A feature name", "A plot"], 0, "The dependent variable is y, the target."),
    ("What is an independent variable?", ["An input predictor", "The model error", "The intercept only", "The test score"], 0, "Independent variables are model inputs."),
    ("What does the slope represent?", ["Feature effect on prediction", "Number of rows", "Missing values", "The target name"], 0, "A coefficient describes how the prediction changes with a feature."),
    ("What is the intercept?", ["Predicted value when inputs are zero", "Maximum error", "Number of features", "Dataset size"], 0, "The intercept is the baseline term in the equation."),
    ("Simple Linear Regression uses how many predictors?", ["One", "Two", "Ten only", "None"], 0, "Simple regression has one independent variable."),
    ("Multiple Linear Regression uses?", ["Several predictors", "No target", "Only text", "Only one row"], 0, "Multiple regression includes two or more predictors."),
    ("What does OLS minimize?", ["Sum of squared residuals", "Number of columns", "Accuracy", "Class imbalance"], 0, "Ordinary Least Squares minimizes squared errors."),
    ("What is a residual?", ["Actual minus predicted", "A feature name", "The intercept", "A duplicate row"], 0, "Residual = y - ŷ."),
    ("What does MSE measure?", ["Average squared error", "Average feature", "Target class", "Training percentage"], 0, "MSE squares residuals before averaging."),
    ("What does MAE measure?", ["Average absolute error", "Coefficient sign", "Data type", "Model count"], 0, "MAE averages absolute residual magnitudes."),
    ("Why use RMSE?", ["It expresses error in target units", "It removes all errors", "It selects features", "It creates labels"], 0, "The square root returns the error to the target scale."),
    ("What does R² describe?", ["Explained target variation", "Number of residuals", "Feature units", "Training time"], 0, "R² compares explained variation to total variation."),
    ("What is overfitting?", ["Learning noise too closely", "Using no features", "A low dataset size", "A missing target"], 0, "An overfit model performs well on training data but poorly on unseen data."),
    ("Why reserve test data?", ["To estimate unseen performance", "To increase duplicates", "To change the target", "To remove coefficients"], 0, "The held-out test set gives a fairer evaluation."),
]


def page_exercise():
    page_header("Exercise", "Check your understanding with a guided 15-question regression quiz.")
    if "linear_quiz_answers" not in st.session_state:
        st.session_state.linear_quiz_answers = {}
    if "linear_quiz_index" not in st.session_state:
        st.session_state.linear_quiz_index = 0
    if "linear_quiz_submitted" not in st.session_state:
        st.session_state.linear_quiz_submitted = False
    index = st.session_state.linear_quiz_index
    question, options, correct, explanation = QUESTIONS[index]
    st.progress((index + 1) / len(QUESTIONS), text=f"Question {index + 1} of {len(QUESTIONS)}")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"#### {index + 1}. {question}")
    answer = st.radio("Choose one answer", options, index=None, key=f"linear_quiz_{index}")
    if answer is not None:
        st.session_state.linear_quiz_answers[index] = answer
    st.markdown("</div>", unsafe_allow_html=True)
    previous, next_column, submit = st.columns([1, 1, 2])
    with previous:
        if st.button("← Previous", disabled=index == 0):
            st.session_state.linear_quiz_index -= 1
            st.rerun()
    with next_column:
        if st.button("Next →", disabled=index == len(QUESTIONS) - 1):
            st.session_state.linear_quiz_index += 1
            st.rerun()
    with submit:
        if st.button("Submit Quiz", type="primary"):
            st.session_state.linear_quiz_submitted = True
    if st.session_state.linear_quiz_submitted:
        score = sum(st.session_state.linear_quiz_answers.get(i) == item[1][item[2]] for i, item in enumerate(QUESTIONS))
        st.success(f"Score: {score}/{len(QUESTIONS)} ({score / len(QUESTIONS):.0%})")
        with st.expander("Review incorrect answers"):
            for i, (_, answer_options, answer_index, answer_explanation) in enumerate(QUESTIONS):
                if st.session_state.linear_quiz_answers.get(i) != answer_options[answer_index]:
                    st.warning(f"Q{i + 1}: **{answer_options[answer_index]}** — {answer_explanation}")


def page_references():
    page_header("References", "Official documentation and learning resources for this experiment.")
    references = [
        ("Scikit-learn LinearRegression", "Ordinary Least Squares regression API and parameters.", "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"),
        ("Scikit-learn Diabetes dataset", "Dataset description, feature names, and loading API.", "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html"),
        ("NumPy documentation", "Numerical arrays and mathematical operations.", "https://numpy.org/doc/stable/"),
        ("Pandas documentation", "DataFrame inspection and data preparation.", "https://pandas.pydata.org/docs/"),
        ("Matplotlib documentation", "Scientific plotting and chart customization.", "https://matplotlib.org/stable/"),
        ("Seaborn documentation", "Statistical visualizations including heatmaps.", "https://seaborn.pydata.org/"),
    ]
    for title, description, url in references:
        card(title, f"<p>{description}</p><a href='{url}' target='_blank'>Open documentation →</a>")


def page_history():
    page_header("Experiment History", "Review and compare saved Linear Regression executions.")
    records = history_records()
    if not records:
        st.info("No saved experiments yet. Train and evaluate a model in Simulation, then save the result.")
        return
    frame = pd.DataFrame(records)
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.subheader("Manage saved runs")
    selected = st.selectbox("Select a record", range(len(records)), format_func=lambda i: f"Run {i + 1} · {records[i].get('date', 'Unknown date')}")
    delete, clear = st.columns(2)
    with delete:
        if st.button("Delete selected run"):
            records.pop(selected)
            HISTORY_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")
            st.success("Selected run deleted.")
            st.rerun()
    with clear:
        confirm = st.checkbox("Confirm clear all history", key="confirm_linear_clear")
        if st.button("Clear all history", disabled=not confirm):
            HISTORY_FILE.unlink(missing_ok=True)
            st.success("History cleared.")
            st.rerun()


def main():
    style_app()
    pages = ["Aim", "Theory", "Procedure", "Data Preparation", "Simulation", "Exercise", "References", "Experiment History"]
    labels = {page: page for page in pages}
    if "page" not in st.session_state:
        st.session_state.page = "Aim"
    with st.sidebar:
        st.markdown('<div class="sidebar-brand"><strong>Linear Regression</strong></div>', unsafe_allow_html=True)
        st.caption("Navigation")
        displayed = [labels[page] for page in pages]
        selected = st.radio("Navigate", displayed, index=pages.index(st.session_state.page))
        st.session_state.page = pages[displayed.index(selected)]
        st.divider()
        st.caption("Machine Learning Virtual Lab")
    {
        "Aim": page_aim, "Theory": page_theory, "Procedure": page_procedure,
        "Data Preparation": page_data, "Simulation": page_simulation,
        "Exercise": page_exercise, "References": page_references,
        "Experiment History": page_history,
    }[st.session_state.page]()


if __name__ == "__main__":
    main()
