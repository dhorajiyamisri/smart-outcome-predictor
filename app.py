import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Outcome Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 35px;
    border-radius: 22px;
    background: linear-gradient(135deg, #07111f, #123456);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    color: #dbeafe;
}

.card {
    padding: 22px;
    border-radius: 18px;
    background: white;
    border: 1px solid #e5e7eb;
    margin-bottom: 18px;
}

.metric-card {
    padding: 22px;
    border-radius: 18px;
    background: white;
    border: 1px solid #e5e7eb;
    text-align: center;
}

.result-card {
    padding: 28px;
    border-radius: 22px;
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    color: white;
    text-align: center;
}

.small-text {
    color: #64748b;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODELS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSIFICATION_MODEL_PATH = os.path.join(
    BASE_DIR, "classification_model.pkl"
)

REGRESSION_MODEL_PATH = os.path.join(
    BASE_DIR, "regression_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR, "preprocessor.pkl"
)


@st.cache_resource
def load_models():

    classification_model = joblib.load(
        CLASSIFICATION_MODEL_PATH
    )

    regression_model = joblib.load(
        REGRESSION_MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    return (
        classification_model,
        regression_model,
        preprocessor
    )


# =========================================================
# CHECK MODEL FILES
# =========================================================

required_files = [
    CLASSIFICATION_MODEL_PATH,
    REGRESSION_MODEL_PATH,
    PREPROCESSOR_PATH
]

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]

if missing_files:

    st.error("❌ Required ML files are missing.")

    for file in missing_files:
        st.write("Missing:", os.path.basename(file))

    st.stop()


classification_model, regression_model, preprocessor = load_models()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    <h2>🎓 Smart Outcome</h2>
    <p style="color:#64748b;">
    ML-powered learner outcome prediction
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🔮 Predict Outcome",
        "📊 Model Performance",
        "🧪 What-If Simulator",
        "🔍 Dataset Explorer",
        "🧠 ML Pipeline",
        "ℹ️ About Project"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Built with Python • Scikit-learn • Streamlit"
)


# =========================================================
# OVERVIEW
# =========================================================

if page == "🏠 Overview":

    st.markdown("""
    <div class="hero">

    <h1>🎓 Smart Outcome Predictor</h1>

    <p>
    An interactive Machine Learning application designed to
    predict learner completion status and estimate final score.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.subheader("📌 Project Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🤖 ML Models", "2")

    with c2:
        st.metric("🎯 Prediction Type", "Classification")

    with c3:
        st.metric("📈 Score Prediction", "Regression")

    with c4:
        st.metric("⚡ Application", "Interactive ML")

    st.divider()

    st.subheader("🚀 What This App Does")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="card">

        <h3>🔮 Outcome Prediction</h3>

        <p>
        Enter learner profile and learning activity information
        to generate ML predictions.
        </p>

        <ul>
        <li>Completion Status</li>
        <li>Predicted Final Score</li>
        <li>Learner Outcome</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">

        <h3>🧪 What-If Analysis</h3>

        <p>
        Change learner activity variables and observe how the
        model prediction changes.
        </p>

        <ul>
        <li>Activity simulation</li>
        <li>Score comparison</li>
        <li>Outcome comparison</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

    st.subheader("🧠 Machine Learning Workflow")

    st.info(
        "Dataset → Preprocessing → Feature Engineering → "
        "Model Training → Prediction → Evaluation → "
        "Interactive Application"
    )


# =========================================================
# PREDICT OUTCOME
# =========================================================

elif page == "🔮 Predict Outcome":

    st.title("🔮 Predict Learner Outcome")

    st.write(
        "Enter learner information and learning activity "
        "to generate predictions using the trained ML models."
    )

    st.divider()

    # -----------------------------------------------------
    # LEARNER PROFILE
    # -----------------------------------------------------

    st.subheader("👤 Learner Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=10,
            max_value=100,
            value=25
        )

    with col2:
        country_region = st.text_input(
            "Country / Region",
            value="India"
        )

    with col3:
        device_type = st.text_input(
            "Device Type",
            value="Laptop"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        education_background = st.text_input(
            "Education Background",
            value="Bachelor"
        )

    with col2:
        course_level = st.text_input(
            "Course Level",
            value="Beginner"
        )

    with col3:
        course_category = st.text_input(
            "Course Category",
            value="Data Science"
        )

    st.divider()

    # -----------------------------------------------------
    # LEARNING ACTIVITY
    # -----------------------------------------------------

    st.subheader("📚 Learning Activity")

    col1, col2, col3 = st.columns(3)

    with col1:
        week_of_year = st.number_input(
            "Week of Year",
            min_value=1,
            max_value=53,
            value=20
        )

    with col2:
        sessions = st.number_input(
            "Sessions",
            min_value=0,
            value=10
        )

    with col3:
        time_spent_hours = st.number_input(
            "Time Spent (Hours)",
            min_value=0.0,
            value=20.0,
            step=0.5
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        videos_watched = st.number_input(
            "Videos Watched",
            min_value=0,
            value=15
        )

    with col2:
        quiz_attempts = st.number_input(
            "Quiz Attempts",
            min_value=0,
            value=5
        )

    with col3:
        assignments_submitted = st.number_input(
            "Assignments Submitted",
            min_value=0,
            value=5
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        forum_posts = st.number_input(
            "Forum Posts",
            min_value=0,
            value=3
        )

    with col2:
        avg_quiz_score = st.slider(
            "Average Quiz Score",
            0.0,
            100.0,
            70.0
        )

    with col3:
        attendance_rate = st.slider(
            "Attendance Rate",
            0.0,
            100.0,
            75.0
        )

    st.divider()

    # -----------------------------------------------------
    # INPUT DATAFRAME
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "age": [age],

        "country_region": [country_region],

        "device_type": [device_type],

        "education_background": [
            education_background
        ],

        "course_level": [
            course_level
        ],

        "course_category": [
            course_category
        ],

        "week_of_year": [
            week_of_year
        ],

        "sessions": [
            sessions
        ],

        "time_spent_hours": [
            time_spent_hours
        ],

        "videos_watched": [
            videos_watched
        ],

        "quiz_attempts": [
            quiz_attempts
        ],

        "assignments_submitted": [
            assignments_submitted
        ],

        "forum_posts": [
            forum_posts
        ],

        "avg_quiz_score": [
            avg_quiz_score
        ],

        "attendance_rate": [
            attendance_rate
        ]
    })

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if st.button(
        "🚀 Predict Learner Outcome",
        width="stretch"
    ):

        try:

            # =============================================
            # CLASSIFICATION
            # =============================================

            transformed_input = preprocessor.transform(
                input_data
            )

            completion_prediction = (
                classification_model.predict(
                    transformed_input
                )[0]
            )

            # =============================================
            # CLASSIFICATION PROBABILITY
            # =============================================

            completion_probability = None

            if hasattr(
                classification_model,
                "predict_proba"
            ):

                probabilities = (
                    classification_model.predict_proba(
                        transformed_input
                    )[0]
                )

                completion_probability = (
                    float(np.max(probabilities)) * 100
                )

            # =============================================
            # REGRESSION
            # =============================================

            score_prediction = (
                regression_model.predict(
                    input_data
                )[0]
            )

            score_prediction = float(
                np.clip(
                    score_prediction,
                    0,
                    100
                )
            )

            # =============================================
            # RESULT
            # =============================================

            st.success(
                "✅ Prediction generated successfully!"
            )

            st.divider()

            r1, r2, r3 = st.columns(3)

            with r1:

                if completion_prediction == 1:

                    st.markdown("""
                    <div class="result-card">

                    <h2>✅ Completed</h2>

                    <p>Predicted Completion Status</p>

                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.markdown("""
                    <div class="result-card">

                    <h2>⚠️ Not Completed</h2>

                    <p>Predicted Completion Status</p>

                    </div>
                    """, unsafe_allow_html=True)

            with r2:

                st.markdown(
                    f"""
                    <div class="result-card">

                    <h2>{score_prediction:.1f}</h2>

                    <p>Predicted Final Score</p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with r3:

                if completion_probability is not None:

                    st.markdown(
                        f"""
                        <div class="result-card">

                        <h2>
                        {completion_probability:.1f}%
                        </h2>

                        <p>Model Confidence</p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown("""
                    <div class="result-card">

                    <h2>ML</h2>

                    <p>Prediction Generated</p>

                    </div>
                    """, unsafe_allow_html=True)

            st.divider()

            st.subheader("📋 Prediction Input")

            st.dataframe(
                input_data,
                width="stretch"
            )

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Evaluation results obtained during model development."
    )

    st.divider()

    st.subheader("🎯 Classification")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Accuracy", "72.19%")

    with c2:
        st.metric("Precision", "71.18%")

    with c3:
        st.metric("Recall", "71.92%")

    with c4:
        st.metric("F1 Score", "71.02%")

    with c5:
        st.metric("ROC-AUC", "77.09%")

    st.info(
        "Final classification model: Soft Voting Classifier"
    )

    st.divider()

    st.subheader("📈 Regression")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("MAE", "7.7646")

    with r2:
        st.metric("RMSE", "9.6907")

    with r3:
        st.metric("R²", "0.4974")

    st.info(
        "Final regression model: Stacking Regressor"
    )

    st.divider()

    st.subheader("🤖 Ensemble Models Explored")

    models = pd.DataFrame({
        "Model": [
            "Bagging",
            "AdaBoost",
            "Gradient Boosting",
            "LightGBM",
            "XGBoost",
            "Voting",
            "Stacking"
        ],
        "Type": [
            "Ensemble",
            "Boosting",
            "Boosting",
            "Boosting",
            "Boosting",
            "Ensemble",
            "Ensemble"
        ]
    })

    st.dataframe(
        models,
        width="stretch"
    )


# =========================================================
# WHAT-IF SIMULATOR
# =========================================================

elif page == "🧪 What-If Simulator":

    st.title("🧪 What-If Outcome Simulator")

    st.write(
        "Experiment with learner activity and observe "
        "how the trained regression model responds."
    )

    st.divider()

    st.subheader("🎛️ Change Learning Activity")

    col1, col2 = st.columns(2)

    with col1:

        sim_sessions = st.slider(
            "Sessions",
            0,
            100,
            10
        )

        sim_time = st.slider(
            "Time Spent Hours",
            0.0,
            200.0,
            20.0,
            step=1.0
        )

        sim_videos = st.slider(
            "Videos Watched",
            0,
            100,
            15
        )

        sim_quiz = st.slider(
            "Quiz Attempts",
            0,
            50,
            5
        )

    with col2:

        sim_assignments = st.slider(
            "Assignments Submitted",
            0,
            50,
            5
        )

        sim_forum = st.slider(
            "Forum Posts",
            0,
            50,
            3
        )

        sim_quiz_score = st.slider(
            "Average Quiz Score",
            0.0,
            100.0,
            70.0
        )

        sim_attendance = st.slider(
            "Attendance Rate",
            0.0,
            100.0,
            75.0
        )

    simulation_data = pd.DataFrame({

        "age": [25],

        "country_region": ["India"],

        "device_type": ["Laptop"],

        "education_background": ["Bachelor"],

        "course_level": ["Beginner"],

        "course_category": ["Data Science"],

        "week_of_year": [20],

        "sessions": [sim_sessions],

        "time_spent_hours": [sim_time],

        "videos_watched": [sim_videos],

        "quiz_attempts": [sim_quiz],

        "assignments_submitted": [
            sim_assignments
        ],

        "forum_posts": [
            sim_forum
        ],

        "avg_quiz_score": [
            sim_quiz_score
        ],

        "attendance_rate": [
            sim_attendance
        ]
    })

    if st.button(
        "🔍 Run What-If Simulation",
        width="stretch"
    ):

        try:

            simulated_score = (
                regression_model.predict(
                    simulation_data
                )[0]
            )

            simulated_score = float(
                np.clip(
                    simulated_score,
                    0,
                    100
                )
            )

            st.success(
                "Simulation completed successfully!"
            )

            st.metric(
                "🎯 Simulated Final Score",
                f"{simulated_score:.1f}"
            )

            st.dataframe(
                simulation_data,
                width="stretch"
            )

        except Exception as e:

            st.error(
                "Simulation failed."
            )

            st.exception(e)


# =========================================================
# DATASET EXPLORER
# =========================================================

elif page == "🔍 Dataset Explorer":

    st.title("🔍 Dataset Explorer")

    st.write(
        "Explore the dataset used for the Machine Learning project."
    )

    # Search CSV files automatically

    csv_files = glob.glob(
        os.path.join(BASE_DIR, "**", "*.csv"),
        recursive=True
    )

    if not csv_files:

        st.warning(
            "No CSV dataset found inside project folder."
        )

    else:

        selected_file = st.selectbox(
            "Select Dataset",
            csv_files,
            format_func=lambda x: os.path.basename(x)
        )

        try:

            df = pd.read_csv(
                selected_file
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Rows",
                    df.shape[0]
                )

            with c2:
                st.metric(
                    "Columns",
                    df.shape[1]
                )

            with c3:
                st.metric(
                    "Missing Values",
                    int(df.isnull().sum().sum())
                )

            st.divider()

            tab1, tab2, tab3 = st.tabs(
                [
                    "📋 Preview",
                    "📊 Statistics",
                    "❓ Missing Values"
                ]
            )

            with tab1:

                st.dataframe(
                    df.head(20),
                    width="stretch"
                )

            with tab2:

                st.dataframe(
                    df.describe(include="all").T,
                    width="stretch"
                )

            with tab3:

                missing = (
                    df.isnull()
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                missing = missing[
                    missing > 0
                ]

                if len(missing) == 0:

                    st.success(
                        "No missing values found."
                    )

                else:

                    st.dataframe(
                        missing,
                        width="stretch"
                    )

        except Exception as e:

            st.error(
                "Unable to load dataset."
            )

            st.exception(e)


# =========================================================
# ML PIPELINE
# =========================================================

elif page == "🧠 ML Pipeline":

    st.title("🧠 Machine Learning Pipeline")

    st.write(
        "The application uses the trained preprocessing "
        "and ensemble models from the notebook."
    )

    st.divider()

    st.markdown("""
    ### 📥 Input Features

    ```text
    age
    country_region
    device_type
    education_background
    course_level
    course_category
    week_of_year
    sessions
    time_spent_hours
    videos_watched
    quiz_attempts
    assignments_submitted
    forum_posts
    avg_quiz_score
    attendance_rate
    ```

    ### ⚙️ Preprocessing

    ```text
    Numerical Features
          ↓
    Mean Imputation
          ↓
    StandardScaler

    Categorical Features
          ↓
    Most-Frequent Imputation
          ↓
    OneHotEncoder
    ```

    ### 🤖 Classification

    ```text
    Learner Input
          ↓
    Preprocessor
          ↓
    Soft Voting Classifier
          ↓
    Completion Status
    ```

    ### 📈 Regression

    ```text
    Learner Input
          ↓
    Preprocessor
          ↓
    Stacking Regressor
          ↓
    Final Score
    ```

    ### 🔗 Regression Ensemble

    ```text
    Linear Regression
            +
    Decision Tree Regressor
            +
    SVR
            ↓
    Linear Regression
            ↓
    Final Score
    ```
    """)


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Smart Outcome Predictor")

    st.markdown("""
    <div class="card">

    <h2>🎓 Smart Outcome Predictor</h2>

    <p>
    Smart Outcome Predictor is a Machine Learning project
    designed to analyze learner information and learning
    activity to estimate educational outcomes.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.subheader("🎯 Project Objective")

    st.write(
        "The project combines classification and regression "
        "to provide two complementary learner outcome predictions."
    )

    st.subheader("🤖 Machine Learning")

    st.write(
        "Classification is used for completion status, while "
        "regression is used to estimate final score."
    )

    st.subheader("🛠️ Technology Stack")

    st.markdown("""
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Joblib
    - Streamlit
    - Jupyter Notebook
    """)

    st.subheader("📌 Application Architecture")

    st.code("""
    Dataset
       ↓
    Data Preprocessing
       ↓
    Machine Learning
       ↓
    Trained Models
       ↓
    Joblib
       ↓
    Streamlit Application
       ↓
    Interactive Prediction
    """)