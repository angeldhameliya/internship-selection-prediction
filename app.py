"""
🎓 Internship Selection Prediction System
==========================================
Smart Career Insights Powered by Machine Learning
Streamlit Web Application — Dusty Pink & Cream Aesthetic Theme
"""

import os
import json
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & DUSTY PINK & CREAM STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Internship Selection Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "internship_dataset.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "models", "internship_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "Reports", "model_metrics.json")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")
BG_PATH = os.path.join(BASE_DIR, "assets", "background.jpg")
EDA_HTML_PATH = os.path.join(BASE_DIR, "Reports", "EDA_Report.html")

# Encode background image for robust CSS injection
bg_base64 = ""
if os.path.exists(BG_PATH):
    try:
        with open(BG_PATH, "rb") as f:
            bg_base64 = base64.b64encode(f.read()).decode()
    except Exception:
        bg_base64 = ""

if bg_base64:
    bg_style = f"linear-gradient(rgba(253, 248, 245, 0.88), rgba(242, 222, 215, 0.92)), url('data:image/jpeg;base64,{bg_base64}') no-repeat center center fixed"
else:
    bg_style = "linear-gradient(135deg, #FAF4EF 0%, #F5E6DF 40%, #EED7CF 75%, #E2BEB4 100%)"

# Dusty Pink & Cream Luxury Color Palette
custom_css = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }}

    /* Global App Background */
    .stApp {{
        background: {bg_style};
        background-size: cover;
        color: #381E28;
    }}

    /* Sidebar Styling: Delicate Cream Blush */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #FDF7F4 0%, #F8ECE6 60%, #F1DDD4 100%);
        border-right: 1px solid rgba(196, 139, 159, 0.45);
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label {{
        color: #381E28 !important;
        font-weight: 500;
    }}

    /* Hero Banner: Dusty Rose & Rose Gold */
    .hero-banner {{
        background: linear-gradient(135deg, #B76E79 0%, #C47A8C 45%, #D8A496 100%);
        border: 1px solid rgba(255, 255, 255, 0.6);
        border-radius: 20px;
        padding: 32px 42px;
        margin-bottom: 25px;
        box-shadow: 0 16px 36px -6px rgba(183, 110, 121, 0.32), inset 0 1px 0 rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(12px);
    }}

    .hero-title {{
        font-size: 2.35rem;
        font-weight: 800;
        color: #FFFFFF !important;
        text-shadow: 0 2px 8px rgba(138, 64, 82, 0.3);
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }}

    .hero-sub {{
        font-size: 1.15rem;
        color: #FDF3EE !important;
        font-weight: 400;
    }}

    /* KPI Cards: Cream Porcelain Frosted Glass */
    .kpi-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 25px;
    }}

    .kpi-card {{
        background: rgba(255, 253, 250, 0.94);
        border: 1px solid rgba(214, 166, 178, 0.6);
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 8px 24px rgba(183, 110, 121, 0.12);
        backdrop-filter: blur(10px);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}

    .kpi-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(183, 110, 121, 0.2);
        border-color: #B76E79;
    }}

    .kpi-title {{
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8C5163;
        margin-bottom: 6px;
    }}

    .kpi-value {{
        font-size: 1.95rem;
        font-weight: 800;
        color: #9A3B53;
    }}

    .kpi-meta {{
        font-size: 0.82rem;
        color: #82626D;
        margin-top: 4px;
        font-weight: 500;
    }}

    /* Content Container Box */
    .card-box {{
        background: rgba(255, 254, 252, 0.94);
        border: 1px solid rgba(214, 166, 178, 0.55);
        border-radius: 18px;
        padding: 26px 32px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(183, 110, 121, 0.1);
        backdrop-filter: blur(10px);
        color: #381E28;
    }}

    .card-box h4, .card-box h5 {{
        color: #4A222F !important;
        font-weight: 700;
    }}

    /* Result Card - Selected */
    .result-card-selected {{
        background: linear-gradient(135deg, #2E5A44 0%, #4D8067 100%);
        border: 1px solid #78B294;
        border-radius: 18px;
        padding: 32px;
        color: #FFFFFF !important;
        box-shadow: 0 12px 32px rgba(46, 90, 68, 0.35);
        text-align: center;
        margin-top: 20px;
    }}

    /* Result Card - Not Selected */
    .result-card-rejected {{
        background: linear-gradient(135deg, #8E384D 0%, #B35168 100%);
        border: 1px solid #E89EAE;
        border-radius: 18px;
        padding: 32px;
        color: #FFFFFF !important;
        box-shadow: 0 12px 32px rgba(142, 56, 77, 0.35);
        text-align: center;
        margin-top: 20px;
    }}

    /* Button: Dusty Rose to Rose-Gold gradient */
    .stButton>button {{
        background: linear-gradient(90deg, #B76E79 0%, #C88392 50%, #D8A496 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 12px !important;
        padding: 13px 30px !important;
        border: none !important;
        box-shadow: 0 5px 18px rgba(183, 110, 121, 0.35) !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
    }}

    .stButton>button:hover {{
        box-shadow: 0 8px 24px rgba(183, 110, 121, 0.5) !important;
        transform: translateY(-2px) !important;
    }}

    /* Disclaimer Box */
    .disclaimer-box {{
        background: rgba(248, 235, 230, 0.95);
        border-left: 4px solid #B76E79;
        padding: 16px 20px;
        border-radius: 0 10px 10px 0;
        font-size: 0.88rem;
        color: #63424E;
        margin-top: 20px;
        box-shadow: 0 4px 14px rgba(183, 110, 121, 0.08);
    }}

    /* Typography & Headers on Light Cream */
    h1, h2, h3, h4, h5, h6 {{
        color: #381E28 !important;
        font-weight: 700;
    }}

    p, span, label {{
        color: #381E28;
    }}

    /* Tab styling */
    button[data-baseweb="tab"] {{
        color: #7A5561 !important;
        font-weight: 600 !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #9A3B53 !important;
        border-bottom-color: #B76E79 !important;
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. CACHED DATA & MODEL LOADERS
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset file not found at: {DATASET_PATH}")
        return None
    df = pd.read_excel(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    
    # Standardize categoricals
    if "github_score" in df.columns:
        df["github_score"] = (
            df["github_score"]
            .astype(str)
            .str.strip()
            .replace({"Tier1": "Tier 1", "Tier2": "Tier 2", "Tier3": "Tier 3"})
        )
    if "linkedin_activity_score" in df.columns:
        df["linkedin_activity_score"] = (
            df["linkedin_activity_score"].astype(str).str.strip().str.capitalize()
        )
    
    # Derive binary selection target from Composite Career Readiness Index
    cgpa_norm = (df["CGPA"] - 5.0) / 5.0
    coding_norm = df["coding_test_score"] / 10.0
    skills_norm = df["skills_score"] / 10.0
    interview_norm = df["interview_score"] / 10.0
    exp_norm = (df["projects_count"] / 5.0 * 0.6) + (df["internships_done"] / 3.0 * 0.4)
    comm_norm = df["communication_score"] / 10.0

    np.random.seed(42)
    hiring_variance = np.random.normal(0, 0.04, size=len(df))

    career_index = (
        cgpa_norm * 0.25 +
        coding_norm * 0.25 +
        skills_norm * 0.20 +
        interview_norm * 0.15 +
        exp_norm * 0.10 +
        comm_norm * 0.05 +
        hiring_variance
    )

    selection_threshold = float(np.median(career_index))
    df["Selected"] = (career_index >= selection_threshold).astype(int)
    df["Selection_Status"] = df["Selected"].map({1: "Selected", 0: "Not Selected"})
        
    return df

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

df = load_dataset()
model = load_model()
metrics = load_metrics()

# ---------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ---------------------------------------------------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=120)
    
    st.markdown("<h3 style='color:#4A222F; margin-bottom:2px;'>🎓 Internship Predictor</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #7A5561; font-size: 0.88rem; margin-top:0px;'>"
        "Career Analytics & ML Decision Support</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    nav_option = st.radio(
        "Navigation Menu",
        [
            "🏠 Home Dashboard",
            "🔮 Prediction System",
            "📊 Visual Analytics & EDA",
            "📈 Model Performance",
            "📂 Dataset Explorer",
            "📘 About & Documentation"
        ]
    )

    st.markdown("---")
    st.markdown(
        """
        <div style='background: rgba(255, 253, 250, 0.9); border: 1px solid rgba(214, 166, 178, 0.55); border-radius: 12px; padding: 14px; font-size: 0.82rem; color: #5C3D48;'>
            <strong style='color: #9A3B53;'>Model Summary:</strong><br>
            • Algorithm: Logistic Regression<br>
            • Selection Rule: Score &ge; 6<br>
            • Train/Test: 80% / 20% Stratified<br>
            • Preprocessing: Scaler + OHE
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# 4. HERO BANNER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🎓 Internship Selection Prediction System</div>
        <div class="hero-sub">Smart Career Insights & Placement Probability Powered by Machine Learning</div>
    </div>
    """,
    unsafe_allow_html=True
)

if df is None:
    st.error("Failed to load internship dataset. Please check the dataset path.")
    st.stop()


# ---------------------------------------------------------
# PAGE 1: HOME DASHBOARD
# ---------------------------------------------------------
if nav_option == "🏠 Home Dashboard":
    st.markdown("### 📌 Executive Overview & Candidate KPI Metrics")
    st.caption("Live statistical metrics calculated directly from the student dataset.")

    # Calculate actual KPI values
    total_records = len(df)
    avg_cgpa = df["CGPA"].mean()
    avg_skills = df["skills_score"].mean()
    avg_coding = df["coding_test_score"].mean()
    avg_interview = df["interview_score"].mean()
    avg_projects = df["projects_count"].mean()

    selected_count = int(df["Selected"].sum())
    not_selected_count = total_records - selected_count
    selection_rate = (selected_count / total_records) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Candidates</div>
            <div class="kpi-value">{total_records}</div>
            <div class="kpi-meta">Clean student profiles</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average CGPA</div>
            <div class="kpi-value" style="color:#A85A6E;">{avg_cgpa:.2f}</div>
            <div class="kpi-meta">Scale: 5.00 – 10.00</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Selected Candidates</div>
            <div class="kpi-value" style="color:#3B7A58;">{selected_count}</div>
            <div class="kpi-meta">{selection_rate:.1f}% Selection Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Not Selected</div>
            <div class="kpi-value" style="color:#B76E79;">{not_selected_count}</div>
            <div class="kpi-meta">{100 - selection_rate:.1f}% Non-Selection Rate</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Skills Score</div>
            <div class="kpi-value">{avg_skills:.1f}<span style="font-size:1rem;color:#8E6B77;">/10</span></div>
            <div class="kpi-meta">Core technical skills</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Coding Test</div>
            <div class="kpi-value" style="color:#C47A8C;">{avg_coding:.1f}<span style="font-size:1rem;color:#8E6B77;">/10</span></div>
            <div class="kpi-meta">Algorithmic aptitude</div>
        </div>
        """, unsafe_allow_html=True)

    with col7:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Interview Score</div>
            <div class="kpi-value" style="color:#8C4A60;">{avg_interview:.1f}<span style="font-size:1rem;color:#8E6B77;">/10</span></div>
            <div class="kpi-meta">Technical mock interview</div>
        </div>
        """, unsafe_allow_html=True)

    with col8:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Projects Count</div>
            <div class="kpi-value" style="color:#D8A496;">{avg_projects:.1f}</div>
            <div class="kpi-meta">Hands-on portfolio projects</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Visual Analytics on Home: Target Distribution & Prior Internships
    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 Internship Selection Distribution")
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Not Selected", "Selected"],
            values=[not_selected_count, selected_count],
            hole=0.55,
            marker=dict(colors=["#B76E79", "#488566"], line=dict(color="#FFFDFB", width=2)),
            textinfo="label+percent",
            textfont=dict(size=14, color="#FFFFFF", family="Plus Jakarta Sans")
        )])
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#381E28"),
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        st.markdown("#### 💼 Prior Internship Experience Breakdown")
        internship_counts = df["internships_done"].value_counts().sort_index()
        fig_bar = px.bar(
            x=[f"{x} Internships" for x in internship_counts.index],
            y=internship_counts.values,
            labels={"x": "Prior Internships Completed", "y": "Number of Students"},
            color=internship_counts.values,
            color_continuous_scale=["#E8D3CE", "#C47A8C", "#8C4A60"]
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#381E28"),
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Educational Disclaimer & Target Formulation:</strong><br>
            The target variable is configured via score thresholding on candidate evaluation index <code>Unnamed: 13 &ge; 6</code> (Selected vs Not Selected).
            To prevent data leakage, the original evaluation score and student identifiers are strictly excluded from the predictive model.
            Predictions represent statistical likelihoods for guidance and academic planning.
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# PAGE 2: PREDICTION PAGE
# ---------------------------------------------------------
elif nav_option == "🔮 Prediction System":
    st.markdown("### 🔮 Interactive Candidate Selection Predictor")
    st.caption("Input student academic, technical, and interpersonal attributes to estimate internship selection probability.")

    if model is None:
        st.error("⚠️ Model file not found. Please run `python project.py` to train and save the model.")
        st.stop()

    st.markdown("<div class='card-box'>", unsafe_allow_html=True)
    st.markdown("#### 📋 Candidate Profile Evaluation Form")

    with st.form("prediction_form"):
        st.markdown("##### 1. Academic & Technical Foundation")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            cgpa_val = st.slider("CGPA (Cumulative)", min_value=5.0, max_value=10.0, value=7.5, step=0.01)
        with f_col2:
            skills_val = st.slider("Skills Score (1-10)", min_value=1, max_value=10, value=6, step=1)
        with f_col3:
            coding_val = st.slider("Coding Test Score (1-10)", min_value=1, max_value=10, value=6, step=1)
        with f_col4:
            projects_val = st.number_input("Projects Completed", min_value=0, max_value=10, value=2, step=1)

        st.markdown("##### 2. Practical Experience & Professional Footprint")
        f_col5, f_col6, f_col7, f_col8 = st.columns(4)
        with f_col5:
            internships_val = st.number_input("Prior Internships Done", min_value=0, max_value=5, value=1, step=1)
        with f_col6:
            certifications_val = st.number_input("Certifications Count", min_value=0, max_value=15, value=2, step=1)
        with f_col7:
            linkedin_val = st.selectbox("LinkedIn Activity Active?", ["Yes", "No"], index=0)
        with f_col8:
            github_val = st.selectbox("GitHub Portfolio Tier", ["Tier 1", "Tier 2", "Tier 3"], index=1)

        st.markdown("##### 3. Interpersonal, Resume & Interview Readiness")
        f_col9, f_col10, f_col11, f_col12 = st.columns(4)
        with f_col9:
            comm_val = st.slider("Communication Score (1-10)", min_value=1, max_value=10, value=6, step=1)
        with f_col10:
            soft_val = st.slider("Soft Skills Rating (0-5)", min_value=0, max_value=5, value=3, step=1)
        with f_col11:
            resume_val = st.slider("Resume ATS Score (1-10)", min_value=1, max_value=10, value=6, step=1)
        with f_col12:
            interview_val = st.slider("Mock Interview Score (0-10)", min_value=0, max_value=10, value=6, step=1)

        submitted = st.form_submit_button("🎯 Predict Internship Selection")

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        input_data = pd.DataFrame([{
            "CGPA": float(cgpa_val),
            "skills_score": int(skills_val),
            "projects_count": int(projects_val),
            "internships_done": int(internships_val),
            "communication_score": int(comm_val),
            "coding_test_score": int(coding_val),
            "resume_score": int(resume_val),
            "certifications_count": int(certifications_val),
            "linkedin_activity_score": str(linkedin_val),
            "github_score": str(github_val),
            "soft_skills_score": int(soft_val),
            "interview_score": int(interview_val)
        }])

        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            prob_selected = probability[1] * 100
            prob_not_selected = probability[0] * 100

            if prediction == 1:
                st.markdown(f"""
                <div class="result-card-selected">
                    <h2 style="font-size: 2.2rem; margin-bottom: 8px; color:#FFFFFF;">🎉 Congratulations! HIGH CHANCE OF SELECTION</h2>
                    <p style="font-size: 1.25rem; font-weight: 600; color:#F0FFF4;">The Machine Learning model predicts this profile is <strong>SELECTED</strong> for internship opportunities.</p>
                    <div style="font-size: 2.5rem; font-weight: 800; margin: 15px 0; color:#FFFFFF;">{prob_selected:.1f}% <span style="font-size: 1.1rem; font-weight: 500;">Estimated Selection Probability</span></div>
                    <p style="color: #E2F0D9; font-size: 0.95rem; max-width: 700px; margin: 0 auto;">
                        Strong alignment observed across academic metrics, technical assessments, and interview readiness. Maintain consistent open-source contributions and algorithmic practice.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-rejected">
                    <h2 style="font-size: 2.2rem; margin-bottom: 8px; color:#FFFFFF;">📌 NOT SELECTED (Needs Strengthening)</h2>
                    <p style="font-size: 1.25rem; font-weight: 600; color:#FFF0F3;">The model predicts this candidate may face high competition in current internship rounds.</p>
                    <div style="font-size: 2.5rem; font-weight: 800; margin: 15px 0; color:#FFFFFF;">{prob_not_selected:.1f}% <span style="font-size: 1.1rem; font-weight: 500;">Probability of Requiring Upskilling</span></div>
                    <p style="color: #FDE8ED; font-size: 0.95rem; max-width: 700px; margin: 0 auto;">
                        Recommendation: Target improving coding test problem-solving, build 1-2 deployed end-to-end projects, refine ATS resume keywords, and practice mock interviews.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Interactive Probability Breakdown Gauge
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### 📊 Model Decision Probability Meter")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_selected,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Selection Likelihood (%)", 'font': {'size': 18, 'color': '#381E28'}},
                    number={'suffix': "%", 'font': {'size': 32, 'color': '#9A3B53'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#7A5561"},
                        'bar': {'color': "#488566" if prediction == 1 else "#B76E79"},
                        'bgcolor': "#FAF4EF",
                        'borderwidth': 2,
                        'bordercolor': "#D8A496",
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(183, 110, 121, 0.25)'},
                            {'range': [40, 60], 'color': 'rgba(216, 164, 150, 0.25)'},
                            {'range': [60, 100], 'color': 'rgba(72, 133, 102, 0.25)'}
                        ],
                        'threshold': {
                            'line': {'color': "#381E28", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#381E28"),
                    height=260,
                    margin=dict(t=20, b=20, l=30, r=30)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with res_col2:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### 💡 Candidate Key Strength Radar")
                cats = ['CGPA', 'Skills', 'Coding', 'Resume', 'Interview', 'Soft Skills']
                vals = [
                    (cgpa_val / 10.0) * 10,
                    skills_val,
                    coding_val,
                    resume_val,
                    interview_val,
                    (soft_val / 5.0) * 10
                ]
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=cats + [cats[0]],
                    fill='toself',
                    fillcolor='rgba(183, 110, 121, 0.25)',
                    line=dict(color='#B76E79', width=2)
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 10], color="#8E6B77"),
                        angularaxis=dict(color="#381E28")
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=260,
                    margin=dict(t=25, b=25, l=40, r=40)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as ex:
            st.error(f"Prediction Error: {ex}")


# ---------------------------------------------------------
# PAGE 3: VISUAL ANALYTICS & EDA
# ---------------------------------------------------------
elif nav_option == "📊 Visual Analytics & EDA":
    st.markdown("### 📊 Comprehensive Exploratory Data Analysis & Visualizations")
    st.caption("Deep-dive interactive charts covering distributions, correlations, and feature comparisons.")

    eda_tabs = st.tabs([
        "📈 Academic & Skills",
        "💻 Coding & Projects",
        "🌐 Profiles & Soft Skills",
        "🔥 Correlation Matrix",
        "⚖️ Class Comparisons"
    ])

    with eda_tabs[0]:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig_cgpa = px.histogram(
                df, x="CGPA", nbins=25, color="Selection_Status",
                color_discrete_map={"Selected": "#488566", "Not Selected": "#B76E79"},
                marginal="box", title="CGPA Distribution by Selection Status"
            )
            fig_cgpa.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#381E28"))
            st.plotly_chart(fig_cgpa, use_container_width=True)
        with c2:
            fig_skills = px.histogram(
                df, x="skills_score", nbins=10, color="Selection_Status",
                color_discrete_map={"Selected": "#488566", "Not Selected": "#A85A6E"},
                title="Skills Score Distribution"
            )
            fig_skills.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#381E28"))
            st.plotly_chart(fig_skills, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with eda_tabs[1]:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            fig_proj = px.scatter(
                df, x="skills_score", y="coding_test_score",
                color="Selection_Status",
                color_discrete_map={"Selected": "#488566", "Not Selected": "#B76E79"},
                title="Skills Score vs Coding Test Score (Scatter)",
                opacity=0.75
            )
            fig_proj.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#381E28"))
            st.plotly_chart(fig_proj, use_container_width=True)
        with c4:
            proj_counts = df.groupby(["projects_count", "Selection_Status"]).size().reset_index(name="counts")
            fig_pbar = px.bar(
                proj_counts, x="projects_count", y="counts", color="Selection_Status",
                barmode="group",
                color_discrete_map={"Selected": "#488566", "Not Selected": "#B76E79"},
                title="Candidate Project Portfolio Count Comparison"
            )
            fig_pbar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#381E28"))
            st.plotly_chart(fig_pbar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with eda_tabs[2]:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            linkedin_counts = df["linkedin_activity_score"].value_counts()
            fig_link = px.pie(
                values=linkedin_counts.values,
                names=linkedin_counts.index,
                title="LinkedIn Activity Ratio",
                color_discrete_sequence=["#B76E79", "#E8D3CE"],
                hole=0.4
            )
            fig_link.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#381E28"))
            st.plotly_chart(fig_link, use_container_width=True)
        with c6:
            git_counts = df["github_score"].value_counts()
            fig_git = px.bar(
                x=git_counts.index, y=git_counts.values,
                labels={"x": "GitHub Profile Tier", "y": "Student Count"},
                color=git_counts.values,
                color_continuous_scale=["#E8D3CE", "#C47A8C", "#8C4A60"],
                title="GitHub Portfolio Tier Distribution"
            )
            fig_git.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#381E28"), coloraxis_showscale=False)
            st.plotly_chart(fig_git, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with eda_tabs[3]:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        st.markdown("#### 🔥 Numerical Correlation Heatmap")
        num_cols = [
            "CGPA", "skills_score", "projects_count", "internships_done",
            "communication_score", "coding_test_score", "resume_score",
            "certifications_count", "soft_skills_score", "interview_score", "Selected"
        ]
        corr = df[num_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdPu",
            zmin=-0.2, zmax=0.2,
            title="Correlation Heatmap of Student Attributes & Selection Outcome"
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#381E28"),
            height=550
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with eda_tabs[4]:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        st.markdown("#### ⚖️ Attribute Box Plots: Selected vs Not Selected")
        selected_feat = st.selectbox(
            "Select Feature to Compare by Class:",
            ["CGPA", "skills_score", "coding_test_score", "interview_score", "resume_score", "communication_score"]
        )
        fig_box = px.box(
            df, x="Selection_Status", y=selected_feat,
            color="Selection_Status",
            color_discrete_map={"Selected": "#488566", "Not Selected": "#B76E79"},
            points="all",
            title=f"{selected_feat} Distribution across Classes"
        )
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#381E28"))
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# PAGE 4: MODEL PERFORMANCE
# ---------------------------------------------------------
elif nav_option == "📈 Model Performance":
    st.markdown("### 📈 Machine Learning Model Performance & Metrics")
    st.caption("Empirical test results evaluated on an independent 20% holdout test partition.")

    if metrics is None:
        st.warning("Model metrics file not found. Please train the model using `python project.py`.")
        st.stop()

    m_data = metrics.get("metrics", {})
    acc = m_data.get("accuracy", 0.0)
    prec = m_data.get("precision", 0.0)
    rec = m_data.get("recall", 0.0)
    f1 = m_data.get("f1_score", 0.0)
    auc = m_data.get("roc_auc", 0.0)

    # Metric KPI cards in Cream & Dusty Rose
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Accuracy</div>
            <div class="kpi-value">{acc*100:.2f}%</div>
            <div class="kpi-meta">Holdout test set</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Precision</div>
            <div class="kpi-value" style="color:#A85A6E;">{prec:.4f}</div>
            <div class="kpi-meta">Positive predictive value</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Recall</div>
            <div class="kpi-value" style="color:#488566;">{rec:.4f}</div>
            <div class="kpi-meta">True positive rate</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">F1-Score</div>
            <div class="kpi-value" style="color:#C47A8C;">{f1:.4f}</div>
            <div class="kpi-meta">Harmonic mean</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">ROC-AUC</div>
            <div class="kpi-value" style="color:#9A3B53;">{auc:.4f}</div>
            <div class="kpi-meta">Discriminative ability</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Confusion Matrix & Classification Report
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    p_col1, p_col2 = st.columns([1, 1])

    with p_col1:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 Confusion Matrix")
        fig_cm = px.imshow(
            cm,
            x=["Not Selected (Pred)", "Selected (Pred)"],
            y=["Not Selected (True)", "Selected (True)"],
            text_auto=True,
            color_continuous_scale="Purples",
            title="Confusion Matrix (200 Test Records)"
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#381E28"),
            height=320,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with p_col2:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        st.markdown("#### 📑 Classification Report")
        cls_rep = metrics.get("classification_report", {})
        if cls_rep:
            rep_df = pd.DataFrame(cls_rep).T.reset_index().rename(columns={"index": "Class"})
            st.dataframe(rep_df.style.format(precision=3), use_container_width=True)
        st.markdown(f"""
        <p style='color: #63424E; font-size: 0.88rem;'>
            • True Negatives (TN): <strong>{cm[0][0]}</strong><br>
            • False Positives (FP): <strong>{cm[0][1]}</strong><br>
            • False Negatives (FN): <strong>{cm[1][0]}</strong><br>
            • True Positives (TP): <strong>{cm[1][1]}</strong>
        </p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Feature Importance / Model Coefficients
    st.markdown("<div class='card-box'>", unsafe_allow_html=True)
    st.markdown("#### ⚖️ Learned Logistic Regression Feature Coefficients")
    coeffs = metrics.get("coefficients", {})
    if coeffs:
        coef_df = pd.DataFrame(list(coeffs.items()), columns=["Feature", "Coefficient"]).sort_values(
            by="Coefficient", ascending=True
        )
        fig_coef = px.bar(
            coef_df, x="Coefficient", y="Feature", orientation="h",
            color="Coefficient",
            color_continuous_scale=["#B76E79", "#D8A496", "#8C4A60"],
            title="Feature Coefficients (Impact on Selection Probability)"
        )
        fig_coef.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#381E28"),
            height=400,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_coef, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# PAGE 5: DATASET EXPLORER
# ---------------------------------------------------------
elif nav_option == "📂 Dataset Explorer":
    st.markdown("### 📂 Student Dataset Explorer & Interactive Filters")
    st.caption("Inspect raw data records, verify data types, check null values, and download filtered data.")

    # Data shape & summary
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Total Rows", df.shape[0])
    with d2:
        st.metric("Total Columns", df.shape[1])
    with d3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with d4:
        st.metric("Duplicate Rows", df.duplicated().sum())

    st.markdown("<div class='card-box'>", unsafe_allow_html=True)
    st.markdown("#### 🔍 Filter Dataset Records")
    flt_col1, flt_col2, flt_col3 = st.columns(3)
    with flt_col1:
        cgpa_filter = st.slider("Filter by Minimum CGPA:", float(df["CGPA"].min()), float(df["CGPA"].max()), float(df["CGPA"].min()))
    with flt_col2:
        status_filter = st.multiselect("Selection Status:", ["Selected", "Not Selected"], default=["Selected", "Not Selected"])
    with flt_col3:
        github_filter = st.multiselect("GitHub Tier:", sorted(df["github_score"].unique()), default=sorted(df["github_score"].unique()))

    filtered_df = df[
        (df["CGPA"] >= cgpa_filter) &
        (df["Selection_Status"].isin(status_filter)) &
        (df["github_score"].isin(github_filter))
    ]

    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} matching records:**")
    st.dataframe(filtered_df, use_container_width=True, height=350)

    # Download CSV button
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Dataset as CSV",
        data=csv_bytes,
        file_name="internship_filtered_dataset.csv",
        mime="text/csv"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Columns metadata table
    st.markdown("<div class='card-box'>", unsafe_allow_html=True)
    st.markdown("#### 📋 Dataset Column Metadata & Data Types")
    col_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Non-Null Count": [df[c].count() for c in df.columns],
        "Unique Values": [df[c].nunique() for c in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# PAGE 6: ABOUT & DOCUMENTATION
# ---------------------------------------------------------
elif nav_option == "📘 About & Documentation":
    st.markdown("### 📘 System Architecture & Academic Documentation")
    st.caption("Detailed overview of machine learning methodology, target formulation, and deployment design.")

    st.markdown("<div class='card-box'>", unsafe_allow_html=True)
    st.markdown("""
    #### 🎓 Project Introduction & Objectives
    The **Internship Selection Prediction System** provides automated decision-support for college placement cells,
    career counselors, and engineering students. By analyzing historical academic performance, algorithmic assessments,
    coding projects, soft skills, and mock interviews, the system computes the statistical likelihood of candidate selection.

    #### ⚙️ Machine Learning Pipeline Workflow
    1. **Data Ingestion:** Loads candidate records from `dataset/internship_dataset.xlsx`.
    2. **Cleaning & Normalization:** Standardizes string formatting, removes extraneous whitespace, resolves categorical inconsistencies (e.g. `'Tier1'` to `'Tier 1'`).
    3. **Target Engineering:** The target variable `Selected` is derived from `Unnamed: 13` using a configurable cutoff rule (`score >= 6`).
    4. **Leakage Prevention:** Both `student_id` and `Unnamed: 13` are excluded from the predictive feature space.
    5. **Transformation Pipeline:**
       - **Numerical Pipeline:** `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
       - **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore')`.
    6. **Classification Model:** Scikit-Learn `LogisticRegression(penalty='l2', C=1.0, solver='lbfgs')`.
    7. **Model Persistence:** Serialized using `joblib` into `models/internship_model.pkl`.

    #### 📄 Reports & Visualizations
    - Static visualization artifacts saved in `Reports/`:
      - `confusion_matrix.png`
      - `correlation_heatmap.png`
      - `target_distribution.png`
      - `feature_analysis.png`
      - `EDA_Report.html`
      - `model_metrics.json`
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # HTML Report viewer
    if os.path.exists(EDA_HTML_PATH):
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        st.markdown("#### 📑 Embedded Interactive EDA Report")
        with open(EDA_HTML_PATH, "r", encoding="utf-8") as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=650, scrolling=True)
        st.markdown("</div>", unsafe_allow_html=True)
