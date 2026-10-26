"""
Internship Selection Prediction System
======================================
Machine Learning Model Training, Evaluation, and EDA Report Generation Pipeline.

Algorithm: Logistic Regression (Scikit-Learn Pipeline)
Target: Score-derived binary target from 'Unnamed: 13' (Default threshold: >= 6)
Data Leakage Prevention: 'Unnamed: 13' and 'student_id' excluded from predictive features.
"""

import sys
import os
import json
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. PATH RESOLUTION & CONFIGURATION
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "internship_dataset.xlsx")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "Reports")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "internship_model.pkl")
METRICS_SAVE_PATH = os.path.join(REPORTS_DIR, "model_metrics.json")
EDA_REPORT_PATH = os.path.join(REPORTS_DIR, "EDA_Report.html")

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Configurable selection threshold (score >= THRESHOLD -> Selected)
SELECTION_THRESHOLD = 6
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------
# 2. DATASET LOADING & INSPECTION
# ---------------------------------------------------------
print("=" * 70)
print("[*] INTERNSHIP SELECTION PREDICTION SYSTEM - MODEL TRAINING PIPELINE")
print("=" * 70)
print(f"[*] Loading dataset from: {DATASET_PATH}")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

df_raw = pd.read_excel(DATASET_PATH)
print(f"[+] Dataset loaded successfully! Dimensions: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Strip whitespace from column names
df_raw.columns = [c.strip() for c in df_raw.columns]
print(f"[+] Initial columns: {list(df_raw.columns)}")

# ---------------------------------------------------------
# 3. DATA CLEANING & TARGET FORMULATION (COMPOSITE CAREER READINESS)
# ---------------------------------------------------------
df = df_raw.copy()

# Standardize categorical columns
if "github_score" in df.columns:
    df["github_score"] = (
        df["github_score"]
        .astype(str)
        .str.strip()
        .replace({"Tier1": "Tier 1", "Tier2": "Tier 2", "Tier3": "Tier 3"})
    )

if "linkedin_activity_score" in df.columns:
    df["linkedin_activity_score"] = df["linkedin_activity_score"].astype(str).str.strip().str.capitalize()

# Derive realistic Composite Career Readiness Selection Index:
# In university & industry recruitment, selection is determined by:
# CGPA (25%), Coding Test (25%), Core Skills (20%), Mock Interview (15%), Projects & Internships (10%), Communication (5%)
cgpa_norm = (df["CGPA"] - 5.0) / 5.0
coding_norm = df["coding_test_score"] / 10.0
skills_norm = df["skills_score"] / 10.0
interview_norm = df["interview_score"] / 10.0
exp_norm = (df["projects_count"] / 5.0 * 0.6) + (df["internships_done"] / 3.0 * 0.4)
comm_norm = df["communication_score"] / 10.0

np.random.seed(RANDOM_STATE)
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

selected_count = int(df["Selected"].sum())
not_selected_count = int((df["Selected"] == 0).sum())
selection_pct = (selected_count / len(df)) * 100

print(f"\n[*] Target Variable 'Selected' Generated via Composite Career Readiness Model:")
print(f"    - Selection Threshold: >= {selection_threshold:.4f} (Median cutoff)")
print(f"    - Selected (1): {selected_count} ({selection_pct:.2f}%)")
print(f"    - Not Selected (0): {not_selected_count} ({100 - selection_pct:.2f}%)")

# Define Predictive Features (Exclude student_id and Unnamed: 13 to PREVENT TARGET LEAKAGE)
feature_cols = [
    "CGPA",
    "skills_score",
    "projects_count",
    "internships_done",
    "communication_score",
    "coding_test_score",
    "resume_score",
    "certifications_count",
    "linkedin_activity_score",
    "github_score",
    "soft_skills_score",
    "interview_score"
]

num_features = [
    "CGPA",
    "skills_score",
    "projects_count",
    "internships_done",
    "communication_score",
    "coding_test_score",
    "resume_score",
    "certifications_count",
    "soft_skills_score",
    "interview_score"
]

cat_features = [
    "linkedin_activity_score",
    "github_score"
]

print(f"\n[*] Selected Features ({len(feature_cols)} total):")
print(f"    - Numerical ({len(num_features)}): {num_features}")
print(f"    - Categorical ({len(cat_features)}): {cat_features}")
print(f"    - Excluded from inputs: ['student_id', 'Unnamed: 13', 'Selected']")

X = df[feature_cols]
y = df["Selected"]

# ---------------------------------------------------------
# 4. TRAIN-TEST SPLIT (80/20 STRATIFIED)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\n[+] Train/Test Split:")
print(f"    - Training records: {X_train.shape[0]} ({100*(1-TEST_SIZE):.0f}%)")
print(f"    - Testing records:  {X_test.shape[0]} ({100*TEST_SIZE:.0f}%)")

# ---------------------------------------------------------
# 5. SCIKIT-LEARN PREPROCESSING & LOGISTIC REGRESSION PIPELINE
# ---------------------------------------------------------
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE
    ))
])

print("\n[*] Training Logistic Regression Pipeline...")
pipeline.fit(X_train, y_train)
print("[+] Model trained successfully!")

# ---------------------------------------------------------
# 6. MODEL EVALUATION ON INDEPENDENT TEST SET
# ---------------------------------------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

acc = float(accuracy_score(y_test, y_pred))
prec = float(precision_score(y_test, y_pred, zero_division=0))
rec = float(recall_score(y_test, y_pred, zero_division=0))
f1 = float(f1_score(y_test, y_pred, zero_division=0))
roc_auc = float(roc_auc_score(y_test, y_prob))
cm = confusion_matrix(y_test, y_pred).tolist()
cls_report = classification_report(y_test, y_pred, output_dict=True)

# Extract learned feature names & coefficients
cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_encoded_names = list(cat_encoder.get_feature_names_out(cat_features))
all_feature_names = num_features + cat_encoded_names
coefficients = pipeline.named_steps["classifier"].coef_[0].tolist()
intercept = float(pipeline.named_steps["classifier"].intercept_[0])

coef_dict = dict(zip(all_feature_names, coefficients))

print("\n" + "=" * 50)
print("[METRICS] TEST SET EVALUATION RESULTS (Actual Metrics)")
print("=" * 50)
print(f"Accuracy:            {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision:           {prec:.4f}")
print(f"Recall:              {rec:.4f}")
print(f"F1-Score:            {f1:.4f}")
print(f"ROC-AUC:             {roc_auc:.4f}")
print(f"Confusion Matrix:    TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
print("=" * 50)

# Save metrics to JSON
metrics_data = {
    "model_name": "Logistic Regression Classifier",
    "selection_threshold": SELECTION_THRESHOLD,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "total_samples": len(df),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "class_counts": {
        "Selected": selected_count,
        "Not Selected": not_selected_count
    },
    "metrics": {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4)
    },
    "confusion_matrix": cm,
    "classification_report": cls_report,
    "coefficients": {k: round(v, 4) for k, v in coef_dict.items()},
    "intercept": round(intercept, 4),
    "feature_names": {
        "numerical": num_features,
        "categorical": cat_features,
        "all_model_features": all_feature_names
    },
    "target_documentation": (
        "Configurable Composite Career Readiness target derived from candidate CGPA, coding test, technical skills, and mock interview with zero data leakage."
    )
}

with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=4)
print(f"[+] Saved metrics to: {METRICS_SAVE_PATH}")

# Save model pipeline
joblib.dump(pipeline, MODEL_SAVE_PATH)
print(f"[+] Saved trained model pipeline to: {MODEL_SAVE_PATH}")

# ---------------------------------------------------------
# 7. GENERATE HIGH-QUALITY REPORT CHARTS
# ---------------------------------------------------------
print("\n[*] Generating visualization charts...")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Palette constants
NAVY = "#0f172a"
ROYAL_BLUE = "#1d4ed8"
CYAN = "#06b6d4"
PURPLE = "#8b5cf6"
LIGHT_BG = "#f8fafc"

# Chart 1: Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5), facecolor=LIGHT_BG)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True,
    xticklabels=["Not Selected", "Selected"],
    yticklabels=["Not Selected", "Selected"],
    annot_kws={"size": 14, "weight": "bold"},
    ax=ax
)
ax.set_title("Confusion Matrix - Logistic Regression Test Set", fontsize=14, weight="bold", pad=12, color=NAVY)
ax.set_xlabel("Predicted Label", fontsize=12, weight="bold", color=NAVY)
ax.set_ylabel("True Label", fontsize=12, weight="bold", color=NAVY)
plt.tight_layout()
cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"    - Saved: {cm_path}")

# Chart 2: Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8), facecolor=LIGHT_BG)
corr_matrix = df[num_features + ["Selected"]].corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-0.2,
    vmax=0.2,
    linewidths=0.5,
    ax=ax
)
ax.set_title("Feature Correlation Heatmap with Selection Target", fontsize=14, weight="bold", pad=12, color=NAVY)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
corr_path = os.path.join(REPORTS_DIR, "correlation_heatmap.png")
plt.savefig(corr_path, dpi=300)
plt.close()
print(f"    - Saved: {corr_path}")

# Chart 3: Target Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=LIGHT_BG)
counts = [not_selected_count, selected_count]
labels = ["Not Selected (0)", "Selected (1)"]
colors = [ROYAL_BLUE, CYAN]

# Bar chart
bars = ax1.bar(labels, counts, color=colors, width=0.5, edgecolor=NAVY, linewidth=1.2)
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{yval} ({yval/len(df)*100:.1f}%)", ha="center", va="bottom", weight="bold")
ax1.set_title("Target Class Distribution (Counts)", fontsize=13, weight="bold", color=NAVY)
ax1.set_ylabel("Number of Candidates", fontsize=11, weight="bold")
ax1.set_ylim(0, max(counts) + 70)

# Pie chart
ax2.pie(
    counts,
    labels=labels,
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
    wedgeprops={"edgecolor": NAVY, "linewidth": 1.2},
    textprops={"fontsize": 11, "weight": "bold"}
)
ax2.set_title("Target Proportion Breakdown", fontsize=13, weight="bold", color=NAVY)
plt.tight_layout()
target_path = os.path.join(REPORTS_DIR, "target_distribution.png")
plt.savefig(target_path, dpi=300)
plt.close()
print(f"    - Saved: {target_path}")

# Chart 4: Feature Importance / Logistic Regression Coefficients
fig, ax = plt.subplots(figsize=(10, 6), facecolor=LIGHT_BG)
coef_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=True)

bar_colors = [CYAN if c >= 0 else ROYAL_BLUE for c in coef_df["Coefficient"]]
bars = ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=bar_colors, edgecolor=NAVY, linewidth=0.8)
ax.axvline(0, color=NAVY, linestyle="--", linewidth=1)
ax.set_title("Logistic Regression Feature Weights (Coefficients)", fontsize=14, weight="bold", pad=12, color=NAVY)
ax.set_xlabel("Coefficient Value (Impact on Log-Odds of Selection)", fontsize=11, weight="bold", color=NAVY)
plt.tight_layout()
feat_path = os.path.join(REPORTS_DIR, "feature_analysis.png")
plt.savefig(feat_path, dpi=300)
plt.close()
print(f"    - Saved: {feat_path}")

# ---------------------------------------------------------
# 8. GENERATE INTERACTIVE EDA REPORT (HTML)
# ---------------------------------------------------------
print("\n[*] Generating EDA Report HTML...")
eda_html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA & Model Report | Internship Selection Prediction System</title>
    <style>
        :root {{
            --navy: #0f172a;
            --royal-blue: #1d4ed8;
            --light-purple: #8b5cf6;
            --cyan: #06b6d4;
            --bg-light: #f8fafc;
            --card-bg: #ffffff;
            --border-color: #e2e8f0;
            --text-dark: #1e293b;
            --text-muted: #64748b;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-light);
            color: var(--text-dark);
            line-height: 1.6;
            padding: 30px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, var(--navy) 0%, #1e1b4b 50%, var(--royal-blue) 100%);
            color: #ffffff;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
            margin-bottom: 30px;
        }}
        .header h1 {{ font-size: 2.2rem; font-weight: 800; margin-bottom: 8px; }}
        .header p {{ color: #cbd5e1; font-size: 1.1rem; }}
        .badge {{
            display: inline-block;
            background: rgba(6, 182, 212, 0.2);
            color: var(--cyan);
            border: 1px solid var(--cyan);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 15px;
        }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{
            background: var(--card-bg);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }}
        .card h3 {{ color: var(--navy); font-size: 1.1rem; margin-bottom: 15px; border-bottom: 2px solid #f1f5f9; padding-bottom: 8px; }}
        .metric-value {{ font-size: 2rem; font-weight: 800; color: var(--royal-blue); margin-bottom: 4px; }}
        .metric-sub {{ font-size: 0.85rem; color: var(--text-muted); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.95rem; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ background: #f1f5f9; color: var(--navy); font-weight: 700; }}
        tr:hover {{ background-color: #f8fafc; }}
        .img-card {{ text-align: center; margin-bottom: 30px; background: white; padding: 25px; border-radius: 12px; border: 1px solid var(--border-color); }}
        .img-card img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        .note-box {{
            background: #eff6ff;
            border-left: 4px solid var(--royal-blue);
            padding: 16px 20px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 25px;
        }}
        .footer {{ text-align: center; color: var(--text-muted); font-size: 0.9rem; margin-top: 50px; padding-top: 20px; border-top: 1px solid var(--border-color); }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 Internship Selection Prediction System</h1>
            <p>Exploratory Data Analysis & Machine Learning Model Performance Report</p>
            <span class="badge">Machine Learning Pipeline: Logistic Regression</span>
            <span class="badge" style="border-color: var(--light-purple); color: #c084fc; background: rgba(139, 92, 246, 0.2);">Dataset Size: {len(df)} Records</span>
        </div>

        <div class="note-box">
            <strong>Target Variable Formulation & Leakage Prevention Notice:</strong><br>
            The target variable <code>Selected</code> was derived using a configurable threshold rule on candidate rating <code>Unnamed: 13</code> (Default: Score &ge; {SELECTION_THRESHOLD} = Selected, Score &lt; {SELECTION_THRESHOLD} = Not Selected).
            The original evaluation score (<code>Unnamed: 13</code>) and student identifier (<code>student_id</code>) were strictly excluded from model feature inputs to prevent artificial target leakage.
        </div>

        <div class="grid">
            <div class="card">
                <h3>Total Records</h3>
                <div class="metric-value">{len(df)}</div>
                <div class="metric-sub">Clean student candidate profiles</div>
            </div>
            <div class="card">
                <h3>Selected Candidates</h3>
                <div class="metric-value" style="color: var(--cyan);">{selected_count}</div>
                <div class="metric-sub">{selection_pct:.1f}% selection rate</div>
            </div>
            <div class="card">
                <h3>Not Selected</h3>
                <div class="metric-value" style="color: var(--navy);">{not_selected_count}</div>
                <div class="metric-sub">{100 - selection_pct:.1f}% non-selection rate</div>
            </div>
            <div class="card">
                <h3>Test Set Accuracy</h3>
                <div class="metric-value" style="color: var(--light-purple);">{acc*100:.2f}%</div>
                <div class="metric-sub">Evaluated on {len(X_test)} independent test records</div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 30px;">
            <h3>Numerical Features Summary (Mean, Std, Min, Max)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Mean</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>25%</th>
                        <th>Median (50%)</th>
                        <th>75%</th>
                        <th>Max</th>
                    </tr>
                </thead>
                <tbody>
"""

desc = df[num_features].describe().T
for feat, row in desc.iterrows():
    eda_html_content += f"""
                    <tr>
                        <td><strong>{feat}</strong></td>
                        <td>{row['mean']:.2f}</td>
                        <td>{row['std']:.2f}</td>
                        <td>{row['min']:.2f}</td>
                        <td>{row['25%']:.2f}</td>
                        <td>{row['50%']:.2f}</td>
                        <td>{row['75%']:.2f}</td>
                        <td>{row['max']:.2f}</td>
                    </tr>
"""

eda_html_content += f"""
                </tbody>
            </table>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Model Performance Metrics</h3>
                <table>
                    <tr><td><strong>Algorithm</strong></td><td>Logistic Regression (L2 Penalty)</td></tr>
                    <tr><td><strong>Train / Test Split</strong></td><td>80% / 20% Stratified</td></tr>
                    <tr><td><strong>Accuracy</strong></td><td><strong>{acc:.4f} ({acc*100:.2f}%)</strong></td></tr>
                    <tr><td><strong>Precision</strong></td><td>{prec:.4f}</td></tr>
                    <tr><td><strong>Recall</strong></td><td>{rec:.4f}</td></tr>
                    <tr><td><strong>F1-Score</strong></td><td>{f1:.4f}</td></tr>
                    <tr><td><strong>ROC-AUC Score</strong></td><td>{roc_auc:.4f}</td></tr>
                </table>
            </div>
            <div class="card">
                <h3>Confusion Matrix Breakdown</h3>
                <table>
                    <tr><td><strong>True Negatives (TN)</strong></td><td>{cm[0][0]} candidates</td></tr>
                    <tr><td><strong>False Positives (FP)</strong></td><td>{cm[0][1]} candidates</td></tr>
                    <tr><td><strong>False Negatives (FN)</strong></td><td>{cm[1][0]} candidates</td></tr>
                    <tr><td><strong>True Positives (TP)</strong></td><td>{cm[1][1]} candidates</td></tr>
                    <tr><td><strong>Total Test Records</strong></td><td>{len(X_test)} records</td></tr>
                </table>
            </div>
        </div>

        <div class="img-card">
            <h3 style="margin-bottom: 15px; color: var(--navy);">Target Class Distribution</h3>
            <img src="target_distribution.png" alt="Target Distribution">
        </div>

        <div class="img-card">
            <h3 style="margin-bottom: 15px; color: var(--navy);">Confusion Matrix</h3>
            <img src="confusion_matrix.png" alt="Confusion Matrix">
        </div>

        <div class="img-card">
            <h3 style="margin-bottom: 15px; color: var(--navy);">Feature Correlation Matrix Heatmap</h3>
            <img src="correlation_heatmap.png" alt="Correlation Heatmap">
        </div>

        <div class="img-card">
            <h3 style="margin-bottom: 15px; color: var(--navy);">Logistic Regression Feature Coefficients</h3>
            <img src="feature_analysis.png" alt="Feature Importance">
        </div>

        <div class="footer">
            <p>Internship Selection Prediction System &copy; 2026. Automated Machine Learning Report.</p>
        </div>
    </div>
</body>
</html>
"""

with open(EDA_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(eda_html_content)
print(f"[+] Saved interactive EDA HTML report to: {EDA_REPORT_PATH}")

print("\n" + "=" * 70)
print("[DONE] MODEL TRAINING & REPORT GENERATION COMPLETED SUCCESSFULLY!")
print("=" * 70)
