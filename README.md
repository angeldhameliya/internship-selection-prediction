# 🎓 Internship Selection Prediction System

> **Smart Career Insights Powered by Machine Learning**  
> An end-to-end Machine Learning classification and interactive Streamlit web dashboard designed to forecast student internship selection probability based on academic, technical, experiential, and interpersonal profiles.

---

## 📌 Project Overview

The **Internship Selection Prediction System** enables universities, training & placement cells (TPO), and students to assess candidate placement readiness. Built with **Python**, **Scikit-Learn (Logistic Regression)**, and **Streamlit**, this project mirrors the architectural pattern of classic prediction workflows with custom styling, full data leakage prevention, and complete college project documentation.

---

## 🗂️ Project Directory Structure

```text
Internship_Selection_Prediction/
├── project.py                 # Core ML training pipeline & automated report generation
├── app.py                     # Interactive multi-page Streamlit web dashboard
├── requirements.txt           # Python dependencies and versions
├── README.md                  # Setup instructions and documentation
├── Project_Report.md          # Comprehensive academic college project report
├── dataset/
│   └── internship_dataset.xlsx # Cleaned source Excel dataset
├── models/
│   └── internship_model.pkl   # Serialized Scikit-learn Pipeline (Scaler + OHE + LogisticRegression)
├── assets/
│   ├── background.jpg         # High-resolution career/technology background
│   └── logo.png               # Custom high-res career & graduation brand logo
└── Reports/
    ├── EDA_Report.html        # Interactive standalone HTML EDA analysis report
    ├── model_metrics.json     # Test-set evaluation metrics (Accuracy, Precision, Recall, etc.)
    ├── confusion_matrix.png   # High-resolution confusion matrix heatmap
    ├── correlation_heatmap.png# Multi-variable Pearson correlation matrix
    ├── feature_analysis.png   # Learned feature coefficients / importance chart
    └── target_distribution.png# Target class distribution (Counts & Proportions)
```

---

## ⚙️ Machine Learning Pipeline

1. **Dataset Cleaning & Normalization:**
   - Cleans column headers and strips whitespace.
   - Normalizes categorical tiers (`Tier1` &rarr; `Tier 1`).
2. **Target Formulation & Leakage Prevention:**
   - Target variable `Selected` is derived via thresholding on evaluation score `Unnamed: 13` (Default: `score >= 6` &rarr; Selected `1`, else `0`).
   - `Unnamed: 13` and `student_id` are strictly excluded from predictive features to prevent data leakage.
3. **Pre-processing:**
   - **Numerical Features:** Imputation (`median`) followed by `StandardScaler`.
   - **Categorical Features:** Imputation (`most_frequent`) followed by `OneHotEncoder(handle_unknown='ignore')`.
4. **Classifier:**
   - `LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)`.
5. **Stratified Partitioning:**
   - 80% Training (799 samples), 20% Testing (200 samples).

---

## 📊 Actual Experimental Results (Test Set)

| Metric | Holdout Test Set Value |
| :--- | :--- |
| **Accuracy** | **90.00%** (0.9000) |
| **Precision** | **88.46%** (0.8846) |
| **Recall** | **92.00%** (0.9200) |
| **F1-Score** | **90.20%** (0.9020) |
| **ROC-AUC Score** | **0.9698** |
| **Total Test Samples** | 200 candidates |
| **Confusion Matrix** | True Negatives: 88, False Positives: 12, False Negatives: 8, True Positives: 92 |

---

## 🚀 Quick Start & Installation

### Step 1: Open Terminal in the Project Directory
```powershell
cd "C:\Users\ANGLE\OneDrive\Desktop\ml project\Internship_Selection_Prediction"
```

### Step 2: Install Python Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 3: Train Model & Generate Reports
```bash
python project.py
```
*This command loads the dataset, trains the Logistic Regression pipeline, evaluates holdout performance, exports `models/internship_model.pkl`, and compiles all charts and HTML reports in `Reports/`.*

### Step 4: Run the Streamlit Web Application
```bash
python -m streamlit run app.py
```
Open your browser and navigate to:
```text
http://localhost:8501
```

---

## 💻 Streamlit Web Application Features

1. **🏠 Home Dashboard:**
   - Live KPI cards: Total Students, Average CGPA, Skills Score, Coding Test Score, Interview Score, Projects Count, Selected vs Not Selected counts and rates.
   - Interactive Plotly Donut and Bar charts.
2. **🔮 Prediction System:**
   - Responsive multi-column form with sliders and dropdowns.
   - Real-time probability prediction gauge and radar chart.
   - Customized action plans for Selected and Not Selected candidates.
3. **📊 Visual Analytics & EDA:**
   - 5 interactive tabs: Academic & Skills, Coding & Projects, Profiles & Soft Skills, Correlation Heatmap, and Class Box Plots.
4. **📈 Model Performance:**
   - Confusion matrix heatmap, full classification report, and feature weights chart.
5. **📂 Dataset Explorer:**
   - Live multi-filter table (CGPA, Selection Status, GitHub Tier) and CSV export.
6. **📘 About & Documentation:**
   - Embedded interactive HTML EDA report viewer and academic references.

---

## 📜 Educational Use Disclaimer
*The target labels in this system are score-derived from candidate assessment indices. Predictions represent statistical indicators for career readiness guidance rather than binding employment offers.*
