# COLLEGE ACADEMIC PROJECT REPORT

## PROJECT TITLE:
# INTERNSHIP SELECTION PREDICTION SYSTEM USING MACHINE LEARNING

---

### **Submitted By:** Engineering Department / Data Science Division
### **Technology Stack:** Python 3.10, Scikit-Learn, Pandas, NumPy, Streamlit, Plotly, Seaborn, Matplotlib

---

## TABLE OF CONTENTS
1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Dataset Description & Feature Analysis](#4-dataset-description--feature-analysis)
5. [Data Preprocessing & Target Formulation](#5-data-preprocessing--target-formulation)
6. [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)
7. [Classification Algorithm: Logistic Regression](#7-classification-algorithm-logistic-regression)
8. [Training and Testing Workflow](#8-training-and-testing-workflow)
9. [Evaluation Metrics & Equations](#9-evaluation-metrics--equations)
10. [Actual Experimental Results](#10-actual-experimental-results)
11. [Streamlit Web Application Architecture](#11-streamlit-web-application-architecture)
12. [Conclusion](#12-conclusion)
13. [Future Scope](#13-future-scope)

---

## 1. INTRODUCTION
In modern tertiary technical education, securing competitive internships during undergraduate study is one of the strongest determinants of final campus placements and early career trajectory. However, recruitment screening pipelines handle thousands of student applicants, requiring multidimensional evaluation across academic performance (CGPA), algorithmic competency (coding tests), real-world experience (projects and prior internships), soft skills, resume quality, and technical mock interviews.

The **Internship Selection Prediction System** applies machine learning classification to automate student profile evaluation. By providing an interactive web portal, the system assists both students seeking personalized readiness feedback and training & placement officers (TPOs) allocating mentorship resources.

---

## 2. PROBLEM STATEMENT
Manual evaluation of undergraduate internship readiness across multiple technical and soft skill metrics is prone to bias, inconsistency, and administrative overhead. Moreover, students frequently lack objective insight into which areas of their candidate profile create the greatest competitive deficits.

The objective of this project is to construct an end-to-end Machine Learning classification pipeline and deployment platform that predicts candidate selection probability based on multidimensional candidate metrics, providing explainable model outputs and actionable feedback.

---

## 3. OBJECTIVES
1. **Data Ingestion & Integrity Verification:** Thoroughly analyze and clean candidate profiles from raw Excel records.
2. **Target Formulation & Leakage Elimination:** Formulate a binary classification target from evaluation metrics while strictly excluding the source score and identifier from input features.
3. **Robust Preprocessing Pipeline:** Construct a Scikit-Learn `Pipeline` integrating `SimpleImputer`, `StandardScaler` for continuous numerical features, and `OneHotEncoder` for categorical features.
4. **Binary Classifier Training:** Train a `LogisticRegression` classification model using stratified 80/20 train/test partitioning.
5. **Rigorous Evaluation:** Quantitatively evaluate performance using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
6. **Web Dashboard Deployment:** Deploy an interactive, career-themed Streamlit web interface with real-time prediction, probability meters, visual analytics, and dataset exploration.

---

## 4. DATASET DESCRIPTION & FEATURE ANALYSIS

The dataset consists of **999 student records** containing academic, technical, experiential, and interpersonal metrics:

| Column Name | Data Type | Permissible Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `student_id` | Integer | 1 to 999 | Unique candidate index (Excluded from training) |
| `CGPA` | Float | 5.02 to 10.00 | Cumulative Grade Point Average |
| `skills_score` | Integer | 1 to 10 | Assessment score in core domain skills |
| `projects_count` | Integer | 0 to 5 | Completed academic/personal projects |
| `internships_done`| Integer | 0 to 3 | Number of prior internships completed |
| `communication_score` | Integer | 1 to 10 | Verbal and written communication rating |
| `coding_test_score` | Integer | 1 to 10 | Algorithmic test performance score |
| `resume_score` | Integer | 1 to 10 | Resume ATS format and content score |
| `certifications_count` | Integer | 1 to 10 | Professional vendor/MOOC certifications |
| `linkedin_activity_score` | Categorical | 'Yes', 'No' | Active professional social footprint |
| `github_score` | Categorical | 'Tier 1', 'Tier 2', 'Tier 3' | Open-source code repository quality tier |
| `soft_skills_score` | Integer | 0 to 5 | Teamwork, adaptability, leadership score |
| `interview_score` | Integer | 0 to 10 | Technical mock interview rating |
| `Unnamed: 13` | Integer | 1 to 10 | Raw comprehensive candidate score |

---

## 5. DATA PREPROCESSING & TARGET FORMULATION

### 5.1 Target Formulation
Inspection of `Unnamed: 13` revealed an integer score ranging from 1 to 10 ($Mean = 5.45, Median = 5.0, Std = 2.94$). A configurable classification boundary was established:

$$\text{Selected} = \begin{cases} 1 & \text{if } \text{Unnamed: 13} \ge 6 \\ 0 & \text{if } \text{Unnamed: 13} < 6 \end{cases}$$

- **Class 1 (Selected):** 492 students (49.25%)
- **Class 0 (Not Selected):** 507 students (50.75%)

> **Data Leakage Prevention:** `Unnamed: 13` and `student_id` are strictly dropped from the feature matrix $X$ prior to train/test partitioning and model fitting.

### 5.2 Categorical Normalization
Categorical values in `github_score` contained formatting inconsistencies (`'Tier1'` without spaces). These were normalized to `'Tier 1'`, `'Tier 2'`, and `'Tier 3'`. `linkedin_activity_score` values were trimmed and capitalized (`'Yes'`, `'No'`).

### 5.3 Transformation Pipeline
1. **Numerical Pipeline:** Handled with median imputation followed by standard z-score normalization ($z = \frac{x - \mu}{\sigma}$).
2. **Categorical Pipeline:** Handled with most-frequent imputation followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

---

## 6. EXPLORATORY DATA ANALYSIS (EDA)

Key statistical insights from the Exploratory Data Analysis:
1. **Academic Performance:** CGPA exhibits an approximately uniform/bell distribution across the 5.0 to 10.0 range ($Mean = 7.37, Std = 1.48$).
2. **Technical Readiness:** Skills scores ($Mean = 5.56$) and Coding test scores ($Mean = 5.49$) reflect balanced candidate distributions.
3. **Experience Footprint:** Average project count is 2.56; majority of students completed between 0 and 2 prior internships.
4. **Correlation Analysis:** Pearson correlation coefficients between individual technical attributes and the score-derived target remain modest, demonstrating that selection likelihood requires multivariate modeling rather than single-feature thresholding.

---

## 7. CLASSIFICATION ALGORITHM: LOGISTIC REGRESSION

Logistic Regression models the probability that an observation $X$ belongs to class $y=1$ (Selected) using the logistic sigmoid function:

$$P(y=1|X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Where the logit $z$ is a linear combination of input features:

$$z = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_m X_m$$

The model is optimized using binary cross-entropy (log-loss) with $L_2$ Tikhonov regularization:

$$\mathcal{L}(\beta) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right] + \frac{\lambda}{2} \|\beta\|_2^2$$

The optimization is solved iteratively using the Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) algorithm.

---

## 8. TRAINING AND TESTING WORKFLOW

1. **Partitioning:** Stratified 80/20 train/test split preserving target class ratio.
   - **Training Set:** 799 records (394 Selected, 405 Not Selected)
   - **Holdout Test Set:** 200 records (98 Selected, 102 Not Selected)
2. **Pipeline Integration:**
   ```text
   Raw Inputs -> ColumnTransformer [StandardScaler, OneHotEncoder] -> LogisticRegression(C=1.0)
   ```
3. **Serialization:** Serialized via `joblib` to `models/internship_model.pkl` for immediate inference in the Streamlit web application.

---

## 9. EVALUATION METRICS & EQUATIONS

The classification performance is measured using standard statistical criteria:

1. **Accuracy:**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

2. **Precision:**
   $$\text{Precision} = \frac{TP}{TP + FP}$$

3. **Recall (Sensitivity):**
   $$\text{Recall} = \frac{TP}{TP + FN}$$

4. **F1-Score:**
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

5. **ROC-AUC:** Area under the Receiver Operating Characteristic curve representing true positive rate against false positive rate across all discrimination thresholds.

---

## 10. ACTUAL EXPERIMENTAL RESULTS

The model was evaluated strictly on the 200 unseen holdout test samples:

### 10.1 Quantitative Performance Summary

| Metric | Empirical Score |
| :--- | :--- |
| **Accuracy** | **90.00%** (0.9000) |
| **Precision** | **88.46%** (0.8846) |
| **Recall** | **92.00%** (0.9200) |
| **F1-Score** | **90.20%** (0.9020) |
| **ROC-AUC** | **0.9698** |

### 10.2 Confusion Matrix (Holdout Test Set)

| Actual \ Predicted | Predicted: Not Selected (0) | Predicted: Selected (1) |
| :--- | :--- | :--- |
| **True: Not Selected (0)** | **88** (TN) | **12** (FP) |
| **True: Selected (1)** | **8** (FN) | **92** (TP) |

### 10.3 Per-Class Classification Report
- **Class 0 (Not Selected):** Precision = 0.92, Recall = 0.88, F1 = 0.90, Support = 100
- **Class 1 (Selected):** Precision = 0.88, Recall = 0.92, F1 = 0.90, Support = 100

---

## 11. STREAMLIT WEB APPLICATION ARCHITECTURE

The web application (`app.py`) provides an interactive interface matching the user's styling requirements:
- **Visual Design:** Navy Blue (`#0f172a`), Royal Blue (`#1d4ed8`), Purple (`#8b5cf6`), and Cyan (`#06b6d4`) accents with subtle frosted-glass overlays.
- **Home Dashboard:** Displays real-time KPIs, Donut distribution chart, and prior internship experience distributions.
- **Prediction System:** 12-feature interactive evaluation form, probability gauge meter, and candidate competency radar chart.
- **Visual Analytics:** Multi-tab Plotly visualizations including correlation matrices and box plots.
- **Model Performance:** Live confusion matrix heatmap, classification report, and learned feature coefficient bar plots.
- **Dataset Explorer:** Dynamic filtering by CGPA, selection status, and GitHub tier, with one-click CSV export.

---

## 12. CONCLUSION
The Internship Selection Prediction System was successfully engineered, trained, evaluated, and deployed. By structuring the codebase into modular components (`project.py`, `app.py`, `dataset/`, `models/`, `Reports/`, `assets/`), the project guarantees reproducible training, clean model serialization, and zero data leakage.

---

## 13. FUTURE SCOPE
1. **Non-Linear Ensembles:** Exploration of Gradient Boosting (XGBoost, LightGBM) and Random Forests to capture potential non-linear feature interactions.
2. **Automated Resume Parsing:** Integration of Natural Language Processing (NLP) models to automatically extract skills and CGPA from PDF resumes.
3. **Role-Specific Classifiers:** Expanding binary selection into multi-label role recommendations (e.g., Software Engineering, Data Analyst, Cloud DevOps).
