# Python Data Science & Analytics 2.0 📊🤖

Welcome to the **Python Data Science & Analytics 2.0** repository. This project is a curated portfolio of advanced machine learning and data analytics tasks, demonstrating end-to-end data pipelines, predictive modeling, model interpretability (XAI), cost-benefit threshold tuning, and interactive business intelligence (BI) dashboarding.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-3.2.0-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.57.0-red.svg)](https://streamlit.io/)
[![XAI Enabled](https://img.shields.io/badge/interpretability-SHAP%20%2F%20LIME-purple.svg)](#1-term-deposit-subscription-prediction-bank-marketing)

---

## 📂 Table of Contents
- [🔍 Repository Structure](#-repository-structure)
- [📈 Project Summaries](#-project-summaries)
  - [1. Term Deposit Subscription Prediction](#1-term-deposit-subscription-prediction)
  - [2. Customer Segmentation](#2-customer-segmentation)
  - [3. Energy Consumption Time Series Forecasting](#3-energy-consumption-time-series-forecasting)
  - [4. Loan Default Risk with Business Cost Optimization](#4-loan-default-risk-with-business-cost-optimization)
  - [5. Interactive Business Dashboard (Streamlit)](#5-interactive-business-dashboard-streamlit)
- [⚙️ Setup and Installation](#-setup-and-installation)
- [🚀 How to Run the Dashboard](#-how-to-run-the-dashboard)

---

## 🔍 Repository Structure
```bash
├── Term_Deposit_Prediction/          # Task 1: Classification & XAI (Bank Marketing)
│   ├── data/
│   │   └── bank-additional-full.csv  # Demographics and economic features dataset
│   └── bank_marketing_prediction.ipynb # Fully executed analysis & SHAP/LIME plots
│
├── Customer_Segmentation/            # Task 2: Unsupervised Clustering (Mall Customers)
│   ├── data/
│   │   └── Mall_Customers.csv        # Age, income, and spending scores dataset
│   └── customer_segmentation.ipynb   # K-Means, Silhouette, and PCA/t-SNE notebooks
│
├── Time_Series_Forecasting/          # Task 3: Time Series Analysis (Household Power)
│   ├── data/
│   │   └── household_power_consumption_resampled.csv # Daily resampled energy dataset
│   └── time_series_forecasting.ipynb # SARIMAX, XGBoost, and Exponential Smoothing
│
├── Loan_Default_Optimization/        # Task 4: Risk Modeling & Cost Tuning (Home Credit)
│   ├── data/
│   │   └── application_train.csv     # Risk scores and demographic simulated dataset
│   └── loan_default_optimization.ipynb # Classifier training & cost-minimizing threshold sweep
│
└── Business_Dashboard/               # Task 5: BI Dashboard (Global Superstore)
    ├── data/
    │   └── Global_Superstore.csv     # Multi-region sales transactions dataset
    ├── business_dashboard_prep.ipynb # Data preparation & preliminary KPI checks
    └── dashboard.py                  # Streamlit application with custom glassmorphic UI
```

---

## 📈 Project Summaries

### 1. Term Deposit Subscription Prediction
* **Objective**: Predict whether a bank customer will subscribe to a term deposit based on direct marketing campaigns.
* **Methodology**: Dropped the `duration` column to prevent data leakage (per UCI guidelines). Engineered numeric scalers and categorical encoders.
* **Models**: Logistic Regression, Random Forest, and XGBoost Classifier (ROC-AUC ~`0.80`).
* **Explainable AI (XAI)**:
  - **SHAP (SHapley Additive exPlanations)**: TreeExplainer plots highlight global feature importance.
  - **LIME (Local Interpretable Model-agnostic Explanations)**: Custom-tuned instance explanation graphs for individual customer profiles.

### 2. Customer Segmentation
* **Objective**: Segment customers based on demographic features and spending behaviors to formulate target marketing campaigns.
* **Methodology**: Scaled numeric metrics, mapped the Elbow curve (WCSS) and Silhouette scores to determine optimal clusters ($k=5$).
* **Dimensionality Reduction**: Utilized **PCA** and **t-SNE** to compress 3D spaces into a 2D visualization of the segments:
  - **Target / Premium** (High Income, High Spend) -> *Loyalty programs, luxury collections.*
  - **Careful** (High Income, Low Spend) -> *Quality assurance, value-driven campaigns.*
  - **Spendthrifts** (Low Income, High Spend) -> *Flash sales, discounts, micro-financing.*
  - **Sensible** (Low Income, Low Spend) -> *Utility deals, essentials bundles.*
  - **Standard** (Average Income, Average Spend) -> *Seasonal offers, email campaigns.*

### 3. Energy Consumption Time Series Forecasting
* **Objective**: Forecast short-term household energy consumption using time-based features.
* **Methodology**: Resampled 130MB of raw minute-level readings into daily averages. Engineered temporal calendars (`DayOfWeek`, `IsWeekend`), lags ($t-1$, $t-7$, $t-30$), and rolling averages.
* **Models**: Compared **SARIMAX**, **XGBoost Regressor**, and **Holt-Winters Exponential Smoothing**.
* **Results**: XGBoost captures local daily variance perfectly, while Exponential Smoothing and SARIMAX capture seasonal cycles robustly.

### 4. Loan Default Risk with Business Cost Optimization
* **Objective**: Train default risk scoring models and optimize the classification decision boundary to minimize actual financial losses.
* **Cost Matrix**:
  - **False Negative (FN)**: Approving a client who defaults -> **Cost = \$10,000** (lost principal).
  - **False Positive (FP)**: Rejecting a client who would repay -> **Cost = \$1,500** (marketing + interest loss).
* **Optimization**: Swept probability thresholds from `0.0` to `1.0`. Since FNs are highly expensive, the optimal threshold shifts **lower** (around **0.18**), saving the bank substantial capital over the default `0.50` threshold.

### 5. Interactive Business Dashboard (Streamlit)
* **Objective**: Build an executive interactive dashboard for regional sales, profits, and segment distributions.
* **App Features**:
  - Dynamic sidebar filters (Region, Category, Sub-Category, Date range).
  - Custom dark-theme config with premium glassmorphism CSS metric cards.
  - Hover-enabled interactive Plotly charts (Monthly Sales Trend, Segment Donut Chart, Top 5 Customers, Category Bar Chart).
  - Direct filtered raw transactional table views.

---

## ⚙️ Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/M-Awad-27/Python-Data-Science-Analytics-2.0.git
   cd Python-Data-Science-Analytics-2.0
   ```

2. **Install dependencies**:
   Make sure you have Python 3.10+ installed, then run:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost shap lime plotly streamlit nbconvert
   ```

---

## 🚀 How to Run the Dashboard

To launch the Streamlit BI dashboard locally:
```bash
streamlit run Business_Dashboard/dashboard.py
```
The dashboard will open in your default browser at `http://localhost:8501`.
