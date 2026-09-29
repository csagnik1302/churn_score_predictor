# Customer Churn Risk Prediction & MLOps Pipeline

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.16-blue.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue.svg)](https://github.com/features/actions)
[![AWS Fargate](https://img.shields.io/badge/AWS-ECS%20Fargate-orange.svg)](https://aws.amazon.com/fargate/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An enterprise-grade, end-to-end Machine Learning Operations (MLOps) system designed to predict **multi-class customer churn risk** across 5 distinct churn severity categories (Classes 0 to 4). 

The platform integrates custom data cleaning and MICE numerical imputation, hyperparameter optimization using **Optuna** cross-validation, experiment tracking via **MLflow**, automated CI/CD container workflows via **GitHub Actions**, web API and interactive UI serving using **FastAPI** and **Gradio**, and scalable cloud orchestration on **AWS ECS Fargate** behind an **Application Load Balancer (ALB)**.

---

## 📌 Table of Contents

- [Project Overview & Key Features](#-project-overview--key-features)
- [Customer Segmentation & Business Rubric](#-customer-segmentation--business-rubric)
- [Repository Directory Structure](#-repository-directory-structure)
- [End-to-End MLOps Data & Model Pipeline](#-end-to-end-mlops-data--model-pipeline)
- [Continuous Integration & Deployment (CI/CD)](#-continuous-integration--deployment-cicd)
- [Model Benchmarking & Experimental Results](#-model-benchmarking--experimental-results)
- [In-Depth Inferences & Experimental Notes](#-in-depth-inferences--experimental-notes)
- [Quick Setup & Installation Guide](#-quick-setup--installation-guide)
- [API & UI Usage](#-api--ui-usage)
- [Docker & AWS Deployment Architecture](#-docker--aws-deployment-architecture)
- [Generating Technical PDF Report](#-generating-technical-pdf-report)

---

## 🚀 Project Overview & Key Features

* **Multi-Class Churn Risk Classification:** Predicts ordinal churn risk scores (1 to 5 in raw business terms, zero-indexed to 0 to 4 for gradient boosting algorithms).
* **Automated Data Preprocessing Pipeline:** Stratified train/test split, metadata cleanup, invalid/sentinel value handling, temporal feature engineering, and hybrid missing value imputation (Mode + MICE IterativeImputer).
* **Hyperparameter Optimization with Optuna:** 5-fold Stratified K-Fold cross-validation optimizing for Macro-averaged F1 score across gradient boosting algorithms (XGBoost, LightGBM, CatBoost).
* **MLflow Experiment Tracking:** Complete logging of hyperparameter configurations, evaluation metrics (accuracy, recall, macro/weighted F1), dataset signatures, and serialized PyFunc model artifacts.
* **Automated CI/CD Workflow:** GitHub Actions workflow (`.github/workflows/ci.yml`) automatically builds multi-platform Docker images via Docker Buildx on push to `main` and publishes image `csagnik1302/ml_img:latest` to Docker Hub.
* **Dual API & Interactive Web Interface:** FastAPI REST server (`POST /predict`, `GET /` health check) with Pydantic request validation, mounted alongside a user-friendly Gradio web interface (`/ui`).
* **AWS ECS Fargate Cloud Deployment:** Containerized microservice running Python 3.12, deployed on serverless AWS Fargate (0.5 vCPU, 1 GB RAM) registered with an HTTP ALB target group for zero-downtime rolling updates.

---

## 📊 Customer Segmentation & Business Rubric

The project categorizes customer risk across 5 distinct profiles based on engagement, membership, spending habits, and feedback:

| Raw Score / ML Class | Customer Profile | Behavioral & Financial Metrics | Strategic Retention Action |
|---|---|---|---|
| **Score 1 & 2**<br/>*(Class 0 & 1)* | **Happy & Loyal** | Positive feedback only (*"Products in Stock"*, *"Quality Customer Care"*, *"Reasonable Price"*). Median transaction ~$52k (2x other groups). Wallet points 750–800; logins every ~10 days. Exclusively Gold, Silver, Platinum, or Premium members. | Protect relationship; avoid unnecessary discount spend or spam. |
| **Score 3**<br/>*(Class 2)* | **Middling Customers** | Negative feedback present; mid-tier membership. Median transaction ~$26k, wallet points ~750. Neutral/borderline engagement. | Targeted feedback resolution & proactive engagement. |
| **Score 4**<br/>*(Class 3)* | **Unhappy & Drifting** | Negative feedback prevalent. Skews to Basic/No Membership, or lower Silver/Gold tiers. Wallet points ~$660. Highest risk of sliding to active churn. | Service quality remediation & active intervention. |
| **Score 5**<br/>*(Class 4)* | **High Risk Churner** | Negative feedback dominant; 63–64% of all Basic and No Membership customers land here. Lowest wallet points (~600). | Aggressive win-back campaigns or targeted incentives. |

---

## 📂 Repository Directory Structure

```text
mlops/
├── .github/
│   └── workflows/
│       └── ci.yml                      # GitHub Actions CI/CD workflow definition
├── data/
│   ├── model/
│   │   ├── model_optimal_parameters/   # Saved Optuna best parameters (JSON)
│   │   ├── model_performance/          # Classification metrics & evaluation CSVs
│   │   └── optuna_data/                # Complete Optuna trial history CSVs
│   ├── parsed/                         # Preprocessed train/test data & labels
│   └── raw/                            # Raw train, test, and sample submission files
├── mlruns/                             # MLflow experiment tracking storage
├── notebook/
│   ├── eda.ipynb                       # Exploratory Data Analysis notebook
│   └── model_tests.ipynb               # Hyperparameter search notebook
├── src/
│   ├── app/
│   │   └── app.py                      # FastAPI server & Gradio UI interface
│   ├── data/
│   │   ├── load_data.py                # Data ingestion module
│   │   └── preprocess.py              # Cleaning, MICE imputation, OHE, scaling
│   ├── models/
│   │   ├── evaluate.py                 # Classification report & confusion matrix
│   │   ├── train.py                    # Model training & MLflow logging
│   │   └── tune.py                     # OptunaSearchCV hyperparameter tuning
│   ├── pipeline/
│   │   └── run_pipeline.py             # End-to-end execution script
│   └── serving/
│       ├── inference.py                # Production prediction pipeline
│       ├── models/                     # Loaded MLflow model artifacts
│       └── trained_models/             # Joblib encoders & standard scalers
├── config.toml                         # Pipeline hyperparameter configuration
├── dockerfile                          # Containerization build file
├── generate_report.py                  # ReportLab PDF report generation script
├── pyproject.toml                      # Project dependencies & environment spec
├── README.md                           # Documentation & quick start guide
└── rubric.txt                          # Business rubric reference
```

---

## 🤖 Continuous Integration & Deployment (CI/CD)

The project includes an automated GitHub Actions CI/CD workflow defined in [`.github/workflows/ci.yml`](file:///.github/workflows/ci.yml):

```yaml
name: Build and Push to Docker Hub

on:
    push:
        branches: [main]

jobs:
    build-and-push:
        runs-on: ubuntu-latest
        steps:
            - name: Checkout code
              uses: actions/checkout@v4

            - name: Setup Docker Buildx
              uses: docker/setup-buildx-action@v3

            - name: Login to Docker Hub
              uses: docker/login-action@v3
              with:
                username: ${{secrets.DOCKERHUB_USERNAME}}
                password: ${{secrets.DOCKERHUB_TOKEN}}

            - name: Build and Push Docker image
              uses: docker/build-push-action@v5
              with:
                context: .
                file: ./dockerfile
                push: true
                tags: csagnik1302/ml_img:latest
```

### Key Workflow Actions:
1. **Trigger:** Automatically runs on every `push` event to the `main` branch.
2. **Environment:** Executes on `ubuntu-latest`.
3. **Build Engine:** Uses `docker/setup-buildx-action@v3` for modern, multi-platform Docker container compilation.
4. **Registry Push:** Authenticates securely via `docker/login-action@v3` using repository secrets (`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`) and publishes image `csagnik1302/ml_img:latest` to Docker Hub.
5. **AWS ECS Link:** AWS ECS Fargate tasks pull `csagnik1302/ml_img:latest` directly during automated deployment rollouts.

---

## 🔄 End-to-End MLOps Data & Model Pipeline

```text
Raw Data (train.csv)
       │
       ▼
1. Preprocessing (src/data/preprocess.py)
   ├── Stratified Train/Test Split (80/20)
   ├── Drop Identifiers & Standardize Column Names
   ├── Sentinel Value Handling (-999 -> NaN, Negative values -> NaN)
   ├── Temporal Engineering (days_since_joined = '2021-04-01' - joining_date)
   ├── Target Complete Case Analysis (CCA)
   ├── Hybrid Imputation (Mode + MICE IterativeImputer)
   ├── Categorical One-Hot Encoding (OneHotEncoder drop='first')
   ├── Multicollinearity Filtering (Threshold = 0.85)
   └── StandardScaler Normalization & Target Zero-Indexing (0-4)
       │
       ▼
2. Hyperparameter Tuning (src/models/tune.py)
   └── OptunaSearchCV (5-Fold Stratified K-Fold CV optimizing Macro F1)
       │
       ▼
3. Model Training & Experiment Tracking (src/models/train.py)
   ├── Sample Weight Computation (class_weight='balanced')
   ├── Model Fitting (XGBoost Classifier)
   └── MLflow Logging (Params, Metrics, Input Datasets, Model Signatures)
       │
       ▼
4. Continuous Integration & Push (CI/CD .github/workflows/ci.yml)
   └── GitHub Push (main) -> Docker Buildx -> Push to Docker Hub (csagnik1302/ml_img:latest)
       │
       ▼
5. Cloud Serving (AWS ECS Fargate & ALB)
   ├── Pulls csagnik1302/ml_img:latest
   ├── Health Check GET / on Port 8000
   ├── FastAPI REST Endpoint (POST /predict)
   └── Gradio Interactive Web Interface (/ui)
```

---

## 📈 Model Benchmarking & Experimental Results

Three gradient boosting architectures were tuned using **OptunaSearchCV** across 30 trial iterations with 5-fold Stratified K-Fold cross-validation:

| Model Architecture | Best Optuna CV (Macro F1) | Test Accuracy | Test Macro F1 | Test Weighted F1 | Optimal Hyperparameters |
|---|---|---|---|---|---|
| **XGBoost Classifier** *(Champion)* | 0.7725 | **78.27%** | **0.7689** | **0.7777** | `n_estimators`: 107, `learning_rate`: 0.0358, `max_depth`: 8, `subsample`: 0.88, `colsample_bytree`: 0.73, `gamma`: 4.04 |
| **LightGBM Classifier** | **0.7727** | 77.99% | 0.7585 | 0.7746 | `n_estimators`: 129, `learning_rate`: 0.0392, `num_leaves`: 255, `max_depth`: 8, `min_child_samples`: 27 |
| **CatBoost Classifier** | 0.7700 | **78.27%** | 0.7642 | 0.7725 | `iterations`: 144, `learning_rate`: 0.1979, `depth`: 4, `l2_leaf_reg`: 1.42, `random_strength`: 3.68 |

### Champion Model (XGBoost) Per-Class Metrics

| Class Index | Customer Segment | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|---|
| **Class 0** | Happy / Premium | 0.7108 | 0.8271 | 0.7646 | 214 |
| **Class 1** | Happy / High Value | 0.8000 | 0.6758 | 0.7327 | 219 |
| **Class 2** | Middling | 0.9803 | 0.8357 | 0.9023 | 834 |
| **Class 3** | Unhappy / Drifting | 0.7615 | 0.5471 | 0.6368 | 817 |
| **Class 4** | High Risk Churner | 0.6828 | 0.9898 | 0.8081 | 783 |
| **Macro Avg** | *Unweighted Mean* | 0.7871 | 0.7751 | **0.7689** | 2867 |
| **Weighted Avg** | *Support Weighted* | 0.8028 | 0.7827 | **0.7777** | 2867 |

---

## 🧠 In-Depth Inferences & Experimental Notes

The codebase includes three technical notes providing key methodological insights:

1. **Note 1 (Optimization Metric Selection):**
   * **Why Macro F1:** Optuna requires a single scalar target. Macro F1 balances precision and recall simultaneously per class. Optimizing for Recall alone would sacrifice precision (flagging loyal customers as churners and wasting marketing budget). Macro F1 ensures minority churn classes are not drowned out by majority active classes.
   * **Diagnostic Metric Layering:** After tuning, model performance is evaluated against Class-Specific Recall, Class-Specific Precision, Confusion Matrix, and Cohen's Kappa.

2. **Note 2 (Empirical Validation — Macro F1 vs. Recall Tuning):**
   * **Experimental Finding:** Model 1 (F1-tuned) beat Model 2 (Recall-tuned) on **BOTH** Macro F1 (**0.769 vs 0.764**) AND Macro Recall (**0.775 vs 0.771**).
   * **Why Recall-Tuning Failed:** A recall-only search over-indexed on Class 4 (~0.99 recall, ~0.68 precision) because it suffered no false-positive penalty. On Class 3 (the hardest category to predict), recall *dropped* under recall tuning (**0.547 -> 0.523**).
   * **Conclusion:** Standardizing on Macro-averaged F1 is empirically validated as the superior tuning objective.

3. **Note 3 (AWS Cloud Infrastructure & Container Deployment):**
   * **Decoupled Architecture:** Deployed on AWS ECS Fargate (0.5 vCPU, 1 GB RAM) behind an internet-facing Application Load Balancer (ALB on HTTP port 80).
   * **Why Ephemeral IPs are Decoupled:** Fargate task IP addresses change whenever containers restart. Routing through ALB DNS provides a stable endpoint and seamless rolling deployments.
   * **Liveness Monitoring:** ALB Target Group performs HTTP health checks against `GET /` expecting `{"status": "ok"}`.

---

## ⚡ Quick Setup & Installation Guide

### Prerequisites

* Python 3.12+
* [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`
* Docker (optional, for containerization)

### Step 1: Clone Repository & Install Dependencies

```bash
git clone https://github.com/csagnik1302/mlops.git
cd mlops

# Create virtual environment and install dependencies using uv
uv sync
```

*(Alternatively using standard pip: `pip install -e .`)*

### Step 2: Run End-to-End Pipeline

Execute raw data loading, preprocessing, model training, MLflow logging, and evaluation in a single command:

```bash
uv run python -m src.pipeline.run_pipeline
```

### Step 3: Run Hyperparameter Tuning with Optuna

To launch 30-trial Optuna cross-validation search for optimal hyperparameters:

```bash
uv run python -m src.models.tune
```

---

## 🌐 API & UI Usage

### Start FastAPI Server & Gradio UI

Launch the Uvicorn application server:

```bash
uv run uvicorn src.app.app:app --host 0.0.0.0 --port 8000 --reload
```

Once running:
* **Interactive Gradio Web UI:** Navigate to `http://localhost:8000/ui`
* **FastAPI OpenAPI Interactive Docs:** Navigate to `http://localhost:8000/docs`
* **Health Check Endpoint:** `GET http://localhost:8000/` -> `{"status": "ok"}`

### Sample REST Request (`POST /predict`)

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 43,
  "days_since_last_login": 56.0,
  "avg_time_spent_in_seconds": 134.0,
  "avg_transaction_value": 45.0,
  "avg_frequency_login_days": 6.0,
  "points_in_wallet": 45.0,
  "joining_date": "2015-04-19",
  "gender": "M",
  "region_category": "Village",
  "membership_category": "Silver Membership",
  "joined_through_referral": "Yes",
  "preferred_offer_types": "Without Offers",
  "medium_of_operation": "Smartphone",
  "internet_option": "Mobile_Data",
  "used_special_discount": "Yes",
  "offer_application_preference": "Yes",
  "past_complaint": "No",
  "complaint_status": "Not Applicable",
  "feedback": "Poor Website"
}'
```

**Response:**
```json
3
```
*(Returns integer class prediction 0 to 4 representing churn risk level).*

---

## 🐳 Docker & AWS Deployment Architecture

### Build & Run Docker Container Locally

```bash
# Build Docker image
docker build -t customer-churn-api .

# Run container mapping port 8000
docker run -p 8000:8000 customer-churn-api
```

### AWS Cloud Serving Architecture

```text
Client Request
      │
      │ HTTP :80
      ▼
Internet-Facing Application Load Balancer (ALB)
      │
      │ HTTP :8000 (Target Group Health Check GET /)
      ▼
AWS ECS Fargate Service (Desired Count: 1 Task, 0.5 vCPU / 1 GB RAM)
      │
      ▼
FastAPI / Uvicorn Server (Container Port 8000)
      │
      ▼
ML Inference Engine (Loads Scaler, Encoder & Trained Model)
      │
      ▼
JSON Churn Risk Prediction (Classes 0-4)
```

---

## 📄 Generating Technical PDF Report

This project includes an automated ReportLab script that compiles a multi-page report:

```bash
uv run python generate_report.py
```

Output report: `churn_prediction_project_report.pdf`

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
