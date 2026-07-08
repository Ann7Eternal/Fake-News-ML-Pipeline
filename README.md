# End-to-End ML Pipeline & Model Monitoring
**[🟢 Live Demo: Click Here to Test the Application](https://fake-news-ml-pipeline-csfse3fng8hbaev2uh6chs.streamlit.app/)**

**Author:** Yashaswi Harsh | B.Tech, DIT University

**Domain:** Machine Learning Operations (MLOps) & Natural Language Processing (NLP)

## 📌 Project Abstract

Most academic machine learning projects stop at the Jupyter Notebook phase. This project demonstrates a production-grade, end-to-end Machine Learning Pipeline built for the real-world deployment of a Fake News Detection system. It emphasizes **System Design**, **Data Pipeline Automation**, and **Live Model Monitoring** to handle inference latency and data drift.

## 🏗️ System Architecture

The architecture is modularized into three distinct sequential nodes to mimic an enterprise CI/CD ML environment:

1. **Data Ingestion Node (`data_pipeline.py`):** Automates the extraction, cleaning, shuffling, and splitting of raw data into training and validation sets.
2. **Training & Artifact Registry (`train.py`):** Utilizes a high-efficiency TF-IDF Vectorizer and Logistic Regression model, tracking performance metrics before exporting the model weights as serialized `.pkl` artifacts.
3. **Deployment & Monitoring Node (`app.py`):** A Streamlit-based web interface that caches model artifacts for low-latency inference and features a custom CSV-based logger to track system health in real-time.

## 📊 Live Monitoring & Data Drift (Case Study)

In production systems, static models inevitably degrade as real-world data evolves. This system includes a custom monitoring logger (`monitoring_log.csv`) that tracks:

* **Timestamp**
* **Input Length**
* **Inference Latency** (System performance)
* **Confidence Score** (Model certainty)

**Observed Data Drift:** During live inference testing, the model misclassified a legitimate 2026 World Cup sports article (Argentina defeating Egypt) as "Fake News."

* **The Cause:** The training dataset was heavily skewed toward older, dry political news. The dramatic, emotional language typical of sports journalism ("stunning comeback," "storming back") closely mirrored the linguistic patterns the model associated with fabricated content.
* **The Solution:** The monitoring logger successfully captured this anomaly, demonstrating exactly why production models require automated logging and scheduled retraining pipelines to handle domain mismatch and temporal data drift.

## 🚀 Quick Start Guide

### 1. Environment Setup

Create an isolated environment and install dependencies:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install pandas scikit-learn streamlit joblib

```

### 2. Execute the Pipeline

Run the pipeline sequentially to build the model from scratch.

**Phase 1: Data Processing**

```bash
python data_pipeline.py

```

*Expected Output: `train_data.csv` and `test_data.csv` generated in the `/data` folder.*

**Phase 2: Model Training**

```bash
python train.py

```

*Expected Output: Accuracy metrics printed to console; `tfidf_vectorizer.pkl` and `logistic_model.pkl` generated in the `/models` folder.*

**Phase 3: Deployment**

```bash
streamlit run app.py

```

*Expected Output: Local server launches at `http://localhost:8501`. Inference telemetry will automatically log to `monitoring_log.csv`.*

## 🔮 Future Scope

To scale this Minimum Viable Product (MVP) to enterprise standards, the following integrations are planned:

* **Containerization:** Wrapping the pipeline in Docker for OS-agnostic deployment.
* **Cloud Infrastructure:** Migrating the local artifact registry to Google Cloud Storage (GCS) and deploying the Streamlit container via Google Cloud Run.
* **Advanced Orchestration:** Replacing standard Python scripts with Apache Airflow or MLflow for automated hyperparameter tuning.
