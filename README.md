# FinGuard Pro  
## Explainable AI for Financial Fraud Detection and Anti-Financial Crime Compliance

FinGuard Pro is a **real-time, explainable financial fraud detection system** designed for **UPI-style digital payment transactions**.  
The system combines **machine learning**, **rule-based AML screening**, and **SHAP-based explainability** to deliver transparent, compliance-ready fraud risk assessments.

This project is associated with an **IEEE international conference publication**.

---

## Problem Statement

Modern fraud detection systems often achieve high predictive accuracy, but operate as **black boxes**, making them unsuitable for **regulated financial environments**. Financial institutions require systems that not only detect fraud effectively but also **explain why a transaction is flagged**, enabling auditability, trust, and human oversight.

---

## Solution Overview

FinGuard Pro addresses this challenge through a **hybrid architecture** that integrates:

- High-recall machine learning models for fraud detection
- SHAP-based explainability for transparent decision-making
- Rule-based AML screening for known financial crime patterns
- Real-time APIs and dashboards for operational use

The system prioritizes **missed-fraud minimization** while ensuring that every decision is **human-interpretable**.

---

## System Architecture

```
UPI Transaction Stream
↓
Feature Engineering
↓
XGBoost Fraud Detection Model
↓
SHAP Explainability Engine
↓
Rule-based AML Screening
↓
Flask REST APIs
↓
User & Analyst Dashboards
```

---

## Key Features

- Real-time fraud risk scoring for digital payments  
- SHAP-based **global and local explanations**  
- Analyst dashboards for transaction investigation  
- Rule-based AML detection (structuring, velocity, geo anomalies)  
- Fuzzy name screening against watchlists and PEPs  
- Modular ML pipeline for experimentation and extension  

---

## Dataset

- **Synthetic UPI transaction dataset**
- ~**100,000 transactions**
- ~**3% fraud rate**
- Designed using **RBI fraud typologies** and PaySim-inspired behavioral models
- Includes simulated **AML watchlist entities (~1,000 profiles)**

The dataset was generated to reflect **realistic Indian digital payment behavior** while preserving privacy.

---

## Machine Learning Approach

- **Problem Type:** Binary classification (fraud / non-fraud)
- **Primary Model:** XGBoost
- **Key Challenges Addressed:**
  - Severe class imbalance
  - False-negative minimization
  - Interpretability in high-risk decisions

### Model Strategy
- Threshold tuning to prioritize recall
- Feature engineering focused on transaction behavior patterns
- Evaluation across multiple fraud-rate scenarios

---

## Results & Evaluation

### Model Performance (XGBoost)

- **Recall:** 94%
- **Precision:** 41%
- **F1-score:** 0.57
- **Accuracy:** 97%
- **AUC-ROC:** ≈ 0.995
- **Inference Latency:** < 3 seconds per transaction

The system prioritizes **high recall** to minimize missed fraudulent transactions, while managing false positives through downstream verification.

### Robustness Analysis
The model was evaluated under varying fraud rates (0.5%, 1%, and 5%) and consistently maintained high recall, demonstrating robustness under realistic UPI fraud scenarios.

---

## Explainability & Compliance

Explainability is a **core design principle**, not an afterthought.

FinGuard Pro integrates **SHAP (TreeExplainer)** to provide:

- **Global explanations** for feature importance
- **Local explanations** (waterfall plots) for individual transactions

These explanations enable:
- Transparent fraud investigation
- Regulatory auditability
- Human-in-the-loop decision-making

---

## AML & Rule-Based Screening

In addition to ML-based detection, FinGuard Pro includes:

- **Fuzzy name screening** using RapidFuzz
- Rule-based detection of:
  - Transaction structuring
  - Round-tripping
  - Velocity anomalies
  - Geographic inconsistencies

This hybrid approach strengthens detection coverage and aligns with real-world anti-financial crime workflows.

---

## Tech Stack

- **Language:** Python  
- **ML & XAI:** XGBoost, scikit-learn, SHAP  
- **Backend:** Flask, REST APIs  
- **Visualization:** Streamlit  
- **Database:** PostgreSQL  

---

## Installation

```bash
git clone https://github.com/Somaskandan931/FinGuardPro.git
cd FinGuardPro
pip install -r requirements.txt
```

## Running the Application

```bash
python app.py
```

or for dashboards:

```bash
streamlit run dashboard.py
```

---

## Limitations

- Synthetic dataset; real-world deployment would require live data validation
- Designed for structured transaction data only
- Explainability introduces minor inference overhead

---

## Future Work

- Streaming ingestion using Kafka
- Automated compliance report generation
- Model drift detection and monitoring
- Deep learning models with explainability

---

## Publication

**FinGuard Pro: Explainable AI for Financial Fraud Detection and Anti-Financial Crime Compliance**  
Presented at the 4th International Conference on Applied Artificial Intelligence and Computing (ICAAIC 2025)  
Published in IEEE Conference Proceedings  
ISBN: 979-8-3315-6587-9

---

## License

Academic and research use only.
