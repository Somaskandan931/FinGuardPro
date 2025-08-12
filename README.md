Here’s your cleaned-up **Markdown (`.md`)** version — ready to save as `README.md` and submit.

```markdown
# FinGuard Pro – Real-Time Financial Fraud Detection and Compliance Platform

**FinGuard Pro** is an end-to-end artificial intelligence system for detecting fraudulent financial transactions in real time.  
The platform combines deep learning models, SHAP-based explainability, automated PDF reporting, Streamlit dashboards, and a Flask-based REST API.  
It is designed for deployment by compliance teams, auditors, and end-users, with support for both monitoring and investigation workflows.

---

## 1. Project Structure

```

FinGuardPro/
├── models/            # Trained model files and preprocessing assets
├── data/              # CSV input datasets for training/testing
├── dashboards/        # Streamlit dashboards (Admin and User)
├── explain/           # SHAP utilities and generated visualizations
├── reports/           # PDF and ZIP report generation logic
├── api/               # Flask API for real-time fraud detection
├── notebooks/         # Jupyter/Colab notebooks for model training
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── .gitignore

````

---

## 2. Features

| Module                             | Description                                                  |
|------------------------------------|--------------------------------------------------------------|
| Fraud Detection Model              | Autoencoder + Dense layers for real-time fraud scoring       |
| SHAP Explainability                 | Visual interpretation of transaction feature importance      |
| PDF Report Generator                | Automated fraud audit reports                                |
| Batch Reporting (ZIP)               | Bulk generation of flagged transaction reports               |
| Admin Dashboard                     | Role-based secure view for auditors and compliance teams     |
| User Dashboard                      | Personalized dashboard for individual risk monitoring        |
| REST API (Flask)                    | Backend service for fraud prediction and SHAP plot delivery  |

---

## 3. Installation

### 3.1 Clone the Repository
```bash
git clone https://github.com/Somaskandan931/finguard-pro.git
cd FinGuardPro
````

### 3.2 Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3.3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Usage Guide

### 4.1 Model Training

Open the following notebook in Jupyter or Google Colab:

```
notebooks/training_pipeline.ipynb
```

After training, the following files are generated:

* `models/fraud_detection_model.h5`
* `models/scaler.pkl`
* `models/label_encoders.pkl`

---

### 4.2 Running the Admin Dashboard

```bash
streamlit run dashboards/admin_dashboard.py
```

**Alternative:**

```bash
python run_dashboards.py admin
```

**Default Credentials:**

* `admin` / `admin123`
* `compliance` / `admin123`

**Key Functions:**

* Transaction CSV upload and analysis
* Real-time fraud risk scoring
* Advanced analytics and visualizations
* CSV export of results
* System monitoring and configuration

---

### 4.3 Running the User Dashboard

```bash
streamlit run dashboards/user_dashboard.py
```

**Alternative:**

```bash
python run_dashboards.py user
```

**Default Credentials:**

* `user1` / `user123`
* `user2` / `user123`
* `demo` / `user123`

**Key Functions:**

* Personal transaction upload and analysis
* Risk visualization and assessment
* Security recommendations
* Report export options

---

### 4.4 Running the Flask API

```bash
python api/api_server.py
```

**Endpoints:**

* `POST /predict` → Accepts JSON input; returns fraud score and SHAP plot URL
* `GET /shap-image` → Returns the most recent SHAP plot as PNG

---

## 5. Batch Report Generation

Reports can be generated via the Admin Dashboard or programmatically:

```python
from reports.zip_reports import generate_batch_reports
import pandas as pd

df = pd.read_csv("data/test_transactions.csv")
generate_batch_reports(df)
```

---

## 6. Requirements

```
streamlit
streamlit-authenticator
tensorflow
pandas
numpy
scikit-learn
matplotlib
shap
fpdf
joblib
flask
PyYAML
```

---

## 7. Security Considerations

* Passwords are securely hashed using `streamlit-authenticator`
* Session cookies ensure role-based access isolation
* SHAP explanations restricted to administrator accounts

---

## 8. Deployment

* **Streamlit Dashboards**: Deploy to [Streamlit Cloud](https://streamlit.io/cloud)
* **Flask API**: Deploy to [Render](https://render.com) or [Railway](https://railway.app)


