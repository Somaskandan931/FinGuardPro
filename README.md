
# 💼 FinGuard Pro – Real-Time Financial Fraud Detection & Compliance System

FinGuard Pro is an end-to-end deep learning-based system for detecting fraudulent financial transactions in real time. It includes SHAP-based explainability, PDF reporting, Streamlit dashboards, and a Flask API — built for both compliance teams and end users.

---

## 📂 Project Structure

```
FinGuardPro/
├── models/                   # Trained model and preprocessors
├── data/                    # CSV input for training/testing
├── dashboards/              # Admin & user Streamlit dashboards
├── explain/                 # SHAP utils + saved images
├── reports/                 # PDF/ZIP report generation logic
├── api/                     # Flask API for real-time fraud detection
├── notebooks/               # Model training notebook (Colab/Jupyter)
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore
```

---

## ✅ Features

| Module                     | Description                                            |
|----------------------------|--------------------------------------------------------|
| 🧠 Fraud Model (Autoencoder + Dense) | Real-time scoring of suspicious transactions     |
| 🔍 SHAP Explainability     | Bar/force plots for transaction feature importance     |
| 📄 PDF Report Generator    | One-click fraud audit reports                          |
| 📦 Batch Reports (ZIP)     | Generate all flagged transactions in bulk              |
| 🖥️ Admin Dashboard         | Secure role-based view for auditors/compliance         |
| 👤 User Dashboard          | End-user dashboard to track transaction risks          |
| 🌐 REST API                | Flask API for backend integration                      |

---

## ⚙️ Installation

### 📦 Clone Repo

```bash
git clone https://github.com/Somaskandan931/finguard-pro.git
cd FinGuardPro
```

### 🐍 Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 📦 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 1️⃣ Train the model

Open in Jupyter or Colab:

```
notebooks/training_pipeline.ipynb
```

After training, it will save:
- `models/fraud_detection_model.h5`
- `models/scaler.pkl`
- `models/label_encoders.pkl`

---

### 2️⃣ Run Admin Dashboard

```bash
streamlit run dasboards/admin_dashboard.py
```

**OR use the launcher script:**

```bash
python run_dashboards.py admin
```

**Login Credentials:**
- Username: `admin`
- Password: `admin123`

**OR**

- Username: `compliance`
- Password: `admin123`

**Features:**
- Modern, responsive UI with gradient backgrounds
- Multi-page dashboard navigation
- Upload and analyze transaction CSV files
- Real-time fraud detection with risk scoring
- Advanced analytics and visualizations
- Export results to CSV
- System status monitoring
- Settings configuration

---

### 3️⃣ Run User Dashboard

```bash
streamlit run dasboards/user_dashboard.py
```

**OR use the launcher script:**

```bash
python run_dashboards.py user
```

**Login Credentials:**
- Username: `user1`
- Password: `user123`

**OR**

- Username: `user2`
- Password: `user123`

**OR**

- Username: `demo`
- Password: `user123`

**Features:**
- Personal financial security dashboard
- Upload and analyze personal transactions
- Risk assessment with visual indicators
- Transaction analytics and charts
- Export personal reports
- Security tips and recommendations
- Customizable user settings

---

### 4️⃣ Run Flask API

```bash
python api/api_server.py
```

**Endpoints:**

- `/predict` (POST): Accepts JSON of transaction → returns fraud score + SHAP URL  
- `/shap-image` (GET): Returns most recent SHAP plot as PNG

---

## 📁 Generate Batch Reports (ZIP)

Can be triggered via Admin Dashboard  
OR run manually:

```python
from reports.zip_reports import generate_batch_reports
df = pd.read_csv("data/test_transactions.csv")
generate_batch_reports(df)
```

---

## ✅ Requirements

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

## 🛡️ Security Notes

- Passwords are securely hashed using `streamlit-authenticator`
- Session cookies isolate user/admin dashboard
- SHAP explanations are only shown to admins

---

## 🌐 Deployment

- **Streamlit Dashboards** → https://streamlit.io/cloud
- **Flask API** → https://render.com or https://railway.app

---

## 👨‍💻 Contributors

- Somaskandan Rajagopal (Project Owner)

---

## 📜 License

MIT License. Use freely for academic, research, or fintech prototype use.

---