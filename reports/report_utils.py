from fpdf import FPDF
from datetime import datetime

class FraudPDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "FinGuard Pro – Fraud Transaction Report", ln=True, align="C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_transaction_details(self, txn_data, fraud_score, risk_level):
        self.set_font("Arial", "", 12)
        self.cell(0, 10, f"Generated on: {datetime.now()}", ln=True)
        self.ln(5)
        for key, value in txn_data.items():
            self.cell(0, 8, f"{key}: {value}", ln=True)
        self.cell(0, 10, f"Fraud Score: {fraud_score}", ln=True)
        self.cell(0, 10, f"Risk Level: {risk_level}", ln=True)
        self.ln(5)

    def add_shap_plot(self, image_path="shap_explanation.png"):
        self.image(image_path, w=170)

def generate_fraud_report(txn_data, fraud_score, risk_level, save_path="fraud_report.pdf"):
    pdf = FraudPDFReport()
    pdf.add_page()
    pdf.add_transaction_details(txn_data, fraud_score, risk_level)
    pdf.add_shap_plot()
    pdf.output(save_path)
    print(f"✅ PDF report saved to {save_path}")
    return save_path
