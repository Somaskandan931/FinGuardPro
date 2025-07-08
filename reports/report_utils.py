from fpdf import FPDF
from datetime import datetime
import os

FONT_PATH = "fonts/DejaVuSans.ttf"  # Adjust path if needed


class FraudPDFReport(FPDF):
    def __init__(self, font_family="Arial"):
        super().__init__()
        self.font_family = font_family

    def header(self):
        self.set_font(self.font_family, "B", 14)
        self.cell(0, 10, self._safe_text("FinGuard Pro – Fraud Transaction Report"), ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_family, "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_transaction_details(self, txn_data, fraud_score, risk_level):
        self.set_font(self.font_family, "", 12)
        self.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        self.ln(5)

        for key, value in txn_data.items():
            display_value = self._safe_text(str(value))
            if len(display_value) > 60:
                display_value = display_value[:57] + "..."
            self.cell(50, 8, self._safe_text(f"{key}:"), border=0)
            self.multi_cell(0, 8, display_value)

        self.ln(2)
        self.set_font(self.font_family, "B", 12)
        self.cell(0, 10, f"Fraud Score: {fraud_score:.4f}", ln=True)
        self.cell(0, 10, f"Risk Level: {risk_level}", ln=True)
        self.ln(5)

    def add_shap_plot(self, image_path=None, max_width=170):
        if image_path and os.path.isfile(image_path):
            self.image(image_path, w=max_width)
            self.ln(5)
        else:
            self.set_font(self.font_family, "I", 10)
            self.cell(0, 10, "SHAP explanation image not found or path invalid.", ln=True)

    def _safe_text(self, text):
        # Replace problematic unicode chars with ascii alternatives
        if not isinstance(text, str):
            text = str(text)
        replacements = {
            "\u2013": "-",  # en dash
            "\u2014": "-",  # em dash
            "\u2019": "'",  # right single quote
            "\u2018": "'",  # left single quote
            "\u201c": '"',  # left double quote
            "\u201d": '"',  # right double quote
            "\u2026": "...",  # ellipsis
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text


def generate_fraud_report(
    txn_data: dict,
    fraud_score: float,
    risk_level: str,
    save_path: str = "C:/Users/somas/PycharmProjects/FinGuardPro/output/fraud_report.pdf",
    shap_image_path: str = "C:/Users/somas/PycharmProjects/FinGuardPro/output/shap_explanation.png"
) -> str:
    pdf = FraudPDFReport()

    # Try registering DejaVu font, else fallback to Arial
    if os.path.exists(FONT_PATH):
        pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
        pdf.add_font("DejaVu", "B", FONT_PATH, uni=True)
        pdf.add_font("DejaVu", "I", FONT_PATH, uni=True)
        pdf.font_family = "DejaVu"
    else:
        print(f"⚠️ Font not found at {FONT_PATH}, falling back to Arial.")
        pdf.font_family = "Arial"

    pdf.add_page()
    pdf.add_transaction_details(txn_data, fraud_score, risk_level)
    pdf.add_shap_plot(shap_image_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pdf.output(save_path)
    print(f"✅ PDF report saved to {save_path}")
    return save_path
