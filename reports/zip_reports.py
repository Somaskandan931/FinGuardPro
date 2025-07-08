import os
import zipfile
from reports.report_utils import generate_fraud_report

def generate_batch_reports(
    df,
    fraud_score_column="fraud_score",
    output_dir="reports/fraud_reports",
    threshold=0.3
):
    os.makedirs(output_dir, exist_ok=True)
    pdf_paths = []

    for _, row in df.iterrows():
        score = row.get(fraud_score_column, 0)
        if score < threshold:
            continue

        risk_level = "High Risk" if score > 0.7 else "Suspicious"
        txn_data = row.to_dict()
        txn_id = txn_data.get("transaction_id", "unknown")
        pdf_filename = f"{output_dir}/txn_{txn_id}_{risk_level.replace(' ', '_')}.pdf"
        generate_fraud_report(txn_data, score, risk_level, save_path=pdf_filename)
        pdf_paths.append(pdf_filename)

    # Create ZIP of all PDFs
    zip_path = os.path.join(output_dir, "all_fraud_reports.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for pdf in pdf_paths:
            arcname = os.path.basename(pdf)
            zipf.write(pdf, arcname=arcname)

    return zip_path, pdf_paths
