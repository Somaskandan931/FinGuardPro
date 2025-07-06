import os
import zipfile
from report_utils import generate_fraud_report

def generate_batch_reports(
    df,
    fraud_score_column="fraud_score",
    output_dir="fraud_reports",
    threshold=0.3
):
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_paths = []

    for i, row in df.iterrows():
        score = row[fraud_score_column]
        if score < threshold:
            continue  # Skip low-risk transactions

        risk_level = "High Risk" if score > 0.7 else "Suspicious"
        txn_data = row.to_dict()
        pdf_filename = f"{output_dir}/txn_{i}_{risk_level.replace(' ', '_')}.pdf"
        generate_fraud_report(txn_data, score, risk_level, save_path=pdf_filename)
        pdf_paths.append(pdf_filename)

    # Create zip file
    zip_name = "all_fraud_reports.zip"
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for path in pdf_paths:
            zipf.write(path)

    return zip_name, pdf_paths
