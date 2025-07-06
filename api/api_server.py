from flask import Flask, request, jsonify, send_file
import pandas as pd
import os

from modules.decision_engine import run_decision_engine
from modules.name_screening import add_names_to_watchlist
from explain.explain_utils import generate_shap_plot
from reports.report_utils import generate_pdf_report

app = Flask(__name__)

# Health check
@app.route('/')
def index():
    return jsonify({"message": "✅ FinGuard Pro API is live."})

# POST /predict — Run fraud detection
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        results_df = run_decision_engine(df)
        return jsonify(results_df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# GET /explain/<transaction_id> — SHAP Explanation
@app.route('/explain/<transaction_id>', methods=['GET'])
def explain(transaction_id):
    try:
        shap_path = generate_shap_plot(transaction_id)
        if not os.path.exists(shap_path):
            return jsonify({"error": "SHAP explanation not found"}), 404
        return send_file(shap_path, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# POST /add_aml — Add names to AML list
@app.route('/add_aml', methods=['POST'])
def add_aml():
    try:
        data = request.get_json()
        names = data.get("names", [])
        if not names:
            return jsonify({"error": "No names provided"}), 400
        add_names_to_watchlist(names)
        return jsonify({"message": f"{len(names)} name(s) added to AML watchlist."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# GET /report/<transaction_id> — Download PDF
@app.route('/report/<transaction_id>', methods=['GET'])
def report(transaction_id):
    try:
        pdf_path = generate_pdf_report(transaction_id)
        if not os.path.exists(pdf_path):
            return jsonify({"error": "Report not found"}), 404
        return send_file(pdf_path, mimetype='application/pdf')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
