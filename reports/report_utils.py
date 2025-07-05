from fpdf import FPDF
from datetime import datetime
import os


class FraudPDFReport( FPDF ) :
    def header ( self ) :
        self.set_font( "Arial", "B", 14 )
        self.cell( 0, 10, "FinGuard Pro - Fraud Transaction Report", ln=True, align="C" )
        self.ln( 5 )

    def footer ( self ) :
        self.set_y( -15 )
        self.set_font( "Arial", "I", 8 )
        self.cell( 0, 10, f"Page {self.page_no()}", align="C" )

    def add_transaction_details ( self, txn_data, fraud_score, risk_level ) :
        self.set_font( "Arial", "B", 12 )
        self.cell( 0, 10, "Transaction Details", ln=True )
        self.ln( 2 )

        self.set_font( "Arial", "", 10 )
        self.cell( 0, 8, f"Generated on: {datetime.now().strftime( '%Y-%m-%d %H:%M:%S' )}", ln=True )
        self.ln( 3 )

        # Add transaction data
        for key, value in txn_data.items() :
            if key not in ['fraud_score', 'fraud_label'] :  # Skip computed fields
                self.cell( 0, 6, f"{key}: {value}", ln=True )

        self.ln( 3 )
        self.set_font( "Arial", "B", 10 )
        self.cell( 0, 8, f"Fraud Score: {fraud_score:.4f}", ln=True )
        self.cell( 0, 8, f"Risk Level: {risk_level}", ln=True )
        self.ln( 5 )

    def add_shap_plot ( self, image_path="C:/Users/somas/PycharmProjects/FinGuardPro/explain/shap_explanation.png" ) :
        if os.path.exists( image_path ) :
            self.set_font( "Arial", "B", 12 )
            self.cell( 0, 10, "SHAP Explanation", ln=True )
            self.ln( 2 )

            try :
                # Add image with proper sizing
                self.image( image_path, x=10, w=190 )
            except Exception as e :
                self.set_font( "Arial", "", 10 )
                self.cell( 0, 8, f"Error loading SHAP image: {e}", ln=True )
        else :
            self.set_font( "Arial", "", 10 )
            self.cell( 0, 8, "SHAP explanation image not available", ln=True )


def generate_fraud_report ( txn_data, fraud_score, risk_level,
                            save_path="C:/Users/somas/PycharmProjects/FinGuardPro/reports/fraud_report.pdf" ) :
    """
    Generate a PDF report for a fraud transaction

    Args:
        txn_data: Dictionary containing transaction data
        fraud_score: Fraud probability score
        risk_level: Risk level string
        save_path: Path to save the PDF report

    Returns:
        Path to the generated PDF report
    """
    try :
        # Ensure directory exists
        save_dir = os.path.dirname( save_path )
        if save_dir and not os.path.exists( save_dir ) :
            os.makedirs( save_dir )

        pdf = FraudPDFReport()
        pdf.add_page()
        pdf.add_transaction_details( txn_data, fraud_score, risk_level )
        pdf.add_shap_plot()
        pdf.output( save_path )

        print( f"✅ PDF report saved to {save_path}" )
        return save_path

    except Exception as e :
        print( f"❌ Error generating PDF report: {e}" )
        return None