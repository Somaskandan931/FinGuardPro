import os
import zipfile
from reports.report_utils import generate_fraud_report


def generate_batch_reports (
        df,
        fraud_score_column="fraud_score",
        output_dir="C:/Users/somas/PycharmProjects/FinGuardPro/reports/batch_reports",
        threshold=0.3,
        zip_path="C:/Users/somas/PycharmProjects/FinGuardPro/reports/all_fraud_reports.zip"
) :
    """
    Generate batch PDF reports for flagged transactions

    Args:
        df: DataFrame with transaction data and fraud scores
        fraud_score_column: Column name containing fraud scores
        output_dir: Directory to save individual PDF reports
        threshold: Minimum fraud score threshold for report generation
        zip_path: Path for the output ZIP file

    Returns:
        Tuple of (zip_path, list_of_pdf_paths)
    """
    try :
        # Create output directory
        if not os.path.exists( output_dir ) :
            os.makedirs( output_dir )

        pdf_paths = []

        # Create reports directory if it doesn't exist
        reports_base_dir = os.path.dirname( zip_path )
        if reports_base_dir and not os.path.exists( reports_base_dir ) :
            os.makedirs( reports_base_dir )

        for i, row in df.iterrows() :
            try :
                score = row[fraud_score_column]
                if score < threshold :
                    continue  # Skip low-risk transactions

                risk_level = "High Risk" if score > 0.7 else "Suspicious"
                txn_data = row.to_dict()

                # Create safe filename
                safe_risk_level = risk_level.replace( ' ', '_' ).replace( '/', '_' )
                pdf_filename = f"{output_dir}/txn_{i}_{safe_risk_level}.pdf"

                pdf_path = generate_fraud_report(
                    txn_data=txn_data,
                    fraud_score=score,
                    risk_level=risk_level,
                    save_path=pdf_filename
                )

                if pdf_path and os.path.exists( pdf_path ) :
                    pdf_paths.append( pdf_path )

            except Exception as e :
                print( f"❌ Error generating report for transaction {i}: {e}" )
                continue

        # Create zip file
        if pdf_paths :
            with zipfile.ZipFile( zip_path, 'w', zipfile.ZIP_DEFLATED ) as zipf :
                for path in pdf_paths :
                    if os.path.exists( path ) :
                        # Add file to zip with just the filename, not full path
                        arcname = os.path.basename( path )
                        zipf.write( path, arcname )

            print( f"✅ Created ZIP file with {len( pdf_paths )} reports: {zip_path}" )
        else :
            print( "❌ No PDF reports were generated successfully" )

        return zip_path, pdf_paths

    except Exception as e :
        print( f"❌ Error in batch report generation: {e}" )
        return None, []