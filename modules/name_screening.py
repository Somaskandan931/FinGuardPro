import pandas as pd
from rapidfuzz import fuzz, process
from sqlalchemy import create_engine, text
import logging
import os
from typing import Optional, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PostgreSQL configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'finguard_db'),
    'user': os.getenv('DB_USER', 'finguard_user'),
    'password': os.getenv('DB_PASSWORD', 'Rsomas123**'),  # use env in production
    'port': int(os.getenv('DB_PORT', 5432))
}


def get_database_connection():
    try:
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Connected to PostgreSQL database.")
        return engine
    except Exception as e:
        logger.error(f"❌ DB connection failed: {e}")
        return None


def load_watchlist_from_db() -> pd.DataFrame:
    engine = None
    try:
        engine = get_database_connection()
        if engine is None:
            return pd.DataFrame()

        query = """
        SELECT id, name, aliases, entity_type, risk_category, risk_score, country,
               sanctions_list, status, last_updated
        FROM aml_watchlist
        WHERE status = 'Active'
        ORDER BY risk_score DESC
        """

        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)

        if df.empty:
            logger.warning("⚠️ No active watchlist entries found.")
            return pd.DataFrame()

        # 🔧 Normalize names
        df["name"] = df["name"].astype(str).str.strip().str.lower()

        # 🔧 Process aliases into new rows
        aliases_rows = []
        for _, row in df.iterrows():
            if pd.notna(row["aliases"]):
                aliases = str(row["aliases"]).split(';')
                for alias in aliases:
                    alias = alias.strip().lower()
                    if alias:
                        alias_row = row.copy()
                        alias_row["name"] = alias
                        aliases_rows.append(alias_row)

        if aliases_rows:
            df = pd.concat([df, pd.DataFrame(aliases_rows)], ignore_index=True)

        df = df.drop_duplicates(subset=["name"])
        logger.info(f"✅ Loaded {len(df)} watchlist entries (including aliases).")
        return df

    except Exception as e:
        logger.error(f"❌ Failed to load watchlist: {e}")
        return pd.DataFrame()
    finally:
        if engine:
            engine.dispose()


def check_name_match(name: str, watchlist_df: pd.DataFrame, threshold: int = 85) -> Tuple[
    Optional[str], int, Optional[str], Optional[str], Optional[str], Optional[float]
]:
    if not name.strip() or watchlist_df.empty:
        return None, 0, None, None, None, None

    try:
        name = name.strip().lower()
        valid_names = watchlist_df["name"].dropna().str.strip()
        match = process.extractOne(name, valid_names.tolist(), scorer=fuzz.token_sort_ratio)

        if match and match[1] >= threshold:
            # 🔧 Use .iloc safely in case of duplicates
            matched_entries = watchlist_df[watchlist_df["name"] == match[0]]
            if not matched_entries.empty:
                matched_entry = matched_entries.iloc[0]
                return (
                    match[0], match[1], matched_entry["id"],
                    matched_entry["entity_type"],
                    str(matched_entry["risk_category"]).strip(),
                    matched_entry["risk_score"]
                )
    except Exception as e:
        logger.error(f"❌ Error matching name '{name}': {e}")

    return None, 0, None, None, None, None


def run_name_screening(transactions_df: pd.DataFrame, threshold: int = 85) -> Dict[str, Any]:
    logger.info("🔍 Starting structured name screening...")

    watchlist_df = load_watchlist_from_db()
    if transactions_df.empty:
        return {"error": "Empty input."}

    row = transactions_df.iloc[0]

    sender_name = str(row.get("sender_name", "")).strip()
    recipient_name = str(row.get("recipient_name", "")).strip()

    # 🔧 Match both names
    sender_match, sender_score, sender_id, sender_entity_type, sender_risk_category, sender_risk_score = check_name_match(sender_name, watchlist_df, threshold)
    recipient_match, recipient_score, recipient_id, recipient_entity_type, recipient_risk_category, recipient_risk_score = check_name_match(recipient_name, watchlist_df, threshold)

    # 🔧 Normalize categories for safe comparison
    sender_cat = (sender_risk_category or "").lower()
    recipient_cat = (recipient_risk_category or "").lower()

    sender_result = {
        "name": sender_name,
        "risk_score": sender_risk_score or 0,
        "sanctions_hit": "sanctions" in sender_cat,
        "pep_hit": "pep" in sender_cat,
        "matches_found": bool(sender_match),
        "match_details": [
            f"Match: {sender_match}, Score: {sender_score}, Category: {sender_risk_category}"
        ] if sender_match else []
    }

    recipient_result = {
        "name": recipient_name,
        "risk_score": recipient_risk_score or 0,
        "sanctions_hit": "sanctions" in recipient_cat,
        "pep_hit": "pep" in recipient_cat,
        "matches_found": bool(recipient_match),
        "match_details": [
            f"Match: {recipient_match}, Score: {recipient_score}, Category: {recipient_risk_category}"
        ] if recipient_match else []
    }

    max_risk = max(sender_risk_score or 0, recipient_risk_score or 0)

    if max_risk >= 80:
        overall_risk = "High"
        recommended_action = "Block"
    elif max_risk >= 50:
        overall_risk = "Medium"
        recommended_action = "Review"
    else:
        overall_risk = "Low"
        recommended_action = "Approve"

    logger.info(f"🏁 Screening done. Overall risk: {overall_risk}, Action: {recommended_action}")

    return {
        "transaction_id": row.get("transaction_id"),
        "sender_screening": sender_result,
        "recipient_screening": recipient_result,
        "overall_risk": overall_risk,
        "recommended_action": recommended_action
    }
