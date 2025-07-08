import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid
from faker import Faker
import warnings
from typing import List, Dict, Optional

warnings.filterwarnings('ignore')
fake = Faker('en_IN')
np.random.seed(42)
random.seed(42)


class SyntheticDataGenerator:
    def __init__(self):
        self.transaction_types = ['UPI', 'CARD', 'WALLET', 'NEFT', 'IMPS']
        self.account_types = ['SAVINGS', 'CURRENT', 'SALARY', 'BUSINESS']
        self.device_types = ['MOBILE', 'WEB', 'ATM', 'POS']
        self.locations = ['MUMBAI', 'DELHI', 'BANGALORE', 'CHENNAI', 'KOLKATA', 'HYDERABAD', 'PUNE', 'AHMEDABAD']
        self.merchants = ['AMAZON', 'FLIPKART', 'SWIGGY', 'ZOMATO', 'PAYTM', 'GROCERY_STORE', 'PETROL_PUMP', 'RESTAURANT']
        self.fraud_types = ['STRUCTURING', 'ACCOUNT_TAKEOVER', 'CARD_FRAUD', 'IDENTITY_THEFT', 'MONEY_LAUNDERING', 'PHISHING', 'ROUND_TRIPPING']
        self.suspicious_keywords = ['KUMAR', 'SINGH', 'SHARMA', 'GUPTA', 'AGARWAL', 'JAIN', 'PATEL', 'SHAH']
        self.banks = ['SBI', 'HDFC', 'ICICI', 'AXIS', 'PNB', 'BOB', 'CANARA', 'UNION', 'KOTAK', 'IDFC']
        self.channel_limits = {
            'UPI': 25000,
            'CARD': 100000,
            'WALLET': 10000,
            'NEFT': 200000,
            'IMPS': 200000
        }
        self.users = self._generate_users(3000)
        self.history = {user['user_id']: [] for user in self.users}

    def _generate_users(self, n: int) -> List[Dict]:
        users = []
        for i in range(n):
            name = fake.name()
            if random.random() < 0.05:
                suspicious_pattern = random.choice(self.suspicious_keywords)
                name = f"{fake.first_name()} {suspicious_pattern}"
            upi = f"{fake.user_name()}@{random.choice(self.banks).lower()}"
            user = {
                'user_id': f"USER_{i:05d}",
                'name': name,
                'upi_handle': upi,
                'age': np.random.randint(18, 80),
                'account_type': random.choice(self.account_types),
                'balance': round(np.random.uniform(10000, 500000), 2),
                'location': random.choice(self.locations),
                'risk_profile': np.random.choice(['LOW', 'MEDIUM', 'HIGH'], p=[0.8, 0.15, 0.05]),
                'preferred_txn_type': random.choice(self.transaction_types),
                'is_suspicious_name': int(any(keyword in name.upper() for keyword in self.suspicious_keywords))
            }
            users.append(user)
        return users

    def _apply_txn_limits(self, txn_type: str, amount: float) -> float:
        limits = {
            'UPI': (10, 100000), 'CARD': (100, 200000), 'WALLET': (10, 10000),
            'NEFT': (1000, 500000), 'IMPS': (1000, 500000)
        }
        min_amt, max_amt = limits.get(txn_type, (10, 500000))
        return round(min(max(amount, min_amt), max_amt), 2)

    def _velocity_features(self, user_id: str, timestamp: datetime):
        txns = self.history[user_id]
        last_hour = sum(1 for t in txns if (timestamp - t['timestamp']).total_seconds() <= 3600)
        last_day = sum(1 for t in txns if (timestamp - t['timestamp']).days == 0)
        last_week = sum(1 for t in txns if (timestamp - t['timestamp']).days <= 7)
        return last_hour, last_day, last_week

    def _generate_transaction(self, sender: Dict, recipient: Dict, timestamp: datetime, amount: float, txn_type: str,
                              is_fraud: int = 0, fraud_type: Optional[str] = None,
                              is_round_trip: int = 0, chain_id: Optional[int] = None, chain_pos: Optional[int] = None):
        if sender['balance'] < amount:
            return None

        txns_last_hour, txns_last_day, txns_last_week = self._velocity_features(sender['user_id'], timestamp)
        merchant = random.choice(self.merchants + [None]) if txn_type in ['UPI', 'CARD', 'WALLET'] else None

        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'timestamp': timestamp,
            'sender_id': sender['user_id'],
            'sender_name': sender['name'],
            'sender_upi_handle': sender['upi_handle'],
            'sender_account_type': sender['account_type'],
            'sender_balance_before': sender['balance'],
            'sender_age': sender['age'],
            'sender_risk_profile': sender['risk_profile'],
            'sender_suspicious_name': sender['is_suspicious_name'],
            'recipient_id': recipient['user_id'],
            'recipient_name': recipient['name'],
            'recipient_upi_handle': recipient['upi_handle'],
            'recipient_account_type': recipient['account_type'],
            'recipient_balance_before': recipient['balance'],
            'recipient_suspicious_name': recipient['is_suspicious_name'],
            'amount': amount,
            'transaction_type': txn_type,
            'device_type': random.choice(self.device_types),
            'location': sender['location'],
            'merchant_category': merchant,
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': int(timestamp.weekday() >= 5),
            'txns_last_hour': txns_last_hour,
            'txns_last_day': txns_last_day,
            'txns_last_week': txns_last_week,
            'amount_to_balance_ratio': round(amount / sender['balance'], 3) if sender['balance'] > 0 else 0,
            'amount_vs_channel_limit_ratio': round(amount / self.channel_limits.get(txn_type, 50000), 3),
            'is_round_amount': int(amount % 100 == 0),
            'is_impossible_travel': 0,
            'is_round_trip': is_round_trip,
            'round_trip_chain_id': chain_id,
            'round_trip_position': chain_pos,
            'is_fraud': is_fraud,
            'fraud_type': fraud_type
        }

        sender['balance'] -= amount
        recipient['balance'] += amount
        self.history[sender['user_id']].append(transaction)
        return transaction

    def generate_dataset(self, n_txns=100000, fraud_rate=0.03) -> pd.DataFrame:
        print(f"Generating {n_txns} transactions (fraud rate {fraud_rate*100:.1f}%)")
        n_fraud = int(n_txns * fraud_rate)
        n_normal = n_txns - n_fraud

        transactions = []

        # Normal
        for _ in range(n_normal):
            timestamp = fake.date_time_between(start_date='-90d', end_date='now')
            sender = random.choice(self.users)
            recipient = random.choice(self.users)
            while recipient['user_id'] == sender['user_id']:
                recipient = random.choice(self.users)
            txn_type = sender['preferred_txn_type']
            amount = np.random.uniform(500, 20000)
            amount = self._apply_txn_limits(txn_type, amount)
            txn = self._generate_transaction(sender, recipient, timestamp, amount, txn_type)
            if txn:
                transactions.append(txn)

        # Fraud
        for _ in range(n_fraud):
            timestamp = fake.date_time_between(start_date='-90d', end_date='now')
            fraud_type = random.choice([f for f in self.fraud_types if f != 'ROUND_TRIPPING'])
            sender = random.choice(self.users)
            recipient = random.choice(self.users)
            while recipient['user_id'] == sender['user_id']:
                recipient = random.choice(self.users)
            txn_type = random.choice(self.transaction_types)

            # Channel-specific suspicious thresholds
            if txn_type == 'UPI':
                amount = np.random.uniform(25000, 60000)
            elif txn_type == 'CARD':
                amount = np.random.uniform(80000, 200000)
            elif txn_type == 'WALLET':
                amount = np.random.uniform(8000, 15000)
            elif txn_type in ['NEFT', 'IMPS']:
                amount = np.random.uniform(150000, 500000)
            else:
                amount = np.random.uniform(20000, 100000)

            amount = self._apply_txn_limits(txn_type, amount)
            txn = self._generate_transaction(sender, recipient, timestamp, amount, txn_type, is_fraud=1, fraud_type=fraud_type)
            if txn:
                transactions.append(txn)

        df = pd.DataFrame(transactions)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"[✓] Generated {len(df)} transactions")
        print(f"[✓] Fraudulent: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.2%})")
        return df


# ----------- Run & Save ----------- #
if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    dataset = generator.generate_dataset(n_txns=100000, fraud_rate=0.03)
    dataset.to_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/synthetic_dataset_large.csv", index=False)
    print("✅ Dataset saved to 'synthetic_dataset_large.csv'")
