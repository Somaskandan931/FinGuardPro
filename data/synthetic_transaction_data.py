import pandas as pd
import numpy as np
import random
from datetime import datetime
import uuid
from faker import Faker
import warnings

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
        self.fraud_types = ['STRUCTURING', 'ACCOUNT_TAKEOVER', 'CARD_FRAUD', 'IDENTITY_THEFT', 'MONEY_LAUNDERING', 'PHISHING']
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

    def _generate_users(self, n):
        users = []
        for i in range(n):
            name = fake.name()
            upi = f"{fake.user_name()}@{random.choice(self.banks).lower()}"
            users.append({
                'user_id': f"USER_{i:05d}",
                'name': name,
                'upi_handle': upi,
                'age': np.random.randint(18, 80),
                'account_type': random.choice(self.account_types),
                'balance': round(np.random.uniform(10000, 500000), 2),
                'location': random.choice(self.locations),
                'risk_profile': np.random.choice(['LOW', 'MEDIUM', 'HIGH'], p=[0.8, 0.15, 0.05]),
                'preferred_txn_type': random.choice(self.transaction_types)
            })
        return users

    def _velocity_features(self, user_id, timestamp):
        txns = self.history[user_id]
        last_hour = sum(1 for t in txns if (timestamp - t['timestamp']).total_seconds() <= 3600)
        last_day = sum(1 for t in txns if (timestamp - t['timestamp']).days == 0)
        last_week = sum(1 for t in txns if (timestamp - t['timestamp']).days <= 7)
        return last_hour, last_day, last_week

    def _amount_to_avg_ratio(self, user_id, amount):
        txns = self.history[user_id]
        # Filter transactions that have 'amount' key
        amounts = [t['amount'] for t in txns if 'amount' in t]
        if not amounts:
            return 1.0
        avg_amt = np.mean(amounts)
        return round(amount / avg_amt, 3) if avg_amt > 0 else 1.0

    def _apply_txn_limits(self, txn_type, amount):
        min_amt, max_amt = {
            'UPI': (10, 100000), 'CARD': (100, 200000), 'WALLET': (10, 10000),
            'NEFT': (1000, 500000), 'IMPS': (1000, 500000)
        }.get(txn_type, (10, 500000))
        return round(min(max(amount, min_amt), max_amt), 2)

    def _generate_transaction(self, sender, recipient, timestamp, amount, txn_type, is_fraud=0, fraud_type=None):
        if sender['balance'] < amount:
            return None

        txns_last_hour, txns_last_day, txns_last_week = self._velocity_features(sender['user_id'], timestamp)
        merchant = random.choice(self.merchants + [None]) if txn_type in ['UPI', 'CARD', 'WALLET'] else None

        txn = {
            'timestamp': timestamp,
            'sender_balance_before': sender['balance'],
            'sender_age': sender['age'],
            'recipient_balance_before': recipient['balance'],
            'amount': amount,  # unified key for amount
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': int(timestamp.weekday() >= 5),
            'txns_last_hour': txns_last_hour,
            'txns_last_day': txns_last_day,
            'txns_last_week': txns_last_week,
            'amount_to_balance_ratio': round(amount / sender['balance'], 3) if sender['balance'] > 0 else 0,
            'amount_vs_channel_limit_ratio': round(amount / self.channel_limits.get(txn_type, 50000), 3),
            'is_round_amount': int(amount % 100 == 0),
            'log_amount': round(np.log1p(amount), 3),
            'is_high_value': int(amount > 100000),
            'is_new_receiver': int(len([t for t in self.history[sender['user_id']] if t.get('recipient_id') == recipient['user_id']]) == 0),
            'sender_txn_count': len(self.history[sender['user_id']]),
            'amount_to_avg_ratio': self._amount_to_avg_ratio(sender['user_id'], amount),
            'sender_account_type': sender['account_type'],
            'sender_risk_profile': sender['risk_profile'],
            'recipient_account_type': recipient['account_type'],
            'transaction_type': txn_type,
            'device_type': random.choice(self.device_types),
            'location': sender['location'],
            'merchant_category': merchant,
            'is_fraud': is_fraud,
            'recipient_id': recipient['user_id']  # Needed for is_new_receiver tracking
        }

        sender['balance'] -= amount
        recipient['balance'] += amount
        self.history[sender['user_id']].append(txn)
        return txn

    def generate_dataset(self, n_txns=100000, fraud_rate=0.03):
        print(f"Generating {n_txns} transactions ({fraud_rate*100:.1f}% fraud)")
        n_fraud = int(n_txns * fraud_rate)
        n_normal = n_txns - n_fraud
        transactions = []

        # Normal Transactions
        for _ in range(n_normal):
            timestamp = fake.date_time_between(start_date='-90d', end_date='now')
            sender = random.choice(self.users)
            recipient = random.choice([u for u in self.users if u['user_id'] != sender['user_id']])
            txn_type = sender['preferred_txn_type']
            amount = self._apply_txn_limits(txn_type, np.random.uniform(500, 20000))
            txn = self._generate_transaction(sender, recipient, timestamp, amount, txn_type)
            if txn:
                transactions.append(txn)

        # Fraudulent Transactions
        for _ in range(n_fraud):
            timestamp = fake.date_time_between(start_date='-90d', end_date='now')
            sender = random.choice(self.users)
            recipient = random.choice([u for u in self.users if u['user_id'] != sender['user_id']])
            txn_type = random.choice(self.transaction_types)
            amount = {
                'UPI': np.random.uniform(25000, 60000),
                'CARD': np.random.uniform(80000, 200000),
                'WALLET': np.random.uniform(8000, 15000),
                'NEFT': np.random.uniform(150000, 500000),
                'IMPS': np.random.uniform(150000, 500000)
            }.get(txn_type, np.random.uniform(20000, 100000))

            amount = self._apply_txn_limits(txn_type, amount)
            txn = self._generate_transaction(sender, recipient, timestamp, amount, txn_type, is_fraud=1)
            if txn:
                transactions.append(txn)

        df = pd.DataFrame(transactions)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"[✓] Generated {len(df)} transactions")
        print(f"[✓] Fraudulent: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.2%})")
        return df


# ---------- Run and Save ---------- #
if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    df = generator.generate_dataset(n_txns=100000, fraud_rate=0.03)
    output_path = "//data/synthetic_dataset_large.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset saved to {output_path}")
