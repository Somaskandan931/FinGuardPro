import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import uuid
import os
from sqlalchemy import create_engine, text, Column, String, Integer, Float, Boolean, DateTime, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import psycopg2

# Database configuration
DATABASE_CONFIG = {
    'host' : 'localhost',
    'port' : 5432,
    'database' : 'finguard_db',
    'user' : 'finguard_user',  # or 'postgres' if using superuser
    'password' : 'Rsomas123**'  # Replace with your actual password
}

Base = declarative_base()


class Transaction( Base ) :
    __tablename__ = 'transactions'

    transaction_id = Column( String( 50 ), primary_key=True )
    timestamp = Column( DateTime, nullable=False )
    amount = Column( DECIMAL( 15, 2 ), nullable=False )
    sender_id = Column( String( 50 ), nullable=False )
    recipient_id = Column( String( 50 ), nullable=False )
    sender_account_type = Column( String( 50 ), nullable=False )
    recipient_account_type = Column( String( 50 ), nullable=False )
    device_id = Column( String( 50 ), nullable=True )
    location_ip = Column( String( 45 ), nullable=True )  # IPv4/IPv6
    transaction_type = Column( String( 50 ), nullable=False )
    is_fraud = Column( Integer, nullable=False )
    sender_name = Column( String( 255 ), nullable=False )
    recipient_name = Column( String( 255 ), nullable=False )
    sender_balance_before = Column( DECIMAL( 15, 2 ), nullable=False )
    recipient_balance_before = Column( DECIMAL( 15, 2 ), nullable=False )
    merchant_category_code = Column( Integer, nullable=True )
    created_at = Column( DateTime, default=datetime.utcnow )


class DatabaseManager :
    def __init__ ( self, config ) :
        self.config = config
        self.engine = None
        self.session = None
        self.connect()

    def connect ( self ) :
        """Establish database connection"""
        try :
            # Create connection string
            db_url = f"postgresql://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}"

            # Create engine
            self.engine = create_engine( db_url, echo=False )

            # Test connection
            with self.engine.connect() as conn :
                conn.execute( text( "SELECT 1" ) )

            print( "✅ Database connection successful!" )

            # Create session
            Session = sessionmaker( bind=self.engine )
            self.session = Session()

        except Exception as e :
            print( f"❌ Database connection failed: {e}" )
            print( "\nTroubleshooting tips:" )
            print( "1. Check if PostgreSQL is running" )
            print( "2. Verify database name, username, and password" )
            print( "3. Ensure the database exists" )
            print( "4. Check if the user has proper permissions" )
            raise

    def create_tables ( self ) :
        """Create database tables"""
        try :
            # Create all tables
            Base.metadata.create_all( self.engine )

            # Create indexes for better performance
            with self.engine.connect() as conn :
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);",
                    "CREATE INDEX IF NOT EXISTS idx_transactions_sender ON transactions(sender_id);",
                    "CREATE INDEX IF NOT EXISTS idx_transactions_recipient ON transactions(recipient_id);",
                    "CREATE INDEX IF NOT EXISTS idx_transactions_fraud ON transactions(is_fraud);",
                    "CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(amount);",
                    "CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(transaction_type);",
                    "CREATE INDEX IF NOT EXISTS idx_transactions_device ON transactions(device_id);"
                ]

                for index_sql in indexes :
                    conn.execute( text( index_sql ) )

                conn.commit()

            print( "✅ Database tables and indexes created successfully!" )

        except Exception as e :
            print( f"❌ Error creating tables: {e}" )
            raise

    def insert_transactions_from_csv ( self, csv_path ) :
        """Insert transactions from CSV file"""
        try :
            # Read CSV
            df = pd.read_csv( csv_path )
            print( f"📊 Loading {len( df )} transactions from CSV..." )

            # Convert timestamp column to datetime
            df['timestamp'] = pd.to_datetime( df['timestamp'] )

            # Insert data using pandas to_sql (more efficient for large datasets)
            df.to_sql( 'transactions', self.engine, if_exists='append', index=False, method='multi' )

            print( f"✅ Successfully inserted {len( df )} transactions into database!" )

        except Exception as e :
            print( f"❌ Error inserting data: {e}" )
            raise

    def get_transaction_stats ( self ) :
        """Get transaction statistics"""
        try :
            with self.engine.connect() as conn :
                # Total transactions
                result = conn.execute( text( "SELECT COUNT(*) as total FROM transactions" ) )
                total = result.fetchone()[0]

                # Fraud statistics
                result = conn.execute(
                    text( "SELECT is_fraud, COUNT(*) as count FROM transactions GROUP BY is_fraud" ) )
                fraud_stats = dict( result.fetchall() )

                # Transaction type distribution
                result = conn.execute(
                    text( "SELECT transaction_type, COUNT(*) as count FROM transactions GROUP BY transaction_type" ) )
                type_stats = dict( result.fetchall() )

                # Amount statistics
                result = conn.execute( text( """
                    SELECT 
                        is_fraud,
                        COUNT(*) as count,
                        AVG(amount) as avg_amount,
                        MIN(amount) as min_amount,
                        MAX(amount) as max_amount
                    FROM transactions 
                    GROUP BY is_fraud
                """ ) )
                amount_stats = result.fetchall()

                return {
                    'total_transactions' : total,
                    'fraud_stats' : fraud_stats,
                    'type_stats' : type_stats,
                    'amount_stats' : amount_stats
                }

        except Exception as e :
            print( f"❌ Error getting statistics: {e}" )
            return None

    def execute_query ( self, query ) :
        """Execute custom SQL query"""
        try :
            with self.engine.connect() as conn :
                result = conn.execute( text( query ) )
                return result.fetchall()
        except Exception as e :
            print( f"❌ Error executing query: {e}" )
            return None

    def close ( self ) :
        """Close database connection"""
        if self.session :
            self.session.close()
        if self.engine :
            self.engine.dispose()


# Enhanced Transaction Data Generator with PostgreSQL integration
class TransactionDataGenerator :
    def __init__ ( self, num_users=10000, num_transactions=100000, use_database=False ) :
        self.num_users = num_users
        self.num_transactions = num_transactions
        self.data_dir = r"/\data"
        self.use_database = use_database

        # Create directory if it doesn't exist
        os.makedirs( self.data_dir, exist_ok=True )

        # Initialize database if requested
        self.db_manager = None
        if use_database :
            self.db_manager = DatabaseManager( DATABASE_CONFIG )
            self.db_manager.create_tables()

        # Initialize Faker
        self.fake = Faker( 'en_IN' )
        np.random.seed( 42 )
        random.seed( 42 )

        self.users = self._generate_users()
        self.merchants = self._generate_merchants()
        self.devices = self._generate_devices()

    def _generate_users ( self ) :
        """Generate synthetic user profiles"""
        users = []
        for i in range( self.num_users ) :
            user = {
                'user_id' : f'USER_{i + 1:06d}',
                'name' : self.fake.name(),
                'account_type' : random.choice( ['UPI', 'Wallet', 'Credit Card', 'Debit Card'] ),
                'is_fraudster' : random.random() < 0.02  # 2% of users are fraudsters
            }
            users.append( user )
        return users

    def _generate_merchants ( self ) :
        """Generate merchant profiles"""
        merchants = []
        categories = ['Grocery', 'Fuel', 'Restaurant', 'Entertainment', 'Shopping', 'Healthcare', 'Education',
                      'Utilities']
        for i in range( 1000 ) :
            merchant = {
                'merchant_id' : f'MERCHANT_{i + 1:04d}',
                'name' : self.fake.company(),
                'category' : random.choice( categories ),
                'merchant_category_code' : random.randint( 1000, 9999 ),
                'account_type' : random.choice( ['UPI', 'Wallet', 'Bank Account'] )
            }
            merchants.append( merchant )
        return merchants

    def _generate_devices ( self ) :
        """Generate device profiles"""
        devices = []
        for i in range( 5000 ) :
            device = {
                'device_id' : f'DEVICE_{i + 1:05d}',
                'is_compromised' : random.random() < 0.001  # 0.1% devices are compromised
            }
            devices.append( device )
        return devices

    def _generate_ip_address ( self ) :
        """Generate realistic IP addresses"""
        return self.fake.ipv4()

    def _generate_legitimate_transaction ( self ) :
        """Generate a legitimate transaction"""
        sender = random.choice( self.users )
        transaction_type = random.choice( ['P2P', 'Merchant Payment', 'Bill Pay', 'ATM Withdrawal'] )

        if transaction_type == 'P2P' :
            recipient = random.choice( [u for u in self.users if u['user_id'] != sender['user_id']] )
            amount = round( random.uniform( 10, 5000 ), 2 )
            recipient_name = recipient['name']
            recipient_id = recipient['user_id']
            recipient_account_type = recipient['account_type']
            merchant_category_code = None
            recipient_balance_before = round( random.uniform( 1000, 50000 ), 2 )

        elif transaction_type == 'Merchant Payment' :
            merchant = random.choice( self.merchants )
            amount = round( random.uniform( 50, 10000 ), 2 )
            recipient_name = merchant['name']
            recipient_id = merchant['merchant_id']
            recipient_account_type = merchant['account_type']
            merchant_category_code = merchant['merchant_category_code']
            recipient_balance_before = round( random.uniform( 10000, 200000 ), 2 )

        else :
            amount = round( random.uniform( 100, 20000 ), 2 )
            recipient_name = self.fake.company()
            recipient_id = f"BILL_{random.randint( 1000, 9999 )}"
            recipient_account_type = random.choice( ['UPI', 'Wallet', 'Bank Account'] )
            merchant_category_code = random.randint( 1000, 9999 )
            recipient_balance_before = round( random.uniform( 5000, 100000 ), 2 )

        # Generate timestamp
        base_time = datetime.now() - timedelta( days=random.randint( 0, 365 ) )
        if random.random() < 0.8 :
            hour = random.randint( 8, 22 )
        else :
            hour = random.randint( 0, 7 )

        timestamp = base_time.replace( hour=hour, minute=random.randint( 0, 59 ), second=random.randint( 0, 59 ) )
        device = random.choice( self.devices )

        return {
            'transaction_id' : str( uuid.uuid4() ),
            'timestamp' : timestamp,
            'amount' : amount,
            'sender_id' : sender['user_id'],
            'recipient_id' : recipient_id,
            'sender_account_type' : sender['account_type'],
            'recipient_account_type' : recipient_account_type,
            'device_id' : device['device_id'],
            'location_ip' : self._generate_ip_address(),
            'transaction_type' : transaction_type,
            'is_fraud' : 0,
            'sender_name' : sender['name'],
            'recipient_name' : recipient_name,
            'sender_balance_before' : round( random.uniform( 1000, 50000 ), 2 ),
            'recipient_balance_before' : recipient_balance_before,
            'merchant_category_code' : merchant_category_code
        }

    def _generate_fraudulent_transaction ( self ) :
        """Generate a fraudulent transaction"""
        fraud_type = random.choice( ['stolen_identity', 'structuring', 'unusual_activity', 'high_value'] )

        if fraud_type == 'stolen_identity' :
            fraudster = random.choice( [u for u in self.users if u['is_fraudster']] )
            victim = random.choice( [u for u in self.users if not u['is_fraudster']] )
            transaction = self._generate_legitimate_transaction()
            transaction['sender_id'] = victim['user_id']
            transaction['sender_name'] = victim['name']
            transaction['sender_account_type'] = victim['account_type']
            transaction['amount'] = round( random.uniform( 5000, 50000 ), 2 )
            transaction['timestamp'] = datetime.now() - timedelta( days=random.randint( 0, 30 ) )
            transaction['timestamp'] = transaction['timestamp'].replace( hour=random.randint( 0, 6 ) )

        elif fraud_type == 'structuring' :
            sender = random.choice( [u for u in self.users if u['is_fraudster']] )
            transaction = self._generate_legitimate_transaction()
            transaction['sender_id'] = sender['user_id']
            transaction['sender_name'] = sender['name']
            transaction['sender_account_type'] = sender['account_type']
            transaction['amount'] = round( random.uniform( 199900, 199999 ), 2 )

        elif fraud_type == 'unusual_activity' :
            sender = random.choice( [u for u in self.users if u['is_fraudster']] )
            transaction = self._generate_legitimate_transaction()
            transaction['sender_id'] = sender['user_id']
            transaction['sender_name'] = sender['name']
            transaction['sender_account_type'] = sender['account_type']
            transaction['amount'] = round( random.uniform( 1000, 20000 ), 2 )
            transaction['timestamp'] = datetime.now() - timedelta( days=random.randint( 0, 30 ) )
            transaction['timestamp'] = transaction['timestamp'].replace( hour=3 )

        else :  # high_value
            sender = random.choice( [u for u in self.users if u['is_fraudster']] )
            transaction = self._generate_legitimate_transaction()
            transaction['sender_id'] = sender['user_id']
            transaction['sender_name'] = sender['name']
            transaction['sender_account_type'] = sender['account_type']
            transaction['amount'] = round( random.uniform( 100000, 500000 ), 2 )

        transaction['is_fraud'] = 1
        return transaction

    def generate_dataset ( self, fraud_ratio=0.1 ) :
        """Generate complete dataset"""
        transactions = []
        num_fraud = int( self.num_transactions * fraud_ratio )
        num_legitimate = self.num_transactions - num_fraud

        print( f"🔄 Generating {num_legitimate} legitimate transactions..." )
        for i in range( num_legitimate ) :
            if i % 10000 == 0 :
                print( f"Progress: {i}/{num_legitimate}" )
            transactions.append( self._generate_legitimate_transaction() )

        print( f"🔄 Generating {num_fraud} fraudulent transactions..." )
        for i in range( num_fraud ) :
            if i % 1000 == 0 :
                print( f"Progress: {i}/{num_fraud}" )
            transactions.append( self._generate_fraudulent_transaction() )

        random.shuffle( transactions )
        df = pd.DataFrame( transactions )

        # Ensure column order
        columns_order = [
            'transaction_id', 'timestamp', 'amount', 'sender_id', 'recipient_id',
            'sender_account_type', 'recipient_account_type', 'device_id', 'location_ip',
            'transaction_type', 'is_fraud', 'sender_name', 'recipient_name',
            'sender_balance_before', 'recipient_balance_before', 'merchant_category_code'
        ]
        df = df[columns_order]

        # Save to CSV
        csv_path = os.path.join( self.data_dir, 'synthetic_transaction_data.csv' )
        df.to_csv( csv_path, index=False )
        print( f"💾 Dataset saved to: {csv_path}" )

        # Save to database if enabled
        if self.use_database and self.db_manager :
            print( "🔄 Saving to database..." )
            self.db_manager.insert_transactions_from_csv( csv_path )

        return df


def main () :
    """Main function to demonstrate usage"""
    print( "🚀 FinGuardPro - Transaction Data Generator with PostgreSQL" )
    print( "=" * 60 )

    # Ask user if they want to use database
    use_db = input( "Do you want to use PostgreSQL database? (y/n): " ).lower().strip() == 'y'

    if use_db :
        print( "\n🔧 Setting up database connection..." )
        try :
            # Test database connection
            db_manager = DatabaseManager( DATABASE_CONFIG )
            db_manager.create_tables()
            print( "✅ Database setup successful!" )
            db_manager.close()
        except Exception as e :
            print( f"❌ Database setup failed: {e}" )
            print( "Proceeding with CSV-only mode..." )
            use_db = False

    # Generate data
    print( f"\n📊 Generating synthetic transaction data..." )
    generator = TransactionDataGenerator(
        num_users=5000,
        num_transactions=50000,
        use_database=use_db
    )

    df = generator.generate_dataset( fraud_ratio=0.1 )

    # Display statistics
    print( "\n📈 Dataset Statistics:" )
    print( f"Total transactions: {len( df ):,}" )
    print( f"Fraud transactions: {df['is_fraud'].sum():,}" )
    print( f"Legitimate transactions: {(len( df ) - df['is_fraud'].sum()):,}" )
    print( f"Fraud rate: {df['is_fraud'].mean():.2%}" )

    # Database statistics
    if use_db and generator.db_manager :
        print( "\n🗄️ Database Statistics:" )
        stats = generator.db_manager.get_transaction_stats()
        if stats :
            print( f"Total in database: {stats['total_transactions']:,}" )
            print( f"Fraud in database: {stats['fraud_stats'].get( 1, 0 ):,}" )
            print( f"Legitimate in database: {stats['fraud_stats'].get( 0, 0 ):,}" )
        generator.db_manager.close()

    print( "\n✅ Data generation complete!" )
    print( f"📂 CSV file location: {generator.data_dir}" )
    if use_db :
        print( "🗄️ Data also available in PostgreSQL database" )


if __name__ == "__main__" :
    main()