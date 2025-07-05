import pandas as pd
import random
import csv
import os
from datetime import datetime, timedelta
from faker import Faker
import numpy as np
import psycopg2
from sqlalchemy import create_engine, text
import logging

# Initialize Faker for generating realistic names
fake = Faker()

# Configure logging
logging.basicConfig( level=logging.INFO )
logger = logging.getLogger( __name__ )

# Directory configuration
DATA_DIR = r"C:\Users\somas\PycharmProjects\FinGuard Pro\data"

# PostgreSQL configuration
DB_CONFIG = {
    'host' : 'localhost',
    'database' : 'finguard_db',
    'user' : 'finguard_user',
    'password' : 'Rsomas123**',  # Change this to your PostgreSQL password
    'port' : 5432
}


def ensure_data_directory () :
    """Ensure the data directory exists"""
    if not os.path.exists( DATA_DIR ) :
        os.makedirs( DATA_DIR )
        logger.info( f"Created directory: {DATA_DIR}" )


def create_postgresql_tables () :
    """Create PostgreSQL tables for AML watchlist and transactions"""
    try :
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}" )

        # Create AML watchlist table
        watchlist_table_sql = """
        CREATE TABLE IF NOT EXISTS aml_watchlist (
            id VARCHAR(20) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            aliases TEXT,
            entity_type VARCHAR(50),
            risk_category VARCHAR(100),
            risk_score DECIMAL(3,2),
            country VARCHAR(100),
            date_of_birth DATE,
            passport_number VARCHAR(50),
            national_id VARCHAR(50),
            address TEXT,
            phone_number VARCHAR(50),
            email VARCHAR(100),
            designation VARCHAR(100),
            sanctions_list VARCHAR(20),
            list_date DATE,
            status VARCHAR(20),
            notes TEXT,
            last_updated DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        # Create transactions table
        transactions_table_sql = """
        CREATE TABLE IF NOT EXISTS sample_transactions (
            transaction_id VARCHAR(20) PRIMARY KEY,
            sender_name VARCHAR(255),
            receiver_name VARCHAR(255),
            amount DECIMAL(12,2),
            timestamp TIMESTAMP,
            payment_method VARCHAR(50),
            location VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        with engine.connect() as conn :
            conn.execute( text( watchlist_table_sql ) )
            conn.execute( text( transactions_table_sql ) )
            conn.commit()

        logger.info( "PostgreSQL tables created successfully" )
        return engine

    except Exception as e :
        logger.error( f"Error creating PostgreSQL tables: {e}" )
        return None


def upload_to_postgresql ( df, table_name, engine ) :
    """Upload DataFrame to PostgreSQL table"""
    try :
        # Convert date columns to proper format
        date_columns = ['date_of_birth', 'list_date', 'last_updated']
        for col in date_columns :
            if col in df.columns :
                df[col] = pd.to_datetime( df[col], errors='coerce' )

        # Handle timestamp column for transactions
        if 'timestamp' in df.columns :
            df['timestamp'] = pd.to_datetime( df['timestamp'], errors='coerce' )

        # Upload to PostgreSQL
        df.to_sql( table_name, engine, if_exists='replace', index=False, method='multi' )
        logger.info( f"Successfully uploaded {len( df )} records to {table_name} table" )

    except Exception as e :
        logger.error( f"Error uploading to PostgreSQL: {e}" )


def generate_synthetic_aml_watchlist ( num_entries=1000, output_file='aml_watchlist.csv' ) :
    """
    Generate synthetic AML watchlist data for testing fraud detection systems.

    Parameters:
    num_entries (int): Number of watchlist entries to generate
    output_file (str): Output CSV file name

    Returns:
    pandas.DataFrame: Generated watchlist data
    """

    # Define risk categories and types
    risk_categories = [
        'Terrorism Financing',
        'Drug Trafficking',
        'Money Laundering',
        'Sanctions Violation',
        'Politically Exposed Person (PEP)',
        'Organized Crime',
        'Corruption',
        'Tax Evasion',
        'Fraud',
        'Human Trafficking'
    ]

    entity_types = ['Individual', 'Organization', 'Vessel', 'Aircraft']

    countries = [
        'India', 'Pakistan', 'Afghanistan', 'Iran', 'Iraq', 'Syria', 'North Korea',
        'Russia', 'China', 'Myanmar', 'Somalia', 'Sudan', 'Yemen', 'Libya',
        'Venezuela', 'Belarus', 'Cuba', 'Nicaragua', 'Lebanon', 'Mali'
    ]

    # Lists for generating realistic suspicious names
    suspicious_prefixes = [
        'Al-', 'Abu-', 'Ibn-', 'Bin-', 'El-', 'Omar-', 'Hassan-', 'Ahmed-',
        'Mohammed-', 'Abdul-', 'Khalil-', 'Rashid-', 'Farid-', 'Tariq-'
    ]

    organization_suffixes = [
        'Foundation', 'Trust', 'Holdings', 'Group', 'Network', 'Alliance',
        'Coalition', 'Association', 'Council', 'Front', 'Liberation Army',
        'Movement', 'Organization', 'Syndicate', 'Cartel', 'Gang'
    ]

    watchlist_data = []

    for i in range( num_entries ) :
        entity_type = random.choice( entity_types )

        if entity_type == 'Individual' :
            # Generate individual names
            if random.random() < 0.3 :  # 30% chance of suspicious prefix
                first_name = random.choice( suspicious_prefixes ) + fake.first_name()
            else :
                first_name = fake.first_name()

            last_name = fake.last_name()
            full_name = f"{first_name} {last_name}"

            # Sometimes add aliases
            aliases = []
            if random.random() < 0.4 :  # 40% chance of having aliases
                num_aliases = random.randint( 1, 3 )
                for _ in range( num_aliases ) :
                    aliases.append( fake.name() )

        elif entity_type == 'Organization' :
            # Generate organization names
            base_name = fake.company()
            if random.random() < 0.5 :
                suffix = random.choice( organization_suffixes )
                full_name = f"{base_name} {suffix}"
            else :
                full_name = base_name

            aliases = []
            if random.random() < 0.3 :
                aliases.append( fake.company() )

        elif entity_type == 'Vessel' :
            full_name = f"MV {fake.first_name()} {fake.last_name()}"
            aliases = []

        else :  # Aircraft
            full_name = f"{fake.company()} {random.randint( 100, 999 )}"
            aliases = []

        # Generate other fields
        entry = {
            'id' : f"WL_{i + 1:06d}",
            'name' : full_name,
            'aliases' : '; '.join( aliases ) if aliases else '',
            'entity_type' : entity_type,
            'risk_category' : random.choice( risk_categories ),
            'risk_score' : round( random.uniform( 0.6, 1.0 ), 2 ),  # High risk entities
            'country' : random.choice( countries ),
            'date_of_birth' : fake.date_of_birth( minimum_age=25, maximum_age=80 ).strftime(
                '%Y-%m-%d' ) if entity_type == 'Individual' else '',
            'passport_number' : fake.passport_number() if entity_type == 'Individual' and random.random() < 0.6 else '',
            'national_id' : fake.ssn() if entity_type == 'Individual' and random.random() < 0.4 else '',
            'address' : fake.address().replace( '\n', ', ' ) if random.random() < 0.7 else '',
            'phone_number' : fake.phone_number() if random.random() < 0.5 else '',
            'email' : fake.email() if random.random() < 0.3 else '',
            'designation' : fake.job() if entity_type == 'Individual' and random.random() < 0.5 else '',
            'sanctions_list' : random.choice( ['UN', 'OFAC', 'EU', 'FATF', 'RBI', 'SEBI', 'ED'] ),
            'list_date' : fake.date_between( start_date='-10y', end_date='today' ).strftime( '%Y-%m-%d' ),
            'status' : random.choice( ['Active', 'Inactive', 'Under Review'] ) if random.random() < 0.9 else 'Active',
            'notes' : fake.text( max_nb_chars=200 ) if random.random() < 0.4 else '',
            'last_updated' : fake.date_between( start_date='-1y', end_date='today' ).strftime( '%Y-%m-%d' )
        }

        watchlist_data.append( entry )

    # Create DataFrame
    df = pd.DataFrame( watchlist_data )

    # Add some variations to make names more realistic for fuzzy matching testing
    variations_df = create_name_variations( df.head( 50 ) )  # Create variations for first 50 entries

    # Combine original and variations
    final_df = pd.concat( [df, variations_df], ignore_index=True )

    # Save to CSV
    final_df.to_csv( output_file, index=False )
    print( f"Generated {len( final_df )} AML watchlist entries and saved to {output_file}" )

    return final_df


def create_name_variations ( df ) :
    """
    Create name variations for testing fuzzy matching capabilities.
    This simulates real-world scenarios where names might have slight variations.
    """
    variations = []

    for _, row in df.iterrows() :
        original_name = row['name']

        # Create different types of variations
        variation_types = [
            add_typos,
            add_middle_names,
            change_name_order,
            add_prefixes_suffixes,
            transliteration_variations
        ]

        # Generate 1-3 variations per name
        num_variations = random.randint( 1, 3 )
        selected_variations = random.sample( variation_types, min( num_variations, len( variation_types ) ) )

        for variation_func in selected_variations :
            varied_name = variation_func( original_name )

            # Create new entry with varied name
            new_entry = row.copy()
            new_entry['id'] = f"{row['id']}_VAR_{len( variations ) + 1}"
            new_entry['name'] = varied_name
            new_entry['notes'] = f"Variation of {original_name}"

            variations.append( new_entry )

    return pd.DataFrame( variations )


def add_typos ( name ) :
    """Add common typos to names"""
    typos = {
        'a' : 'e', 'e' : 'a', 'i' : 'y', 'y' : 'i', 'o' : 'u', 'u' : 'o',
        'b' : 'p', 'p' : 'b', 'd' : 't', 't' : 'd', 'g' : 'k', 'k' : 'g',
        'v' : 'w', 'w' : 'v', 'n' : 'm', 'm' : 'n'
    }

    name_chars = list( name.lower() )
    if len( name_chars ) > 3 :
        # Change 1-2 characters
        positions = random.sample( range( len( name_chars ) ), min( 2, len( name_chars ) ) )
        for pos in positions :
            if name_chars[pos] in typos :
                name_chars[pos] = typos[name_chars[pos]]

    return ''.join( name_chars ).title()


def add_middle_names ( name ) :
    """Add middle names or initials"""
    parts = name.split()
    if len( parts ) >= 2 :
        middle_options = [
            fake.first_name(),
            fake.first_name()[0] + '.',
            'bin', 'ibn', 'al', 'el', 'abdul'
        ]
        middle = random.choice( middle_options )
        return f"{parts[0]} {middle} {' '.join( parts[1 :] )}"
    return name


def change_name_order ( name ) :
    """Change the order of name components"""
    parts = name.split()
    if len( parts ) >= 2 :
        random.shuffle( parts )
        return ' '.join( parts )
    return name


def add_prefixes_suffixes ( name ) :
    """Add titles or suffixes"""
    prefixes = ['Dr.', 'Mr.', 'Mrs.', 'Prof.', 'Sheikh', 'Haji']
    suffixes = ['Jr.', 'Sr.', 'III', 'PhD', 'MD']

    if random.random() < 0.5 :
        return f"{random.choice( prefixes )} {name}"
    else :
        return f"{name} {random.choice( suffixes )}"


def transliteration_variations ( name ) :
    """Create transliteration variations"""
    transliterations = {
        'Mohammed' : ['Mohammad', 'Muhammad', 'Muhammed'],
        'Ahmed' : ['Ahmad', 'Ahmet'],
        'Hassan' : ['Hasan', 'Hassaan'],
        'Hussein' : ['Hussain', 'Husain'],
        'Ali' : ['Aly', 'Aliy'],
        'Omar' : ['Umar', 'Omer'],
        'Khalil' : ['Khaleel', 'Halil'],
        'Rashid' : ['Rasheed', 'Rashed']
    }

    for original, variations in transliterations.items() :
        if original in name :
            return name.replace( original, random.choice( variations ) )

    return name


def generate_sample_transactions_with_watchlist_matches ( watchlist_df, num_transactions=100 ) :
    """
    Generate sample transactions that include some matches with the watchlist
    for testing the name screening functionality.
    """
    transactions = []

    for i in range( num_transactions ) :
        # 15% chance of transaction involving watchlist entity
        if random.random() < 0.15 :
            # Pick a random watchlist entry
            watchlist_entry = watchlist_df.sample( 1 ).iloc[0]
            sender_name = watchlist_entry['name']

            # Sometimes use a slight variation
            if random.random() < 0.3 :
                sender_name = add_typos( sender_name )
        else :
            sender_name = fake.name()

        transaction = {
            'transaction_id' : f"TXN_{i + 1:06d}",
            'sender_name' : sender_name,
            'receiver_name' : fake.name(),
            'amount' : round( random.uniform( 1000, 500000 ), 2 ),
            'timestamp' : fake.date_time_between( start_date='-30d', end_date='now' ),
            'payment_method' : random.choice( ['UPI', 'Card', 'Net Banking', 'Wallet'] ),
            'location' : fake.city()
        }

        transactions.append( transaction )

    transactions_df = pd.DataFrame( transactions )
    transactions_df.to_csv( 'sample_transactions.csv', index=False )
    print( f"Generated {len( transactions_df )} sample transactions with some watchlist matches" )

    return transactions_df


# Main execution
if __name__ == "__main__" :
    # Generate the main watchlist
    watchlist_df = generate_synthetic_aml_watchlist( num_entries=1000 )

    # Display statistics
    print( "\n=== Watchlist Statistics ===" )
    print( f"Total entries: {len( watchlist_df )}" )
    print( f"Entity types: {watchlist_df['entity_type'].value_counts().to_dict()}" )
    print( f"Risk categories: {watchlist_df['risk_category'].value_counts().to_dict()}" )
    print( f"Average risk score: {watchlist_df['risk_score'].mean():.2f}" )

    # Display sample entries
    print( "\n=== Sample Entries ===" )
    print( watchlist_df[['name', 'entity_type', 'risk_category', 'risk_score', 'country']].head( 10 ) )

    # Generate sample transactions for testing
    sample_transactions = generate_sample_transactions_with_watchlist_matches( watchlist_df, 100 )

    print( "\n=== Files Generated ===" )
    print( "1. aml_watchlist.csv - Main AML watchlist" )
    print( "2. sample_transactions.csv - Sample transactions with some watchlist matches" )
    print( "\nFiles are ready for use with the FinGuard Pro name screening engine!" )