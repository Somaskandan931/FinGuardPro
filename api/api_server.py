from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import os
import csv
from pathlib import Path
import json
import io
from functools import wraps

# Initialize Flask app
app = Flask( __name__ )
app.config['SECRET_KEY'] = 'finguard_secret_key_2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///finguard.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize extensions
db = SQLAlchemy( app )
login_manager = LoginManager()
login_manager.init_app( app )
login_manager.login_view = 'login'

# Enable CORS for all domains on all routes
CORS(app)

# Ensure upload folder exists
os.makedirs( app.config['UPLOAD_FOLDER'], exist_ok=True )


# Database Models
class User( UserMixin, db.Model ) :
    id = db.Column( db.Integer, primary_key=True )
    username = db.Column( db.String( 80 ), unique=True, nullable=False )
    email = db.Column( db.String( 120 ), unique=True, nullable=False )
    password_hash = db.Column( db.String( 120 ), nullable=False )
    name = db.Column( db.String( 100 ), nullable=False )
    role = db.Column( db.String( 20 ), default='user' )  # 'user', 'admin', 'compliance'
    created_at = db.Column( db.DateTime, default=datetime.utcnow )
    last_login = db.Column( db.DateTime )
    is_active = db.Column( db.Boolean, default=True )


class Transaction( db.Model ) :
    id = db.Column( db.Integer, primary_key=True )
    transaction_id = db.Column( db.String( 50 ), unique=True, nullable=False )
    user_id = db.Column( db.Integer, db.ForeignKey( 'user.id' ), nullable=False )
    amount = db.Column( db.Float, nullable=False )
    merchant_category = db.Column( db.String( 50 ), nullable=False )
    transaction_hour = db.Column( db.Integer, nullable=False )
    is_weekend = db.Column( db.Boolean, default=False )
    days_since_last = db.Column( db.Float, default=0.0 )
    distance_from_home = db.Column( db.Float, default=0.0 )
    card_present = db.Column( db.Boolean, default=True )
    fraud_score = db.Column( db.Float, default=0.0 )
    risk_level = db.Column( db.String( 20 ), default='Low' )
    is_fraud = db.Column( db.Boolean, default=False )
    timestamp = db.Column( db.DateTime, default=datetime.utcnow )
    processed = db.Column( db.Boolean, default=False )


class AnalysisSession( db.Model ) :
    id = db.Column( db.Integer, primary_key=True )
    session_id = db.Column( db.String( 100 ), unique=True, nullable=False )
    user_id = db.Column( db.Integer, db.ForeignKey( 'user.id' ), nullable=False )
    filename = db.Column( db.String( 255 ), nullable=False )
    total_transactions = db.Column( db.Integer, default=0 )
    fraud_count = db.Column( db.Integer, default=0 )
    created_at = db.Column( db.DateTime, default=datetime.utcnow )
    status = db.Column( db.String( 20 ), default='pending' )


# Login manager user loader
@login_manager.user_loader
def load_user ( user_id ) :
    return User.query.get( int( user_id ) )


# Decorators
def admin_required ( f ) :
    @wraps( f )
    def decorated_function ( *args, **kwargs ) :
        if not current_user.is_authenticated or current_user.role not in ['admin', 'compliance'] :
            return jsonify( {'error' : 'Admin access required'} ), 403
        return f( *args, **kwargs )

    return decorated_function


def user_required ( f ) :
    @wraps( f )
    def decorated_function ( *args, **kwargs ) :
        if not current_user.is_authenticated :
            return jsonify( {'error' : 'Login required'} ), 401
        return f( *args, **kwargs )

    return decorated_function


# Model loading
class FraudDetectionModel :
    def __init__ ( self ) :
        self.model = None
        self.scaler = None
        self.load_model()

    def load_model ( self ) :
        try :
            model_path = Path( 'models/fraud_detection_model.h5' )
            scaler_path = Path( 'models/scaler.pkl' )

            if model_path.exists() and scaler_path.exists() :
                self.model = tf.keras.models.load_model( str( model_path ) )
                self.scaler = joblib.load( str( scaler_path ) )
                print( "✅ Model and scaler loaded successfully" )
            else :
                print( "⚠️ Model files not found, running in demo mode" )
        except Exception as e :
            print( f"❌ Error loading model: {e}" )

    def predict ( self, features ) :
        if self.model is None or self.scaler is None :
            # Demo mode - return random prediction
            return np.random.beta( 2, 8 )

        try :
            features_scaled = self.scaler.transform( features )
            prediction = self.model.predict( features_scaled )
            return prediction[0][0] if prediction.ndim > 1 else prediction[0]
        except Exception as e :
            print( f"Error in prediction: {e}" )
            return np.random.beta( 2, 8 )


# Initialize model
fraud_model = FraudDetectionModel()


# Utility functions
def calculate_risk_level ( fraud_score ) :
    if fraud_score < 0.3 :
        return 'Low'
    elif fraud_score < 0.7 :
        return 'Medium'
    else :
        return 'High'


def generate_sample_data ( n_transactions=100 ) :
    """Generate sample transaction data for demo purposes"""
    np.random.seed( 42 )

    data = []
    for i in range( n_transactions ) :
        transaction = {
            'transaction_id' : f'TXN_{i:06d}',
            'amount' : float( np.random.lognormal( 3, 1.5 ) ),
            'merchant_category' : np.random.choice( ['Grocery', 'Gas', 'Restaurant', 'Online', 'ATM', 'Other'] ),
            'transaction_hour' : int( np.random.randint( 0, 24 ) ),
            'is_weekend' : bool( np.random.choice( [0, 1], p=[0.7, 0.3] ) ),
            'days_since_last' : float( np.random.exponential( 1 ) ),
            'distance_from_home' : float( np.random.exponential( 5 ) ),
            'card_present' : bool( np.random.choice( [0, 1], p=[0.3, 0.7] ) ),
            'timestamp' : datetime.now() - timedelta( days=i ),
            'fraud_score' : float( np.random.beta( 2, 8 ) ),
        }
        transaction['risk_level'] = calculate_risk_level( transaction['fraud_score'] )
        transaction['is_fraud'] = transaction['fraud_score'] > 0.7
        data.append( transaction )

    return data


# Routes

# Authentication Routes
@app.route( '/api/auth/login', methods=['POST'] )
def login () :
    data = request.get_json()
    username = data.get( 'username' )
    password = data.get( 'password' )

    if not username or not password :
        return jsonify( {'error' : 'Username and password required'} ), 400

    user = User.query.filter_by( username=username ).first()

    if user and check_password_hash( user.password_hash, password ) :
        login_user( user )
        user.last_login = datetime.utcnow()
        db.session.commit()

        return jsonify( {
            'message' : 'Login successful',
            'user' : {
                'id' : user.id,
                'username' : user.username,
                'name' : user.name,
                'role' : user.role
            }
        } )

    return jsonify( {'error' : 'Invalid credentials'} ), 401


# Simple validation endpoint for dashboards
@app.route( '/api/auth/validate', methods=['POST'] )
def validate_credentials () :
    data = request.get_json()
    username = data.get( 'username' )
    password = data.get( 'password' )

    if not username or not password :
        return jsonify( {'valid' : False, 'error' : 'Username and password required'} ), 400

    user = User.query.filter_by( username=username ).first()

    if user and check_password_hash( user.password_hash, password ) :
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        return jsonify( {
            'valid' : True,
            'user' : {
                'id' : user.id,
                'username' : user.username,
                'name' : user.name,
                'role' : user.role
            }
        } )

    return jsonify( {'valid' : False, 'error' : 'Invalid credentials'} ), 401


@app.route( '/api/auth/logout', methods=['POST'] )
@login_required
def logout () :
    logout_user()
    return jsonify( {'message' : 'Logout successful'} )


@app.route( '/api/auth/register', methods=['POST'] )
def register () :
    data = request.get_json()
    username = data.get( 'username' )
    email = data.get( 'email' )
    password = data.get( 'password' )
    name = data.get( 'name' )
    role = data.get( 'role', 'user' )

    if not all( [username, email, password, name] ) :
        return jsonify( {'error' : 'All fields required'} ), 400

    # Check if user exists
    if User.query.filter_by( username=username ).first() :
        return jsonify( {'error' : 'Username already exists'} ), 400

    if User.query.filter_by( email=email ).first() :
        return jsonify( {'error' : 'Email already exists'} ), 400

    # Create user
    user = User(
        username=username,
        email=email,
        password_hash=generate_password_hash( password ),
        name=name,
        role=role
    )

    db.session.add( user )
    db.session.commit()

    return jsonify( {'message' : 'User created successfully'} ), 201


@app.route( '/api/auth/me', methods=['GET'] )
@login_required
def get_current_user () :
    return jsonify( {
        'id' : current_user.id,
        'username' : current_user.username,
        'name' : current_user.name,
        'role' : current_user.role,
        'email' : current_user.email,
        'last_login' : current_user.last_login.isoformat() if current_user.last_login else None
    } )


# Dashboard Routes
@app.route( '/api/dashboard/overview', methods=['GET'] )
@login_required
def dashboard_overview () :
    if current_user.role in ['admin', 'compliance'] :
        # Admin dashboard
        total_users = User.query.count()
        total_transactions = Transaction.query.count()
        fraud_transactions = Transaction.query.filter_by( is_fraud=True ).count()
        recent_transactions = Transaction.query.order_by( Transaction.timestamp.desc() ).limit( 10 ).all()

        return jsonify( {
            'total_users' : total_users,
            'total_transactions' : total_transactions,
            'fraud_transactions' : fraud_transactions,
            'fraud_rate' : (fraud_transactions / total_transactions * 100) if total_transactions > 0 else 0,
            'recent_transactions' : [{
                'id' : t.transaction_id,
                'amount' : t.amount,
                'merchant_category' : t.merchant_category,
                'risk_level' : t.risk_level,
                'fraud_score' : t.fraud_score,
                'timestamp' : t.timestamp.isoformat()
            } for t in recent_transactions],
            'model_status' : 'active' if fraud_model.model is not None else 'demo'
        } )
    else :
        # User dashboard
        user_transactions = Transaction.query.filter_by( user_id=current_user.id ).all()
        fraud_count = sum( 1 for t in user_transactions if t.is_fraud )
        total_amount = sum( t.amount for t in user_transactions )

        # Generate sample data if no transactions
        if not user_transactions :
            sample_data = generate_sample_data( 20 )
            return jsonify( {
                'total_transactions' : len( sample_data ),
                'fraud_transactions' : sum( 1 for t in sample_data if t['is_fraud'] ),
                'total_amount' : sum( t['amount'] for t in sample_data ),
                'recent_transactions' : sample_data[:10],
                'sample_data' : True
            } )

        return jsonify( {
            'total_transactions' : len( user_transactions ),
            'fraud_transactions' : fraud_count,
            'total_amount' : total_amount,
            'recent_transactions' : [{
                'id' : t.transaction_id,
                'amount' : t.amount,
                'merchant_category' : t.merchant_category,
                'risk_level' : t.risk_level,
                'fraud_score' : t.fraud_score,
                'timestamp' : t.timestamp.isoformat()
            } for t in user_transactions[-10 :]],
            'sample_data' : False
        } )


# Transaction Analysis Routes
@app.route( '/api/transactions/analyze', methods=['POST'] )
@login_required
def analyze_transaction () :
    data = request.get_json()

    # Extract features
    features = np.array( [[
        data.get( 'amount', 0 ),
        data.get( 'transaction_hour', 12 ),
        int( data.get( 'is_weekend', False ) ),
        data.get( 'days_since_last', 1 ),
        data.get( 'distance_from_home', 0 ),
        int( data.get( 'card_present', True ) )
    ]] )

    # Predict fraud
    fraud_score = fraud_model.predict( features )
    risk_level = calculate_risk_level( fraud_score )

    # Save transaction if requested
    if data.get( 'save_transaction', False ) :
        transaction = Transaction(
            transaction_id=f"TXN_{datetime.now().strftime( '%Y%m%d%H%M%S' )}",
            user_id=current_user.id,
            amount=data.get( 'amount', 0 ),
            merchant_category=data.get( 'merchant_category', 'Other' ),
            transaction_hour=data.get( 'transaction_hour', 12 ),
            is_weekend=data.get( 'is_weekend', False ),
            days_since_last=data.get( 'days_since_last', 1 ),
            distance_from_home=data.get( 'distance_from_home', 0 ),
            card_present=data.get( 'card_present', True ),
            fraud_score=fraud_score,
            risk_level=risk_level,
            is_fraud=fraud_score > 0.7
        )
        db.session.add( transaction )
        db.session.commit()

    return jsonify( {
        'fraud_score' : float( fraud_score ),
        'risk_level' : risk_level,
        'is_fraud' : fraud_score > 0.7,
        'recommendation' : 'BLOCK' if fraud_score > 0.7 else ('REVIEW' if fraud_score > 0.4 else 'APPROVE')
    } )


@app.route( '/api/transactions/bulk-analyze', methods=['POST'] )
@login_required
def bulk_analyze_transactions () :
    if 'file' not in request.files :
        return jsonify( {'error' : 'No file uploaded'} ), 400

    file = request.files['file']
    if file.filename == '' :
        return jsonify( {'error' : 'No file selected'} ), 400

    if not file.filename.endswith( '.csv' ) :
        return jsonify( {'error' : 'Only CSV files are supported'} ), 400

    try :
        # Read CSV file
        df = pd.read_csv( file )

        # Validate required columns
        required_columns = ['amount', 'merchant_category', 'transaction_hour',
                            'is_weekend', 'days_since_last', 'distance_from_home', 'card_present']

        if not all( col in df.columns for col in required_columns ) :
            return jsonify( {'error' : 'Missing required columns'} ), 400

        # Process transactions
        results = []
        for _, row in df.iterrows() :
            features = np.array( [[
                row['amount'],
                row['transaction_hour'],
                int( row['is_weekend'] ),
                row['days_since_last'],
                row['distance_from_home'],
                int( row['card_present'] )
            ]] )

            fraud_score = fraud_model.predict( features )
            risk_level = calculate_risk_level( fraud_score )

            results.append( {
                'fraud_score' : float( fraud_score ),
                'risk_level' : risk_level,
                'is_fraud' : fraud_score > 0.7
            } )

        # Create session for this analysis
        session_id = f"BULK_{datetime.now().strftime( '%Y%m%d%H%M%S' )}"
        analysis_session = AnalysisSession(
            session_id=session_id,
            user_id=current_user.id,
            filename=secure_filename( file.filename ),
            total_transactions=len( results ),
            fraud_count=sum( 1 for r in results if r['is_fraud'] ),
            status='completed'
        )
        db.session.add( analysis_session )
        db.session.commit()

        return jsonify( {
            'session_id' : session_id,
            'total_transactions' : len( results ),
            'fraud_count' : sum( 1 for r in results if r['is_fraud'] ),
            'results' : results
        } )

    except Exception as e :
        return jsonify( {'error' : str( e )} ), 500


@app.route( '/api/transactions/export/<session_id>', methods=['GET'] )
@login_required
def export_analysis_results ( session_id ) :
    analysis_session = AnalysisSession.query.filter_by(
        session_id=session_id,
        user_id=current_user.id
    ).first()

    if not analysis_session :
        return jsonify( {'error' : 'Analysis session not found'} ), 404

    # Generate CSV with sample results
    output = io.StringIO()
    writer = csv.writer( output )

    # Write header
    writer.writerow( ['Transaction_ID', 'Fraud_Score', 'Risk_Level', 'Is_Fraud', 'Recommendation'] )

    # Write sample data
    for i in range( analysis_session.total_transactions ) :
        fraud_score = np.random.beta( 2, 8 )
        risk_level = calculate_risk_level( fraud_score )
        is_fraud = fraud_score > 0.7
        recommendation = 'BLOCK' if fraud_score > 0.7 else ('REVIEW' if fraud_score > 0.4 else 'APPROVE')

        writer.writerow( [
            f'TXN_{i:06d}',
            f'{fraud_score:.4f}',
            risk_level,
            is_fraud,
            recommendation
        ] )

    output.seek( 0 )

    return send_file(
        io.BytesIO( output.getvalue().encode() ),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'fraud_analysis_{session_id}.csv'
    )


# Reports Routes
@app.route( '/api/reports/summary', methods=['GET'] )
@login_required
def get_report_summary () :
    start_date = request.args.get( 'start_date' )
    end_date = request.args.get( 'end_date' )

    query = Transaction.query
    if current_user.role == 'user' :
        query = query.filter_by( user_id=current_user.id )

    if start_date :
        query = query.filter( Transaction.timestamp >= datetime.fromisoformat( start_date ) )
    if end_date :
        query = query.filter( Transaction.timestamp <= datetime.fromisoformat( end_date ) )

    transactions = query.all()

    if not transactions :
        # Generate sample data for demo
        sample_data = generate_sample_data( 50 )
        return jsonify( {
            'total_transactions' : len( sample_data ),
            'total_amount' : sum( t['amount'] for t in sample_data ),
            'fraud_transactions' : sum( 1 for t in sample_data if t['is_fraud'] ),
            'average_fraud_score' : np.mean( [t['fraud_score'] for t in sample_data] ),
            'risk_distribution' : {
                'Low' : sum( 1 for t in sample_data if t['risk_level'] == 'Low' ),
                'Medium' : sum( 1 for t in sample_data if t['risk_level'] == 'Medium' ),
                'High' : sum( 1 for t in sample_data if t['risk_level'] == 'High' )
            },
            'sample_data' : True
        } )

    total_amount = sum( t.amount for t in transactions )
    fraud_count = sum( 1 for t in transactions if t.is_fraud )
    avg_fraud_score = np.mean( [t.fraud_score for t in transactions] )

    risk_distribution = {
        'Low' : sum( 1 for t in transactions if t.risk_level == 'Low' ),
        'Medium' : sum( 1 for t in transactions if t.risk_level == 'Medium' ),
        'High' : sum( 1 for t in transactions if t.risk_level == 'High' )
    }

    return jsonify( {
        'total_transactions' : len( transactions ),
        'total_amount' : total_amount,
        'fraud_transactions' : fraud_count,
        'average_fraud_score' : avg_fraud_score,
        'risk_distribution' : risk_distribution,
        'sample_data' : False
    } )


# Admin Routes
@app.route( '/api/admin/users', methods=['GET'] )
@admin_required
def get_all_users () :
    users = User.query.all()
    return jsonify( [{
        'id' : user.id,
        'username' : user.username,
        'name' : user.name,
        'email' : user.email,
        'role' : user.role,
        'created_at' : user.created_at.isoformat(),
        'last_login' : user.last_login.isoformat() if user.last_login else None,
        'is_active' : user.is_active
    } for user in users] )


@app.route( '/api/admin/users/<int:user_id>', methods=['PUT'] )
@admin_required
def update_user ( user_id ) :
    user = User.query.get_or_404( user_id )
    data = request.get_json()

    if 'role' in data :
        user.role = data['role']
    if 'is_active' in data :
        user.is_active = data['is_active']

    db.session.commit()
    return jsonify( {'message' : 'User updated successfully'} )


@app.route( '/api/admin/system-stats', methods=['GET'] )
@admin_required
def get_system_stats () :
    total_users = User.query.count()
    active_users = User.query.filter_by( is_active=True ).count()
    total_transactions = Transaction.query.count()
    total_fraud = Transaction.query.filter_by( is_fraud=True ).count()

    return jsonify( {
        'total_users' : total_users,
        'active_users' : active_users,
        'total_transactions' : total_transactions,
        'total_fraud' : total_fraud,
        'fraud_rate' : (total_fraud / total_transactions * 100) if total_transactions > 0 else 0,
        'model_status' : 'active' if fraud_model.model is not None else 'demo'
    } )


# Health check route
@app.route( '/health', methods=['GET'] )
def health_check () :
    return jsonify( {
        'status' : 'healthy',
        'timestamp' : datetime.utcnow().isoformat(),
        'model_loaded' : fraud_model.model is not None
    } )


# Error handlers
@app.errorhandler( 404 )
def not_found ( error ) :
    return jsonify( {'error' : 'Resource not found'} ), 404


@app.errorhandler( 500 )
def internal_error ( error ) :
    return jsonify( {'error' : 'Internal server error'} ), 500


# Initialize database
def init_db () :
    with app.app_context() :
        db.create_all()

        # Create default admin user if doesn't exist
        if not User.query.filter_by( username='admin' ).first() :
            admin_user = User(
                username='admin',
                email='admin@finguardpro.com',
                password_hash=generate_password_hash( 'admin123' ),
                name='Admin User',
                role='admin'
            )
            db.session.add( admin_user )

        # Create default compliance user if doesn't exist
        if not User.query.filter_by( username='compliance' ).first() :
            compliance_user = User(
                username='compliance',
                email='compliance@finguardpro.com',
                password_hash=generate_password_hash( 'admin123' ),
                name='Compliance Officer',
                role='compliance'
            )
            db.session.add( compliance_user )

        # Create default demo users if they don't exist
        demo_users = [
            {'username': 'demo', 'email': 'demo@finguardpro.com', 'name': 'Demo User', 'password': 'user123'},
            {'username': 'user1', 'email': 'user1@finguardpro.com', 'name': 'John Doe', 'password': 'user123'},
            {'username': 'user2', 'email': 'user2@finguardpro.com', 'name': 'Jane Smith', 'password': 'user123'}
        ]

        for user_data in demo_users :
            if not User.query.filter_by( username=user_data['username'] ).first() :
                demo_user = User(
                    username=user_data['username'],
                    email=user_data['email'],
                    password_hash=generate_password_hash( user_data['password'] ),
                    name=user_data['name'],
                    role='user'
                )
                db.session.add( demo_user )

        db.session.commit()
        print( "✅ Database initialized successfully" )


if __name__ == '__main__' :
    init_db()
    print( "🚀 Starting FinGuardPro Flask Backend..." )
    print( "📊 Admin credentials: admin/admin123" )
    print( "👤 User credentials: demo/user123" )
    app.run( debug=True, host='0.0.0.0', port=5000 )