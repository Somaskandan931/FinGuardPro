#!/usr/bin/env python3
"""
FinGuard Pro - Dashboard Launcher

This script helps you easily run the FinGuard Pro dashboards.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import streamlit_authenticator
        import pandas
        import numpy
        import matplotlib
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        return False

def run_dashboard(dashboard_type):
    """Run the specified dashboard"""
    if not check_dependencies():
        return
    
    dashboard_files = {
        "admin": "dasboards/admin_dashboard.py",
        "user": "dasboards/user_dashboard.py"
    }
    
    if dashboard_type not in dashboard_files:
        print(f"❌ Invalid dashboard type: {dashboard_type}")
        return
    
    dashboard_file = dashboard_files[dashboard_type]
    
    if not os.path.exists(dashboard_file):
        print(f"❌ Dashboard file not found: {dashboard_file}")
        return
    
    print(f"🚀 Starting {dashboard_type} dashboard...")
    print(f"📂 File: {dashboard_file}")
    print("\n" + "="*60)
    
    if dashboard_type == "admin":
        print("🔐 Admin Dashboard Credentials:")
        print("Username: admin")
        print("Password: admin123")
        print("\nOR")
        print("Username: compliance")
        print("Password: admin123")
    else:
        print("🔐 User Dashboard Credentials:")
        print("Username: user1")
        print("Password: user123")
        print("\nOR")
        print("Username: user2")
        print("Password: user123")
        print("\nOR")
        print("Username: demo")
        print("Password: user123")
    
    print("="*60)
    print("🌐 Dashboard will open in your default web browser")
    print("Press Ctrl+C to stop the dashboard")
    print("="*60 + "\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_file])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped!")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def main():
    """Main function"""
    print("🛡️  FinGuard Pro - Dashboard Launcher")
    print("=" * 50)
    
    if len(sys.argv) != 2:
        print("Usage: python run_dashboards.py [admin|user]")
        print("\nOptions:")
        print("  admin  - Run the admin dashboard")
        print("  user   - Run the user dashboard")
        print("\nExamples:")
        print("  python run_dashboards.py admin")
        print("  python run_dashboards.py user")
        return
    
    dashboard_type = sys.argv[1].lower()
    run_dashboard(dashboard_type)

if __name__ == "__main__":
    main()