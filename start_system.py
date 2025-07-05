#!/usr/bin/env python3
"""
FinGuardPro System Startup Script
This script helps you start the complete FinGuardPro system
"""

import subprocess
import sys
import os
import time
import signal
import threading
from pathlib import Path

def print_banner():
    """Print the FinGuardPro banner"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                           🛡️  FinGuardPro System                              ║
    ║                      Advanced Financial Fraud Detection                       ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please check your Python environment.")
        return False

def start_backend():
    """Start the Flask backend API"""
    print("🚀 Starting Backend API Server...")
    try:
        backend_process = subprocess.Popen(
            [sys.executable, "api/api_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        time.sleep(3)  # Give the backend time to start
        return backend_process
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_admin_dashboard():
    """Start the admin dashboard"""
    print("🛡️  Starting Admin Dashboard...")
    try:
        admin_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "dasboards/admin_dashboard.py", 
             "--server.port=8501", "--server.address=0.0.0.0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        return admin_process
    except Exception as e:
        print(f"❌ Failed to start admin dashboard: {e}")
        return None

def start_user_dashboard():
    """Start the user dashboard"""
    print("💳 Starting User Dashboard...")
    try:
        user_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "dasboards/user_dashboard.py", 
             "--server.port=8502", "--server.address=0.0.0.0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        return user_process
    except Exception as e:
        print(f"❌ Failed to start user dashboard: {e}")
        return None

def print_urls():
    """Print the URLs where services are available"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                           🌐 Service URLs                                     ║
    ║                                                                               ║
    ║  🔧 Backend API:        http://localhost:5000                                ║
    ║  🛡️  Admin Dashboard:    http://localhost:8501                                ║
    ║  💳 User Dashboard:     http://localhost:8502                                ║
    ║                                                                               ║
    ║  📝 API Documentation:  http://localhost:5000/health                         ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

def print_credentials():
    """Print the demo credentials"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                           🔐 Demo Credentials                                 ║
    ║                                                                               ║
    ║  👑 Admin Access:                                                             ║
    ║     Username: admin      Password: admin123                                  ║
    ║     Username: compliance Password: admin123                                  ║
    ║                                                                               ║
    ║  👤 User Access:                                                              ║
    ║     Username: demo       Password: user123                                   ║
    ║     Username: user1      Password: user123                                   ║
    ║     Username: user2      Password: user123                                   ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

def print_instructions():
    """Print usage instructions"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                           📋 Instructions                                     ║
    ║                                                                               ║
    ║  1. Wait for all services to start (about 10-15 seconds)                     ║
    ║  2. Open your browser and visit the URLs above                               ║
    ║  3. Use the demo credentials to log in                                        ║
    ║  4. Press Ctrl+C to stop all services                                        ║
    ║                                                                               ║
    ║  🔧 Troubleshooting:                                                          ║
    ║     - If login fails, make sure the backend API is running                   ║
    ║     - Check that all required ports are available                            ║
    ║     - Ensure Python packages are installed correctly                         ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

def main():
    """Main function to start the system"""
    print_banner()
    
    # Check if we're in the right directory
    if not Path("api/api_server.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Start services
    processes = []
    
    # Start backend
    backend_proc = start_backend()
    if backend_proc:
        processes.append(("Backend API", backend_proc))
    else:
        print("❌ Cannot start system without backend API")
        sys.exit(1)
    
    # Start dashboards
    admin_proc = start_admin_dashboard()
    if admin_proc:
        processes.append(("Admin Dashboard", admin_proc))
    
    user_proc = start_user_dashboard()
    if user_proc:
        processes.append(("User Dashboard", user_proc))
    
    # Print information
    print("\n" + "="*80)
    print("🎉 FinGuardPro System Started Successfully!")
    print("="*80)
    
    print_urls()
    print_credentials()
    print_instructions()
    
    print("🔄 System is running... Press Ctrl+C to stop all services")
    print("="*80)
    
    # Handle shutdown
    def signal_handler(sig, frame):
        print("\n\n🛑 Shutting down FinGuardPro system...")
        for name, process in processes:
            print(f"   Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print("✅ All services stopped. Goodbye!")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
            # Check if any process died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"⚠️  {name} has stopped unexpectedly")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()