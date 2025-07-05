#!/usr/bin/env python3
"""
Simple test script to verify the login system works
"""

import requests
import json

def test_authentication():
    """Test the authentication system"""
    API_URL = "http://localhost:5000/api"
    
    print("🧪 Testing FinGuardPro Authentication System")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {"username": "admin", "password": "admin123", "expected_role": "admin"},
        {"username": "compliance", "password": "admin123", "expected_role": "compliance"},
        {"username": "demo", "password": "user123", "expected_role": "user"},
        {"username": "user1", "password": "user123", "expected_role": "user"},
        {"username": "invalid", "password": "wrong", "expected_role": None},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['username']}")
        
        try:
            response = requests.post(
                f"{API_URL}/auth/validate",
                json={
                    "username": test_case["username"],
                    "password": test_case["password"]
                },
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("valid"):
                    user = data.get("user", {})
                    role = user.get("role")
                    name = user.get("name")
                    
                    if role == test_case["expected_role"]:
                        print(f"   ✅ SUCCESS: {name} ({role})")
                    else:
                        print(f"   ⚠️  ROLE MISMATCH: Expected {test_case['expected_role']}, got {role}")
                else:
                    if test_case["expected_role"] is None:
                        print(f"   ✅ SUCCESS: Invalid credentials correctly rejected")
                    else:
                        print(f"   ❌ FAILED: Valid credentials rejected")
            else:
                if test_case["expected_role"] is None:
                    print(f"   ✅ SUCCESS: Invalid credentials correctly rejected (HTTP {response.status_code})")
                else:
                    print(f"   ❌ FAILED: HTTP {response.status_code}")
                    
        except requests.exceptions.ConnectionError:
            print(f"   🔌 CONNECTION ERROR: API server not running")
            return False
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Authentication testing completed!")
    return True

if __name__ == "__main__":
    test_authentication()