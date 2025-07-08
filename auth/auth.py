from flask import request, jsonify
import jwt
import datetime

# Secret key for JWT (keep this secure in production!)
SECRET_KEY = "78863298c43badbd900fd376fbcc2e1020295f1787a3527bd3b840c5090bc450"

# Updated user store with password + role
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"},
    # You can add more users like:
    # "alice": {"password": "alicepass", "role": "user"},
    # "bob": {"password": "bobpass", "role": "admin"},
}

def generate_token(username, role):
    payload = {
        "user": username,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["user"]
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    user_info = USERS.get(username)
    if user_info and user_info["password"] == password:
        role = user_info["role"]
        token = generate_token(username, role)
        return jsonify({"token": token, "role": role})  # ✅ Return both token and role
    else:
        return jsonify({"error": "Invalid username or password"}), 401

def token_required(f):
    from functools import wraps
    from flask import request, jsonify

    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

        if not token:
            return jsonify({"error": "Token is missing"}), 401

        user = verify_token(token)
        if not user:
            return jsonify({"error": "Invalid or expired token"}), 401

        return f(*args, **kwargs)

    return decorated
