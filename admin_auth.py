"""
Admin Authentication System
Simple session-based auth for admin features
"""

import os
import secrets
from functools import wraps
from flask import session, jsonify
import hashlib

# Admin credentials (in production, use environment variables or database)
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD_HASH = os.getenv('ADMIN_PASSWORD_HASH', None)

# Default password: "admin123" (CHANGE THIS!)
if ADMIN_PASSWORD_HASH is None:
    ADMIN_PASSWORD_HASH = hashlib.sha256('admin123'.encode()).hexdigest()

def hash_password(password):
    """Hash a password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    """Verify a password against its hash"""
    return hash_password(password) == password_hash

def login_admin(username, password):
    """
    Authenticate admin user
    Returns: (success, message)
    """
    if username == ADMIN_USERNAME and verify_password(password, ADMIN_PASSWORD_HASH):
        # Generate session token
        session['admin_logged_in'] = True
        session['admin_token'] = secrets.token_hex(16)
        return True, "Login successful"
    
    return False, "Invalid credentials"

def logout_admin():
    """Logout admin user"""
    session.pop('admin_logged_in', None)
    session.pop('admin_token', None)

def is_admin_logged_in():
    """Check if admin is logged in"""
    return session.get('admin_logged_in', False)

def admin_required(f):
    """Decorator to protect admin-only routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_admin_logged_in():
            return jsonify({'error': 'Admin authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def change_admin_password(old_password, new_password):
    """
    Change admin password
    Returns: (success, message)
    """
    global ADMIN_PASSWORD_HASH
    
    if not verify_password(old_password, ADMIN_PASSWORD_HASH):
        return False, "Current password is incorrect"
    
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters"
    
    ADMIN_PASSWORD_HASH = hash_password(new_password)
    
    # Save to environment file (optional)
    try:
        with open('.admin_password', 'w') as f:
            f.write(ADMIN_PASSWORD_HASH)
    except:
        pass
    
    return True, "Password changed successfully"

# Load admin password from file if exists
if os.path.exists('.admin_password'):
    try:
        with open('.admin_password', 'r') as f:
            ADMIN_PASSWORD_HASH = f.read().strip()
    except:
        pass


