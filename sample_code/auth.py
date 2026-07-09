
def login(username, password):
    """Authenticate user with username and password"""
    user = database.query("SELECT * FROM users WHERE username = ?", username)
    
    if user and verify_password(password, user.password_hash):
        return create_session(user)
    
    return None

def logout(session_id):
    """End user session"""
    database.query("DELETE FROM sessions WHERE id = ?", session_id)
    return True

def verify_password(plain_password, password_hash):
    """Check if password matches hash"""
    import hashlib
    return hashlib.sha256(plain_password.encode()).hexdigest() == password_hash
