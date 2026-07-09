
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users from database"""
    users = database.query("SELECT * FROM users")
    return jsonify(users)

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get specific user by ID"""
    user = database.query("SELECT * FROM users WHERE id = ?", user_id)
    if user:
        return jsonify(user[0])
    return jsonify({"error": "User not found"}), 404

@app.route('/api/login', methods=['POST'])
def api_login():
    """Login endpoint"""
    data = request.get_json()
    session = login(data['username'], data['password'])
    if session:
        return jsonify({"session_id": session.id})
    return jsonify({"error": "Invalid credentials"}), 401
