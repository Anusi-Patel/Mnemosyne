"""
MnemSEARCH v0.1 - Weekend Project
A simple code search engine using Week 1 concepts

This is the first prototype of Mnemosyne!
"""

import numpy as np
import os
from pathlib import Path

print("=" * 60)
print("MnemSEARCH v0.1 - Your First Code Search Engine")
print("Building the foundation of Mnemosyne")
print("=" * 60)

# ============================================================
# PART 1: SIMPLE EMBEDDING (Keyword-based)
# ============================================================

def create_simple_embedding(text, keywords):
    """
    Create a simple embedding by counting keywords.
    In real Mnemosyne, we'll use transformer embeddings.
    
    Args:
        text: The code or query text
        keywords: List of keywords to count
    
    Returns:
        numpy array: Simple embedding vector
    """
    text_lower = text.lower()
    embedding = []
    
    for keyword in keywords:
        # Count how many times this keyword appears
        count = text_lower.count(keyword.lower())
        embedding.append(count)
    
    # Normalize to prevent long files from dominating
    embedding = np.array(embedding, dtype=float)
    
    # Add a tiny bit to avoid division by zero
    magnitude = np.linalg.norm(embedding) + 1e-10
    normalized = embedding / magnitude
    
    return normalized

# Define keywords relevant to code
CODE_KEYWORDS = [
    "def", "class", "import", "return", "if", "for", "while",
    "auth", "login", "password", "user", "database", "query",
    "api", "request", "response", "error", "test", "main",
    "function", "method", "variable", "string", "int", "list"
]

print(f"\nUsing {len(CODE_KEYWORDS)} keywords for embeddings")
print(f"Keywords: {', '.join(CODE_KEYWORDS[:10])}...\n")

# ============================================================
# PART 2: CODE CHUNK PROCESSOR
# ============================================================

class CodeChunk:
    """Represents a chunk of code with its embedding"""
    
    def __init__(self, file_path, content, line_start, line_end):
        self.file_path = file_path
        self.content = content
        self.line_start = line_start
        self.line_end = line_end
        self.embedding = None
    
    def create_embedding(self, keywords):
        """Generate embedding for this chunk"""
        self.embedding = create_simple_embedding(self.content, keywords)
    
    def __repr__(self):
        return f"CodeChunk({self.file_path}:{self.line_start}-{self.line_end})"

def chunk_file(file_path, chunk_size=20):
    """
    Split a file into chunks of lines
    
    Args:
        file_path: Path to the file
        chunk_size: Number of lines per chunk
    
    Returns:
        list of CodeChunk objects
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        chunks = []
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i+chunk_size]
            content = ''.join(chunk_lines)
            
            chunk = CodeChunk(
                file_path=file_path,
                content=content,
                line_start=i+1,
                line_end=i+len(chunk_lines)
            )
            chunks.append(chunk)
        
        return chunks
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

# ============================================================
# PART 3: THE SEARCH ENGINE
# ============================================================

class MnemSearch:
    """The core search engine - Week 1 concepts in action!"""
    
    def __init__(self, keywords):
        self.keywords = keywords
        self.code_chunks = []
        self.embeddings_matrix = None
    
    def index_directory(self, directory_path):
        """
        Index all Python files in a directory
        """
        print(f"Indexing directory: {directory_path}")
        
        python_files = list(Path(directory_path).rglob("*.py"))
        print(f"Found {len(python_files)} Python files")
        
        for file_path in python_files:
            chunks = chunk_file(file_path)
            
            for chunk in chunks:
                chunk.create_embedding(self.keywords)
                self.code_chunks.append(chunk)
        
        # Create embeddings matrix (BROADCASTING in action!)
        if self.code_chunks:
            self.embeddings_matrix = np.array([
                chunk.embedding for chunk in self.code_chunks
            ])
            print(f"Created embeddings matrix: {self.embeddings_matrix.shape}")
        else:
            print("No code chunks found!")
    
    def search(self, query, top_k=5):
        """
        Search for code chunks matching the query
        
        Args:
            query: Search query string
            top_k: Number of results to return
        
        Returns:
            list of (CodeChunk, confidence_score) tuples
        """
        if not self.code_chunks:
            print("No indexed code! Run index_directory() first.")
            return []
        
        # Step 1: Create query embedding
        query_embedding = create_simple_embedding(query, self.keywords)
        print(f"\n🔍 Query: '{query}'")
        print(f"Query embedding shape: {query_embedding.shape}")
        
        # Step 2: Compute similarities using DOT PRODUCT
        # This is BROADCASTING: (embedding_dim,) with (num_chunks, embedding_dim)
        similarities = np.dot(self.embeddings_matrix, query_embedding)
        print(f"Computed {len(similarities)} similarities")
        
        # Step 3: Apply SIGMOID to get confidence scores
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        confidences = sigmoid(similarities * 10)  # Scale up for better separation
        
        # Step 4: Rank by confidence (argsort in descending order)
        top_indices = np.argsort(confidences)[::-1][:top_k]
        
        # Step 5: Return results
        results = []
        for idx in top_indices:
            results.append((self.code_chunks[idx], confidences[idx]))
        
        return results
    
    def display_with_context(self, results):
        """Show more context around matches"""
        if not results:
            print("No results found!")
            return

        for i, (chunk, confidence) in enumerate(results, 1):
            print(f"\n{'='*60}")
            print(f"Result #{i} - Confidence: {confidence:.1%}")
            print(f"File: {chunk.file_path}")
            print(f"Lines: {chunk.line_start}-{chunk.line_end}")
            print(f"{'='*60}")

            # Show actual code with line numbers
            lines = chunk.content.split('\n')
            for j, line in enumerate(lines, chunk.line_start):
                if line.strip():
                    print(f"{j:4d} | {line}")

# ============================================================
# PART 4: NEW FEATURES
# ============================================================

def interactive_search():
    """Let user type queries interactively"""
    search_engine = MnemSearch(CODE_KEYWORDS)
    search_engine.index_directory("sample_code")

    print("MnemSEARCH Interactive Mode")
    print("Type 'quit' to exit\n")

    while True:
        query = input("🔍 Search: ")
        if query.lower() == 'quit':
            break

        results = search_engine.search(query, top_k=3)
        search_engine.display_with_context(results)

def evaluate_search_quality(search_engine, test_cases):
    """
    Test cases: [(query, expected_file_name), ...]
    Returns: accuracy score
    """
    correct = 0

    for query, expected_file in test_cases:
        results = search_engine.search(query, top_k=1)
        if results:
            top_result_file = Path(results[0][0].file_path).name
            if expected_file in top_result_file:
                correct += 1
                print(f"✓ '{query}' → {top_result_file}")
            else:
                print(f"✗ '{query}' → {top_result_file} (expected {expected_file})")

    accuracy = correct / len(test_cases)
    print(f"\nAccuracy: {accuracy:.1%}")
    return accuracy

# Test it
test_cases = [
    ("login authentication", "auth.py"),
    ("database query", "database.py"),
    ("api endpoint", "api.py"),
    ("unit test", "tests.py")
]

# ============================================================
# PART 5: DEMO AND TESTING
# ============================================================

def create_sample_files():
    """Create sample Python files for testing"""
    
    sample_dir = Path("sample_code")
    sample_dir.mkdir(exist_ok=True)
    
    # Sample file 1: Authentication
    auth_code = '''
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
'''
    
    # Sample file 2: Database operations
    db_code = '''
class Database:
    def __init__(self, connection_string):
        self.connection = connect(connection_string)
    
    def query(self, sql, *params):
        """Execute SQL query with parameters"""
        cursor = self.connection.cursor()
        cursor.execute(sql, params)
        return cursor.fetchall()
    
    def insert(self, table, data):
        """Insert data into table"""
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        return self.query(sql, *data.values())
    
    def update(self, table, data, condition):
        """Update records in table"""
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {condition}"
        return self.query(sql, *data.values())
'''
    
    # Sample file 3: API endpoints
    api_code = '''
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
'''
    
    # Sample file 4: Tests
    test_code = '''
import unittest

class TestAuth(unittest.TestCase):
    def test_login_success(self):
        """Test successful login"""
        result = login("testuser", "testpass")
        self.assertIsNotNone(result)
    
    def test_login_failure(self):
        """Test failed login with wrong password"""
        result = login("testuser", "wrongpass")
        self.assertIsNone(result)
    
    def test_password_verification(self):
        """Test password verification function"""
        hash_val = "abc123"
        self.assertTrue(verify_password("test", hash_val))

class TestDatabase(unittest.TestCase):
    def test_query(self):
        """Test database query method"""
        db = Database("test.db")
        results = db.query("SELECT * FROM users")
        self.assertIsInstance(results, list)
'''
    
    # Write files
    (sample_dir / "auth.py").write_text(auth_code)
    (sample_dir / "database.py").write_text(db_code)
    (sample_dir / "api.py").write_text(api_code)
    (sample_dir / "tests.py").write_text(test_code)
    
    print(f"✓ Created sample files in {sample_dir}/")
    return sample_dir

# ============================================================
# MAIN DEMO
# ============================================================

if __name__ == "__main__":
    print("\n🎯 DEMO: Building and Testing MnemSEARCH")
    print("=" * 60)
    
    # Create sample files
    sample_dir = create_sample_files()
    
    # Initialize search engine
    print("\n📚 Initializing MnemSEARCH...")
    search_engine = MnemSearch(CODE_KEYWORDS)
    
    # Index the sample directory
    print("\n🔍 Indexing code...")
    search_engine.index_directory(sample_dir)
    
    # Run some searches
    test_queries = [
        "authentication and login",
        "database query operations",
        "API endpoint for users",
        "password verification",
        "unit tests"
    ]
    
    print("\n" + "=" * 60)
    print("RUNNING TEST SEARCHES")
    print("=" * 60)
    
    for query in test_queries:
        results = search_engine.search(query, top_k=3)
        search_engine.display_with_context(results)
        print("\n" + "-" * 60 + "\n")

    # Evaluate search quality
    print("\n" + "=" * 60)
    print("EVALUATING SEARCH QUALITY")
    print("=" * 60)
    accuracy = evaluate_search_quality(search_engine, test_cases)
    print(f"Search accuracy: {accuracy:.1%}")

    print("=" * 60)
    print("🎉 MnemSEARCH v0.1 Demo Complete!")
    print("=" * 60)
    print("\n💡 What you just built:")
    print("✓ Vector embeddings (simple keyword-based)")
    print("✓ Dot product similarity calculation")
    print("✓ Sigmoid confidence scoring")
    print("✓ Broadcasting for batch operations")
    print("✓ Ranking algorithm")
    print("✓ Interactive search mode")
    print("✓ Search quality evaluation")
    print("✓ Enhanced result display with context")
    print("\n🔮 This IS Mnemosyne's architecture!")
    print("Next: Replace simple embeddings with transformers")
    print("\n💻 To try interactive search, call: interactive_search()")
    print("=" * 60)