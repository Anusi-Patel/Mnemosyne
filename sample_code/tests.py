
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
