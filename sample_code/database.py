
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
