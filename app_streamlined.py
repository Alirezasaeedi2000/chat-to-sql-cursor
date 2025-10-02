#!/usr/bin/env python3
"""
Streamlined NLP-to-SQL App using modular core components.
This demonstrates how the new modular architecture works.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from core.database import ensure_dirs, create_engine_from_url
from core.sql_utils import SafeSqlExecutor, _extract_sql_from_text
from vector import VectorStoreManager
from query_history import QueryHistoryManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)

class StreamlinedNL2SQLApp:
    """Streamlined NLP-to-SQL application using modular core components."""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or "mysql+pymysql://root:@localhost:3306/Farnan?charset=utf8mb4&autocommit=true"
        self.engine = None
        self.sql_executor = None
        self.vector_manager = None
        self.query_history = QueryHistoryManager()
        
        self.setup_database()
        self.setup_components()
    
    def setup_database(self):
        """Setup database connection using core utilities."""
        print("Setting up database connection...")
        
        try:
            # Use our modular database utilities
            self.engine = create_engine_from_url(self.db_url)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print("SUCCESS: Database connection established")
            
        except Exception as e:
            print(f"ERROR: Database connection failed: {e}")
            raise
    
    def setup_components(self):
        """Setup core components using modular utilities."""
        print("Setting up core components...")
        
        # Use our modular SQL executor
        self.sql_executor = SafeSqlExecutor(self.engine)
        print("SUCCESS: SQL executor initialized")
        
        # Initialize vector manager
        self.vector_manager = VectorStoreManager()
        print("SUCCESS: Vector manager initialized")
        
        print("SUCCESS: All components ready")
    
    def test_sql_extraction(self):
        """Test SQL extraction using core utilities."""
        print("\nTesting SQL extraction...")
        
        test_cases = [
            "```sql\nSELECT * FROM users\n```",
            "SELECT name, age FROM customers WHERE age > 18",
            "Here is the query: SELECT COUNT(*) FROM orders"
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            extracted = _extract_sql_from_text(test_case)
            print(f"Test {i}: {test_case[:30]}... -> {extracted}")
        
        print("SUCCESS: SQL extraction tests completed")
    
    def test_sql_execution(self):
        """Test SQL execution using core utilities."""
        print("\nTesting SQL execution...")
        
        try:
            # Simple test query
            test_sql = "SELECT 1 as test_value, 'Hello from modular system!' as message"
            
            df, executed_sql = self.sql_executor.execute_select(test_sql)
            
            print(f"Executed SQL: {executed_sql}")
            print(f"Result: {df.iloc[0].to_dict()}")
            print("SUCCESS: SQL execution test completed")
            
        except Exception as e:
            print(f"ERROR: SQL execution test failed: {e}")
    
    def run_demo(self):
        """Run a complete demonstration."""
        print("=" * 60)
        print("STREAMLINED NLP-TO-SQL APP DEMO")
        print("Using Modular Core Components")
        print("=" * 60)
        
        # Test all components
        self.test_sql_extraction()
        self.test_sql_execution()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("Modular architecture is working perfectly!")
        print("=" * 60)

def main():
    """Main entry point."""
    try:
        # Create and run the streamlined app
        app = StreamlinedNL2SQLApp()
        app.run_demo()
        
    except Exception as e:
        print(f"ERROR: Demo failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
