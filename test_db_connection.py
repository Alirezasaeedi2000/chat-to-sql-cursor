#!/usr/bin/env python3
"""
Test script to verify database connection and schema
"""

from app_farnan import FarnanNL2SQLApp
from sqlalchemy import text

def test_database_connection():
    print("🔍 Testing Farnan Database Connection...")
    print("=" * 50)
    
    try:
        # Initialize the app
        app = FarnanNL2SQLApp()
        
        # Get engine info
        engine = app.query_processor.engine
        print(f"✅ Engine created: {engine}")
        print(f"📊 Database URL: {engine.url}")
        
        # Test connection
        with engine.connect() as conn:
            # Check current database
            result = conn.execute(text("SELECT DATABASE() as current_db"))
            current_db = result.fetchone()[0]
            print(f"🗄️  Current database: {current_db}")
            
            # List all tables
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result.fetchall()]
            print(f"📋 Tables in database ({len(tables)}): {tables}")
            
            # Check if we have the expected Farnan tables
            expected_tables = ['employe', 'production_info', 'production_test', 'hygiene_check', 
                             'packaging_info', 'waste_info', 'prices']
            
            missing_tables = [t for t in expected_tables if t not in tables]
            if missing_tables:
                print(f"⚠️  Missing expected tables: {missing_tables}")
            else:
                print("✅ All expected Farnan tables found!")
            
            # Test a simple query
            result = conn.execute(text("SELECT COUNT(*) as total_employees FROM employe"))
            employee_count = result.fetchone()[0]
            print(f"👥 Total employees: {employee_count}")
            
            # Test production_info table
            result = conn.execute(text("SELECT COUNT(*) as total_production_records FROM production_info"))
            production_count = result.fetchone()[0]
            print(f"🏭 Total production records: {production_count}")
            
        print("\n✅ Database connection test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Database connection test FAILED: {e}")
        return False

if __name__ == "__main__":
    test_database_connection()
