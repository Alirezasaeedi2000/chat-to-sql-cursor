#!/usr/bin/env python3
"""
Check Actual Column Names in Farnan Database
"""

import os
import sys
from sqlalchemy import create_engine, text, inspect

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_actual_columns():
    """Check the actual column names in Farnan database"""
    print("🔍 CHECKING ACTUAL COLUMN NAMES IN FARNAN DATABASE")
    print("=" * 60)
    
    connection_string = "mysql+pymysql://root:@localhost:3306/Farnan?charset=utf8mb4&autocommit=true"
    
    try:
        engine = create_engine(connection_string, echo=False)
        inspector = inspect(engine)
        
        # Get all tables
        tables = inspector.get_table_names()
        
        for table_name in tables:
            print(f"\n📋 Table: {table_name}")
            print("-" * 40)
            
            # Get columns
            columns = inspector.get_columns(table_name)
            print("  Columns:")
            for col in columns:
                col_type = str(col['type'])
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                default = f"DEFAULT {col['default']}" if col['default'] else ""
                print(f"    • {col['name']}: {col_type} {nullable} {default}")
            
            # Get sample data
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = result.fetchone()[0]
                print(f"  Records: {count}")
                
                if count > 0:
                    # Get sample row
                    result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 1"))
                    sample = result.fetchone()
                    if sample:
                        print(f"  Sample data: {dict(sample._mapping)}")
        
        print(f"\n✅ Column check complete for {len(tables)} tables")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_actual_columns()
