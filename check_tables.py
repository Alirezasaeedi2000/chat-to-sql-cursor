#!/usr/bin/env python3
"""
Check table structures in Farnan database
"""

from app_farnan import FarnanNL2SQLApp
from sqlalchemy import text

def check_table_structure():
    print("🔍 Checking Farnan Database Table Structures...")
    print("=" * 60)
    
    try:
        app = FarnanNL2SQLApp()
        engine = app.query_processor.engine
        
        # Tables to check
        tables_to_check = ['workers', 'production_info', 'production_test', 'person_hyg', 'pack_waste', 'packaging_info', 'prices']
        
        with engine.connect() as conn:
            for table_name in tables_to_check:
                try:
                    result = conn.execute(text(f'DESCRIBE {table_name}'))
                    columns = result.fetchall()
                    print(f"\n📋 Table: {table_name}")
                    print("-" * 40)
                    for col in columns:
                        print(f"  {col[0]:<20} {col[1]:<15} {'NULL' if col[2] == 'YES' else 'NOT NULL'}")
                except Exception as e:
                    print(f"\n❌ Table {table_name} not found or error: {e}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_table_structure()
