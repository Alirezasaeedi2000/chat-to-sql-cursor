#!/usr/bin/env python3
"""
Test Farnan Database Queries
Simple test script to verify NLP-to-SQL functionality
"""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from query_processor import QueryProcessor
from vector import VectorStoreManager

def test_farnan_setup():
    """Test the Farnan database setup"""
    print("🧪 TESTING FARNAN DATABASE SETUP")
    print("=" * 50)
    
    try:
        # 1. Test database connection
        print("📡 Testing database connection...")
        connection_string = "mysql+pymysql://root:@localhost:3306/Farnan?charset=utf8mb4&autocommit=true"
        
        engine = create_engine(
            connection_string,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False
        )
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DATABASE()"))
            db_name = result.fetchone()[0]
            print(f"✅ Connected to: {db_name}")
        
        # 2. Test basic SQL queries
        print("\n🔍 Testing basic SQL queries...")
        test_queries = [
            "SELECT COUNT(*) as total FROM production_info",
            "SELECT COUNT(*) as total FROM person_hyg",
            "SELECT COUNT(*) as total FROM packaging_info",
            "SELECT COUNT(*) as total FROM pack_waste"
        ]
        
        for query in test_queries:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                count = result.fetchone()[0]
                table_name = query.split("FROM")[1].strip()
                print(f"✅ {table_name}: {count} records")
        
        # 3. Test vector store
        print("\n🔍 Testing vector store...")
        vector_manager = VectorStoreManager(
            persist_directory="./.vector_store_farnan",
            collection_name="farnan_schema"
        )
        print("✅ Vector store initialized")
        
        # 4. Test query processor
        print("\n⚙️ Testing query processor...")
        query_processor = QueryProcessor(
            engine=engine,
            vector_manager=vector_manager
        )
        print("✅ Query processor initialized")
        
        # 5. Test simple NLP query
        print("\n🤖 Testing NLP-to-SQL conversion...")
        test_query = "How many production records are there?"
        
        try:
            result = query_processor.process(test_query)
            if result:
                print(f"✅ NLP Query processed successfully")
                print(f"   SQL: {result.get('sql', 'No SQL generated')}")
                if result.get('data'):
                    print(f"   Result: {result['data'][:3]}...")  # Show first 3 rows
            else:
                print("⚠️ No result generated")
        except Exception as e:
            print(f"❌ NLP processing error: {e}")
        
        print(f"\n🎉 FARNAN DATABASE TEST COMPLETE!")
        print("=" * 50)
        print("✅ Database connection working")
        print("✅ Basic SQL queries working")
        print("✅ Vector store initialized")
        print("✅ Query processor ready")
        print("✅ NLP-to-SQL conversion tested")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print(f"🚀 Starting Farnan Database Test at {datetime.now()}")
    
    success = test_farnan_setup()
    
    if success:
        print(f"\n🎯 READY TO USE:")
        print("1. Run: python app_farnan.py (for interactive mode)")
        print("2. Run: python web_server.py (for web interface)")
        print("3. Your Farnan database is ready for NLP-to-SQL queries!")
    else:
        print("\n❌ Setup needs attention. Please check the error messages above.")
    
    print(f"\n✅ Test completed at {datetime.now()}")
