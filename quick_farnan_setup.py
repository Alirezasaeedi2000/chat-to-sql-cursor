#!/usr/bin/env python3
"""
Quick Farnan Database Setup - Simple approach to get your database working
"""

import os
import sys
import time
from datetime import datetime
from sqlalchemy import create_engine, text

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def quick_farnan_setup():
    """Quick setup for Farnan database without complex vector store management"""
    print("🏭 QUICK FARNAN DATABASE SETUP")
    print("=" * 50)
    
    # 1. Test database connection
    print("📡 Testing database connection...")
    connection_string = "mysql+pymysql://root:@localhost:3306/Farnan?charset=utf8mb4&autocommit=true"
    
    try:
        engine = create_engine(
            connection_string,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DATABASE()"))
            db_name = result.fetchone()[0]
            print(f"✅ Connected to database: {db_name}")
            
            # Get table count
            result = conn.execute(text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = DATABASE()"))
            table_count = result.fetchone()[0]
            print(f"📊 Found {table_count} tables")
        
        # 2. Create a simple configuration file
        print("\n💾 Creating configuration file...")
        config = {
            "database_name": "Farnan",
            "connection_string": connection_string,
            "domain": "Food Production & Quality Control",
            "description": "Production management system for food manufacturing",
            "tables": [
                "production_info", "person_hyg", "packaging_info", "pack_waste",
                "production_test", "packs", "prices", "repo_nc", "transtatus", "users", "workers"
            ],
            "setup_timestamp": datetime.now().isoformat()
        }
        
        import json
        with open("farnan_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("✅ Configuration saved to: farnan_config.json")
        
        # 3. Test a simple query
        print("\n🧪 Testing sample query...")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) as total_rows FROM production_info"))
            count = result.fetchone()[0]
            print(f"✅ Production info table has {count} rows")
        
        return engine
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def create_simple_app_config():
    """Create a simple app configuration for Farnan database"""
    print("\n⚙️ Creating app configuration...")
    
    app_config = f"""
# Farnan Database Configuration
DATABASE_URL = "mysql+pymysql://root:@localhost:3306/Farnan?charset=utf8mb4&autocommit=true"

# Database Settings
DB_POOL_SIZE = 10
DB_MAX_OVERFLOW = 20
DB_POOL_RECYCLE = 3600

# Domain-specific settings
DOMAIN = "Food Production & Quality Control"
ENABLE_ANALYTICS = True
ENABLE_VISUALIZATION = True

# Sample queries for testing
SAMPLE_QUERIES = [
    "Show me production volumes for today",
    "Which employees failed hygiene checks recently?", 
    "What are the current ingredient prices?",
    "How much packaging waste was generated this week?",
    "Show quality test results for recent batches"
]
"""
    
    with open("farnan_app_config.py", "w") as f:
        f.write(app_config)
    
    print("✅ App configuration saved to: farnan_app_config.py")

def test_basic_queries(engine):
    """Test basic queries on the Farnan database"""
    print("\n🔍 Testing basic queries...")
    
    test_queries = [
        "SELECT COUNT(*) as total_production_records FROM production_info",
        "SELECT COUNT(*) as total_hygiene_records FROM person_hyg", 
        "SELECT COUNT(*) as total_packaging_records FROM packaging_info",
        "SELECT COUNT(*) as total_waste_records FROM pack_waste"
    ]
    
    for query in test_queries:
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                row = result.fetchone()
                print(f"✅ {query.split('FROM')[1].strip()}: {row[0]} records")
        except Exception as e:
            print(f"❌ Query failed: {e}")

if __name__ == "__main__":
    print(f"🚀 Starting Quick Farnan Setup at {datetime.now()}")
    
    # Setup database
    engine = quick_farnan_setup()
    
    if engine:
        # Create app configuration
        create_simple_app_config()
        
        # Test basic queries
        test_basic_queries(engine)
        
        print(f"\n🎉 QUICK SETUP COMPLETE!")
        print("=" * 40)
        print("✅ Database connection working")
        print("✅ Configuration files created")
        print("✅ Basic queries tested")
        
        print(f"\n🎯 NEXT STEPS:")
        print("1. Update your main app.py to use the Farnan connection")
        print("2. Test NLP-to-SQL queries")
        print("3. Run: python app.py")
        
        # Close connection
        engine.dispose()
        
    else:
        print("❌ Setup failed. Please check your database connection.")
    
    print(f"\n✅ Setup completed at {datetime.now()}")
