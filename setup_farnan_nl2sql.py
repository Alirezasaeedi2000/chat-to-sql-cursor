#!/usr/bin/env python3
"""
Setup Farnan Database for NLP-to-SQL
Configure and optimize the system for the food production database
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any
from sqlalchemy import text

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from farnan_database_config import (
    FARNAN_DATABASE_CONFIG, 
    FARNAN_SCHEMA_METADATA,
    FARNAN_SYNONYMS,
    get_optimized_connection_string,
    get_enhanced_schema_context,
    get_domain_specific_prompts
)
from enhanced_database_manager import EnhancedDatabaseManager
from vector import VectorStoreManager
from query_processor import QueryProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)

def setup_farnan_database():
    """Setup and optimize the Farnan database for NLP-to-SQL"""
    print("🏭 SETTING UP FARNAN DATABASE FOR NLP-to-SQL")
    print("=" * 60)
    
    try:
        # 1. Create optimized database manager
        print("📡 Creating optimized database connection...")
        db_manager = EnhancedDatabaseManager()
        
        # Set up the connection
        connection_string = get_optimized_connection_string()
        engine = db_manager.create_optimized_engine(connection_string, "farnan_db")
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DATABASE()"))
            db_name = result.fetchone()[0]
            print(f"✅ Connected to database: {db_name}")
        
        # 2. Setup vector store with enhanced metadata
        print("\n🔍 Setting up vector store with schema metadata...")
        vector_manager = VectorStoreManager(
            persist_directory="./.vector_store_farnan",
            collection_name="farnan_schema"
        )
        
        # Clear existing data and rebuild
        print("🧹 Clearing existing vector store...")
        # Delete the existing directory to start fresh
        import shutil
        if os.path.exists("./.vector_store_farnan"):
            shutil.rmtree("./.vector_store_farnan")
        print("✅ Cleared existing vector store")
        
        # Add enhanced schema context
        print("📚 Adding enhanced schema context...")
        schema_context = get_enhanced_schema_context()
        vector_manager.add_documents([schema_context], ["schema_context"])
        
        # Add domain-specific knowledge
        print("🎯 Adding domain-specific knowledge...")
        domain_prompts = get_domain_specific_prompts()
        for domain, prompt in domain_prompts.items():
            vector_manager.add_documents([prompt], [f"domain_{domain}"])
        
        # Add synonyms
        print("🔤 Adding synonym mappings...")
        for term, synonyms in FARNAN_SYNONYMS.items():
            synonym_text = f"{term} synonyms: {', '.join(synonyms)}"
            vector_manager.add_documents([synonym_text], [f"synonym_{term}"])
        
        # Index database schema and sample data
        print("📊 Indexing database schema and sample data...")
        sample_count = vector_manager.upsert_schema_and_samples(
            engine, 
            sample_rows_per_table=3
        )
        print(f"✅ Indexed {sample_count} schema and sample documents")
        
        # 3. Setup query processor with domain-specific configuration
        print("\n⚙️ Setting up query processor...")
        query_processor = QueryProcessor(
            engine=engine,
            vector_store_manager=vector_manager,
            enable_hybrid_rag=True,
            enable_analytical_mode=True,
            enable_visualization=True
        )
        
        # Add domain-specific prompts to processor
        print("🎯 Configuring domain-specific prompts...")
        for domain, prompt in domain_prompts.items():
            query_processor.add_domain_prompt(domain, prompt)
        
        # 4. Test the setup with sample queries
        print("\n🧪 Testing setup with sample queries...")
        test_queries = [
            "Show me the production volumes for today",
            "Which employees failed hygiene checks recently?",
            "What are the current ingredient prices?",
            "How much packaging waste was generated this week?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing: '{query}'")
            try:
                result = query_processor.process_query(query)
                if result and result.get('sql'):
                    print(f"✅ Generated SQL: {result['sql'][:100]}...")
                else:
                    print("⚠️ No SQL generated")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # 5. Save configuration
        print("\n💾 Saving configuration...")
        config_data = {
            "database_config": FARNAN_DATABASE_CONFIG,
            "schema_metadata": FARNAN_SCHEMA_METADATA,
            "setup_timestamp": datetime.now().isoformat(),
            "vector_store_path": "./.vector_store_farnan",
            "sample_documents_indexed": sample_count
        }
        
        import json
        with open("farnan_setup_config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        print("✅ Farnan database setup completed successfully!")
        print(f"📁 Vector store location: ./.vector_store_farnan")
        print(f"📄 Configuration saved to: farnan_setup_config.json")
        
        return db_manager, vector_manager, query_processor
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        LOGGER.error(f"Setup error: {e}", exc_info=True)
        return None, None, None

def create_farnan_web_config():
    """Create web server configuration for Farnan database"""
    print("\n🌐 Creating web server configuration...")
    
    web_config = {
        "database_name": "Farnan Food Production System",
        "database_type": "MySQL",
        "description": "Food production and quality control management system",
        "features": [
            "Production tracking and analytics",
            "Hygiene compliance monitoring", 
            "Waste management analysis",
            "Cost and pricing tracking",
            "Quality control reporting"
        ],
        "sample_queries": [
            "Show production volumes by date",
            "Which staff members have hygiene violations?",
            "What are the most expensive ingredients?",
            "How much waste was generated this month?",
            "Show quality test results for recent batches"
        ],
        "dashboard_sections": [
            {
                "title": "Production Analytics",
                "description": "Track production volumes, efficiency, and trends",
                "queries": ["production volumes", "batch performance", "efficiency metrics"]
            },
            {
                "title": "Quality Control",
                "description": "Monitor hygiene compliance and quality testing",
                "queries": ["hygiene violations", "quality tests", "compliance rates"]
            },
            {
                "title": "Waste Management",
                "description": "Analyze waste generation and reduction opportunities",
                "queries": ["waste volumes", "waste types", "reduction trends"]
            },
            {
                "title": "Cost Analysis",
                "description": "Track ingredient costs and pricing trends",
                "queries": ["ingredient prices", "cost trends", "budget analysis"]
            }
        ]
    }
    
    import json
    with open("farnan_web_config.json", "w") as f:
        json.dump(web_config, f, indent=2)
    
    print("✅ Web configuration saved to: farnan_web_config.json")

if __name__ == "__main__":
    print(f"🚀 Starting Farnan Database Setup at {datetime.now()}")
    
    # Setup database and components
    db_manager, vector_manager, query_processor = setup_farnan_database()
    
    if db_manager and vector_manager and query_processor:
        # Create web configuration
        create_farnan_web_config()
        
        print(f"\n🎉 FARNAN DATABASE SETUP COMPLETE!")
        print("=" * 50)
        print("✅ Optimized database connection configured")
        print("✅ Enhanced vector store with domain knowledge")
        print("✅ Query processor with food production context")
        print("✅ Web interface configuration created")
        
        print(f"\n🎯 READY TO USE:")
        print("1. Run: python app.py (for CLI interface)")
        print("2. Run: python web_server.py (for web interface)")
        print("3. Test with food production queries!")
        
    else:
        print("❌ Setup failed. Please check the error messages above.")
    
    print(f"\n✅ Setup completed at {datetime.now()}")
