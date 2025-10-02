#!/usr/bin/env python3
"""
Farnan Database NLP-to-SQL App
Optimized for food production and quality control system
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from query_processor import QueryProcessor
from vector import VectorStoreManager
from query_history import QueryHistoryManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)

class FarnanNL2SQLApp:
    """NLP-to-SQL application for Farnan food production database"""
    
    def __init__(self):
        self.engine = None
        self.query_processor = None
        self.query_history = QueryHistoryManager()
        self.setup_database()
        self.setup_query_processor()
    
    def setup_database(self):
        """Setup optimized connection to Farnan database"""
        print("Setting up Farnan database connection...")
        
        connection_string = "mysql+pymysql://root:@localhost:3306/Farnan?charset=utf8mb4&autocommit=true"
        
        self.engine = create_engine(
            connection_string,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        
        # Test connection
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT DATABASE()"))
            db_name = result.fetchone()[0]
            print(f"Connected to database: {db_name}")
    
    def setup_query_processor(self):
        """Setup query processor with Farnan-specific context"""
        print("Setting up query processor...")
        
        # Create vector store manager
        vector_manager = VectorStoreManager(
            persist_directory="./.vector_store_farnan",
            collection_name="farnan_schema"
        )
        
        # Create query processor
        self.query_processor = QueryProcessor(
            engine=self.engine,
            vector_manager=vector_manager
        )
        
        print("Query processor ready")
    
    def get_farnan_context(self) -> str:
        """Get Farnan-specific context for better SQL generation"""
        return """
        DATABASE: Farnan Food Production System
        DOMAIN: Food Production & Quality Control
        
        ACTUAL TABLE SCHEMAS (USE ONLY THESE COLUMN NAMES):

packaging_info (0 columns):
- Columns: 
- Purpose: Packaging specifications and batch details
- Note: Contains packaging specifications and batch details

packs (1 columns):
- Columns: date
- Purpose: Packaging configurations and specifications

pack_waste (1 columns):
- Columns: date
- Purpose: Packaging waste tracking and analysis
- Note: Use 'type' column for waste types, 'value' column for amounts

person_hyg (1 columns):
- Columns: per_hy_id
- Purpose: Personnel hygiene compliance tracking
- Note: Personnel hygiene compliance tracking with enum values

prices (1 columns):
- Columns: date
- Purpose: Ingredient and material pricing data

production_info (1 columns):
- Columns: bakeID
- Purpose: Main production records and batch data
- Note: Main production records with ingredient usage and quality data

production_test (1 columns):
- Columns: bakeID
- Purpose: Quality testing and control results

repo_nc (1 columns):
- Columns: id
- Purpose: Non-conformance reports and quality issues

transtatus (2 columns):
- Columns: tranNumber, tranStatus
- Purpose: Transaction status and tracking

users (2 columns):
- Columns: id, userId
- Purpose: System users and access control

workers (2 columns):
- Columns: id, firstName
- Purpose: Employee and personnel information
- Note: Employee information and personnel data

CRITICAL: Use ONLY the exact column names listed above. Do not guess or invent column names.
        """
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process a natural language query"""
        print(f"\nProcessing: '{user_query}'")
        
        try:
            # Use the standard process_query method from QueryProcessor
            result_dict = self.query_processor.process_query(user_query)
            
            # Log the query
            self.query_history.add_entry(
                user_query=user_query,
                sql_query=result_dict.get('sql', ''),
                mode=result_dict.get('mode', 'UNKNOWN'),
                execution_time_ms=result_dict.get('metadata', {}).get('execution_time_ms', 0)
            )
            
            return result_dict
                
        except Exception as e:
            LOGGER.error(f"Query processing error: {e}")
            return {"error": str(e)}
    
    def show_sample_queries(self):
        """Show sample queries for the Farnan database"""
        print("\nSAMPLE QUERIES FOR FARNAN DATABASE:")
        print("=" * 50)
        
        sample_queries = [
            "Show me production volumes for today",
            "Which employees failed hygiene checks recently?",
            "What are the current ingredient prices?",
            "How much packaging waste was generated this week?",
            "Show quality test results for recent batches",
            "What is the average production weight per batch?",
            "Which packaging types are used most frequently?",
            "Show hygiene compliance rates by person",
            "What are the most expensive ingredients?",
            "How has waste generation changed over time?"
        ]
        
        for i, query in enumerate(sample_queries, 1):
            print(f"{i:2d}. {query}")
    
    def run_interactive_mode(self):
        """Run interactive mode for querying"""
        print("\nFARNAN FOOD PRODUCTION NLP-to-SQL SYSTEM")
        print("=" * 60)
        print("Ask questions about your food production data!")
        print("Type 'help' for sample queries, 'quit' to exit")
        
        self.show_sample_queries()
        
        while True:
            try:
                # More robust input handling
                try:
                    user_input = input("\nEnter your question: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break
                except Exception:
                    # Fallback for input issues
                    print("\nInput error. Please try again or type 'quit' to exit.")
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self.show_sample_queries()
                    continue
                elif not user_input:
                    continue
                
                # Process the query
                result = self.process_query(user_input)
                
                if 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    # Display results
                    self.display_result(result)
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                # Don't break on errors, just continue
                continue
    
    def display_result(self, result: Dict[str, Any]):
        """Display query results in a formatted way"""
        mode = result.get('mode', 'unknown').upper()
        print(f"\nRESULT ({mode}):")
        print("-" * 40)
        
        # Show the actual answer based on mode
        if mode == 'SHORT_ANSWER' and result.get('short_answer'):
            print(f"Answer: {result['short_answer']}")
        elif mode == 'TABLE' and result.get('table_markdown'):
            print("Data Table:")
            print(result['table_markdown'])
        elif mode == 'ANALYTICAL' and result.get('analysis'):
            print(f"Analysis: {result['analysis']}")
        elif mode == 'VISUALIZATION' and result.get('visualization_path'):
            print(f"Visualization saved to: {result['visualization_path']}")
        
        # Show technical details
        if result.get('sql'):
            print(f"\nSQL: {result['sql']}")
        
        # Show metadata
        metadata = result.get('metadata', {})
        if metadata.get('confidence'):
            print(f"Confidence: {metadata['confidence']:.2f}")
        if metadata.get('execution_time_ms'):
            print(f"Execution time: {metadata['execution_time_ms']/1000:.2f}s")
        if metadata.get('row_count'):
            print(f"Rows returned: {metadata['row_count']}")
        
        # Show analysis if available
        if result.get('analysis') and mode != 'ANALYTICAL':
            print(f"\nAdditional Analysis: {result['analysis']}")
        
        # Show visualization if available
        if result.get('visualization_path') and mode != 'VISUALIZATION':
            print(f"\nVisualization: {result['visualization_path']}")

def main():
    """Main function"""
    try:
        app = FarnanNL2SQLApp()
        app.run_interactive_mode()
    except Exception as e:
        print(f"Application error: {e}")
        LOGGER.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
