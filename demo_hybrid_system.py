#!/usr/bin/env python3
"""
Demonstration of Hybrid NLP-to-SQL System
Shows the complete workflow from query classification to result generation
"""

import os
import sys
import logging
import time
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from vector import VectorStoreManager
from hybrid_nl2sql_system import HybridNL2SQLSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)


class HybridSystemDemo:
    """Demonstration class for the hybrid NL2SQL system"""
    
    def __init__(self):
        self.system = None
        self.setup_system()
    
    def setup_system(self):
        """Setup the hybrid system"""
        print("🚀 Setting up Hybrid NL2SQL System...")
        
        try:
            # Setup database connection
            connection_string = "mysql+pymysql://root:@localhost:3306/Farnan?charset=utf8mb4&autocommit=true"
            engine = create_engine(
                connection_string,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT DATABASE()"))
                db_name = result.fetchone()[0]
                print(f"✅ Connected to database: {db_name}")
            
            # Setup vector store manager
            vector_manager = VectorStoreManager(
                persist_directory="./.vector_store_farnan",
                collection_name="farnan_schema"
            )
            
            # Initialize hybrid system
            self.system = HybridNL2SQLSystem(engine, vector_manager)
            
            print("✅ Hybrid system ready")
            
        except Exception as e:
            print(f"❌ System setup failed: {e}")
            raise
    
    def demonstrate_query_processing(self, query: str) -> Dict[str, Any]:
        """Demonstrate complete query processing workflow"""
        print(f"\n{'='*60}")
        print(f"🔍 PROCESSING QUERY: {query}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Process query through hybrid system
            result = self.system.process_query(query)
            
            execution_time = time.time() - start_time
            
            # Display results
            self.display_result(result, execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Query processing failed: {e}")
            return {'error': str(e), 'execution_time': execution_time}
    
    def display_result(self, result: Dict[str, Any], execution_time: float):
        """Display query result in formatted way"""
        print(f"\n📊 RESULT:")
        print(f"⏱️ Execution time: {execution_time:.2f}s")
        
        # Show classification info
        metadata = result.get('metadata', {})
        classification = metadata.get('classification', {})
        
        print(f"🎯 Classification:")
        print(f"   Type: {classification.get('type', 'unknown')}")
        print(f"   Complexity: {classification.get('complexity', 'unknown')}")
        print(f"   Confidence: {classification.get('confidence', 0):.2f}")
        print(f"   Handler: {metadata.get('handler_used', 'unknown')}")
        
        # Show result content
        mode = result.get('mode', 'unknown').upper()
        print(f"\n📋 RESULT TYPE: {mode}")
        
        if mode == 'SHORT_ANSWER' and result.get('short_answer'):
            print(f"💡 Answer: {result['short_answer']}")
        elif mode == 'TABLE' and result.get('table_markdown'):
            print("📋 Data Table:")
            print(result['table_markdown'])
        elif mode == 'VISUALIZATION' and result.get('visualization_path'):
            print(f"📊 Visualization: {result['visualization_path']}")
            if result.get('table_markdown'):
                print("📋 Source Data:")
                print(result['table_markdown'])
        elif mode == 'ERROR':
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        else:
            print("❓ Unknown result format")
            print(f"Raw result: {result}")
        
        # Show SQL
        if result.get('sql'):
            print(f"\n🔧 SQL Generated:")
            print(result['sql'])
        
        # Show metadata
        if metadata.get('errors'):
            print(f"\n⚠️ Errors: {metadata['errors']}")
        
        if metadata.get('fallback_reason'):
            print(f"🔄 Fallback reason: {metadata['fallback_reason']}")
    
    def run_demo_queries(self):
        """Run demonstration queries"""
        demo_queries = [
            # Simple scalar queries
            "How many workers are there?",
            "What is the total production volume for today?",
            
            # Simple table queries  
            "Show me all workers",
            "List production batches from last week",
            
            # Visualization queries
            "Show production volumes as a bar chart",
            "Create a pie chart of waste types",
            
            # Analytical queries
            "Analyze hygiene compliance trends",
            "Compare production efficiency between different bake types",
            
            # Complex queries
            "Show me production volumes for today and create a bar chart of the results"
        ]
        
        print("\n🎯 DEMONSTRATION QUERIES")
        print("=" * 60)
        
        results = []
        for i, query in enumerate(demo_queries, 1):
            print(f"\n📝 Query {i}/{len(demo_queries)}")
            result = self.demonstrate_query_processing(query)
            results.append(result)
            
            # Small delay between queries
            time.sleep(1)
        
        # Summary
        self.display_summary(results)
    
    def display_summary(self, results: list):
        """Display summary of all results"""
        print(f"\n{'='*60}")
        print("📈 DEMONSTRATION SUMMARY")
        print(f"{'='*60}")
        
        total_queries = len(results)
        successful_queries = sum(1 for r in results if 'error' not in r and r.get('mode') != 'ERROR')
        
        print(f"Total queries: {total_queries}")
        print(f"Successful: {successful_queries}")
        print(f"Failed: {total_queries - successful_queries}")
        print(f"Success rate: {successful_queries/total_queries*100:.1f}%")
        
        # Handler usage
        handler_usage = {}
        for result in results:
            handler = result.get('metadata', {}).get('handler_used', 'unknown')
            handler_usage[handler] = handler_usage.get(handler, 0) + 1
        
        print(f"\nHandler usage:")
        for handler, count in handler_usage.items():
            print(f"  {handler}: {count}")
        
        # System stats
        if self.system:
            stats = self.system.get_system_stats()
            print(f"\nSystem statistics:")
            print(f"  Average execution time: {stats.get('average_execution_time', 0):.2f}s")
            print(f"  Success rate: {stats.get('success_rate', 0)*100:.1f}%")
    
    def run_interactive_mode(self):
        """Run interactive mode for testing"""
        print("\n🎮 INTERACTIVE MODE")
        print("=" * 60)
        print("Type 'quit' to exit, 'stats' for system statistics, 'health' for health check")
        
        while True:
            try:
                user_input = input("\n🔍 Enter your query: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'stats':
                    if self.system:
                        stats = self.system.get_system_stats()
                        print(f"\n📊 System Statistics:")
                        print(f"Total queries: {stats.get('total_queries', 0)}")
                        print(f"Success rate: {stats.get('success_rate', 0)*100:.1f}%")
                        print(f"Average execution time: {stats.get('average_execution_time', 0):.2f}s")
                        print(f"Handler usage: {stats.get('handler_usage', {})}")
                    else:
                        print("❌ System not initialized")
                    continue
                elif user_input.lower() == 'health':
                    if self.system:
                        health = self.system.health_check()
                        print(f"\n🏥 System Health: {health['overall']}")
                        for component, status in health['components'].items():
                            print(f"  {component}: {status}")
                    else:
                        print("❌ System not initialized")
                    continue
                elif not user_input:
                    continue
                
                # Process query
                self.demonstrate_query_processing(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")


def main():
    """Main function"""
    try:
        demo = HybridSystemDemo()
        
        # Ask user what they want to do
        print("\n🎯 HYBRID NL2SQL SYSTEM DEMO")
        print("=" * 60)
        print("Choose an option:")
        print("1. Run demonstration queries")
        print("2. Interactive mode")
        print("3. Health check")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            demo.run_demo_queries()
        elif choice == '2':
            demo.run_interactive_mode()
        elif choice == '3':
            health = demo.system.health_check()
            print(f"\n🏥 System Health: {health['overall']}")
            for component, status in health['components'].items():
                print(f"  {component}: {status}")
        else:
            print("❌ Invalid choice")
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        LOGGER.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
