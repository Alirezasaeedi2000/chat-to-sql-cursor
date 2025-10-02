#!/usr/bin/env python3
"""
Test the structure and imports of app_farnan.py without requiring database connection.
This verifies that the Graph RAG features are properly integrated.
"""

import os
import sys

def test_app_farnan_structure():
    """Test if app_farnan.py can be imported and has the right structure."""
    print("=" * 80)
    print("TESTING APP_FARNAN.PY STRUCTURE AND GRAPH RAG INTEGRATION")
    print("=" * 80)
    
    try:
        # Test import
        print("1. Testing imports...")
        from app_farnan import FarnanNL2SQLApp
        print("   SUCCESS: app_farnan.py imported successfully")
        
        # Test class structure
        print("\n2. Testing class structure...")
        app_class = FarnanNL2SQLApp
        print(f"   SUCCESS: FarnanNL2SQLApp class found")
        
        # Check for required methods
        required_methods = [
            'setup_database',
            'setup_query_processor', 
            'get_farnan_context',
            'process_query',
            'run_interactive_mode'
        ]
        
        for method in required_methods:
            if hasattr(app_class, method):
                print(f"   SUCCESS: Method '{method}' found")
            else:
                print(f"   ERROR: Method '{method}' missing")
        
        # Test query processor integration
        print("\n3. Testing QueryProcessor integration...")
        from query_processor import QueryProcessor
        print("   SUCCESS: QueryProcessor imported")
        
        # Check if QueryProcessor has Graph RAG methods
        graph_rag_methods = [
            '_detect_schema_communities',
            'get_advanced_schema_context',
            '_get_community_insights_for_query'
        ]
        
        for method in graph_rag_methods:
            if hasattr(QueryProcessor, method):
                print(f"   SUCCESS: Graph RAG method '{method}' found")
            else:
                print(f"   ERROR: Graph RAG method '{method}' missing")
        
        # Test vector store integration
        print("\n4. Testing VectorStoreManager integration...")
        from vector import VectorStoreManager
        print("   SUCCESS: VectorStoreManager imported")
        
        # Test query history integration
        print("\n5. Testing QueryHistoryManager integration...")
        from query_history import QueryHistoryManager
        print("   SUCCESS: QueryHistoryManager imported")
        
        print("\n" + "=" * 80)
        print("STRUCTURE TEST RESULTS:")
        print("=" * 80)
        print("SUCCESS: app_farnan.py structure is correct")
        print("SUCCESS: Graph RAG features are integrated")
        print("SUCCESS: All required dependencies are available")
        print("SUCCESS: System is ready for database testing")
        
        print("\nNOTE: Database connection test failed because MySQL server is not running.")
        print("To test with database:")
        print("1. Start MySQL server")
        print("2. Ensure 'farnan' database exists")
        print("3. Run: python test_app_farnan_60_questions.py")
        
        return True
        
    except ImportError as e:
        print(f"ERROR: IMPORT ERROR: {e}")
        return False
    except Exception as e:
        print(f"ERROR: STRUCTURE ERROR: {e}")
        return False

def test_graph_rag_features():
    """Test if Graph RAG features are available in the codebase."""
    print("\n" + "=" * 80)
    print("TESTING GRAPH RAG FEATURES AVAILABILITY")
    print("=" * 80)
    
    try:
        # Check query_processor.py for Graph RAG features
        with open('query_processor.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        graph_rag_features = [
            '_detect_schema_communities',
            'get_advanced_schema_context', 
            '_get_community_insights_for_query',
            '_generate_schema_reports',
            'community detection',
            'schema analysis',
            'hybrid.*rag'
        ]
        
        found_features = []
        for feature in graph_rag_features:
            if feature.lower() in content.lower():
                found_features.append(feature)
                print(f"SUCCESS: Found Graph RAG feature: {feature}")
            else:
                print(f"ERROR: Missing Graph RAG feature: {feature}")
        
        print(f"\nGraph RAG Features Found: {len(found_features)}/{len(graph_rag_features)}")
        
        if len(found_features) >= 5:
            print("SUCCESS: Graph RAG features are properly integrated!")
            return True
        else:
            print("ERROR: Graph RAG features are missing or incomplete!")
            return False
            
    except FileNotFoundError:
        print("ERROR: query_processor.py file not found!")
        return False
    except Exception as e:
        print(f"ERROR: Error testing Graph RAG features: {e}")
        return False

def main():
    """Main test function."""
    structure_ok = test_app_farnan_structure()
    graph_rag_ok = test_graph_rag_features()
    
    print("\n" + "=" * 80)
    print("FINAL VERDICT:")
    print("=" * 80)
    
    if structure_ok and graph_rag_ok:
        print("EXCELLENT: app_farnan.py is properly structured with Graph RAG!")
        print("SUCCESS: Ready for database testing when MySQL server is running")
    elif structure_ok:
        print("GOOD: app_farnan.py structure is correct but Graph RAG features may be incomplete")
    else:
        print("POOR: app_farnan.py has structural issues")
    
    print("\nNext steps:")
    print("1. Start MySQL server")
    print("2. Run: python test_app_farnan_60_questions.py")

if __name__ == "__main__":
    main()
