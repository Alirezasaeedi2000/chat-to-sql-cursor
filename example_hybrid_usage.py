#!/usr/bin/env python3
"""
Example Usage of Hybrid NLP-to-SQL System
Demonstrates the complete workflow for a sample query
"""

import os
import sys
import json
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Example query processing workflow
def demonstrate_hybrid_workflow():
    """Demonstrate the complete hybrid system workflow"""
    
    print("🚀 HYBRID NLP-to-SQL SYSTEM WORKFLOW DEMONSTRATION")
    print("=" * 60)
    
    # Sample query
    query = "How many workers are there?"
    print(f"\n📝 INPUT QUERY: {query}")
    
    # Step 1: Query Classification
    print(f"\n🔍 PHASE 1: QUERY CLASSIFICATION")
    print("-" * 40)
    
    classification_result = {
        "complexity": "simple",
        "type": "scalar", 
        "confidence": 0.95,
        "reasoning": "Query asks for a count of workers - single value expected",
        "handler_type": "simple",
        "fallback_reason": None
    }
    
    print(f"✅ Classification Result:")
    print(f"   Type: {classification_result['type']}")
    print(f"   Complexity: {classification_result['complexity']}")
    print(f"   Confidence: {classification_result['confidence']:.2f}")
    print(f"   Handler: {classification_result['handler_type']}")
    print(f"   Reasoning: {classification_result['reasoning']}")
    
    # Step 2: Handler Selection
    print(f"\n🎯 PHASE 2: HANDLER SELECTION")
    print("-" * 40)
    
    selected_handler = "SimpleQueryHandler"
    print(f"✅ Selected Handler: {selected_handler}")
    print(f"   Reason: Query classified as simple scalar")
    print(f"   Fallback: ComplexQueryHandler (if needed)")
    
    # Step 3: SQL Generation
    print(f"\n🔧 PHASE 3: SQL GENERATION")
    print("-" * 40)
    
    # Try deterministic builder first
    deterministic_sql = "SELECT COUNT(*) FROM `workers` LIMIT 50"
    print(f"✅ Deterministic Builder Result:")
    print(f"   SQL: {deterministic_sql}")
    print(f"   Source: Pattern matching for 'how many workers'")
    
    # Step 4: SQL Validation
    print(f"\n✅ PHASE 4: SQL VALIDATION")
    print("-" * 40)
    
    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": ["Query missing LIMIT clause"]  # This would be caught in real system
    }
    
    print(f"✅ Validation Result:")
    print(f"   Valid: {validation_result['is_valid']}")
    print(f"   Errors: {validation_result['errors']}")
    print(f"   Warnings: {validation_result['warnings']}")
    
    # Step 5: SQL Execution
    print(f"\n⚡ PHASE 5: SQL EXECUTION")
    print("-" * 40)
    
    # Simulated execution result
    execution_result = {
        "data": [{"COUNT(*)": 25}],
        "execution_time_ms": 45,
        "row_count": 1
    }
    
    print(f"✅ Execution Result:")
    print(f"   Data: {execution_result['data']}")
    print(f"   Execution Time: {execution_result['execution_time_ms']}ms")
    print(f"   Row Count: {execution_result['row_count']}")
    
    # Step 6: Result Formatting
    print(f"\n📊 PHASE 6: RESULT FORMATTING")
    print("-" * 40)
    
    formatted_result = {
        "mode": "SHORT_ANSWER",
        "short_answer": "25",
        "sql": deterministic_sql,
        "metadata": {
            "handler_used": "simple",
            "classification": classification_result,
            "confidence": 0.95,
            "deterministic": True,
            "execution_time_ms": execution_result['execution_time_ms'],
            "row_count": execution_result['row_count'],
            "errors": None,
            "code_decision": "continued_existing_code"
        }
    }
    
    print(f"✅ Formatted Result:")
    print(f"   Mode: {formatted_result['mode']}")
    print(f"   Answer: {formatted_result['short_answer']}")
    print(f"   Handler: {formatted_result['metadata']['handler_used']}")
    print(f"   Deterministic: {formatted_result['metadata']['deterministic']}")
    print(f"   Confidence: {formatted_result['metadata']['confidence']:.2f}")
    
    # Final Output
    print(f"\n🎯 FINAL OUTPUT (JSON)")
    print("-" * 40)
    print(json.dumps(formatted_result, indent=2))
    
    return formatted_result


def demonstrate_complex_query():
    """Demonstrate workflow for a complex query"""
    
    print(f"\n\n🔍 COMPLEX QUERY DEMONSTRATION")
    print("=" * 60)
    
    # Complex query
    query = "Show me production volumes as a bar chart"
    print(f"\n📝 INPUT QUERY: {query}")
    
    # Classification
    classification_result = {
        "complexity": "simple",
        "type": "visualization",
        "confidence": 0.88,
        "reasoning": "Query explicitly requests a bar chart visualization",
        "handler_type": "visualization",
        "fallback_reason": None
    }
    
    print(f"\n✅ Classification: {classification_result['type']} ({classification_result['complexity']})")
    print(f"   Handler: {classification_result['handler_type']}")
    print(f"   Confidence: {classification_result['confidence']:.2f}")
    
    # Visualization processing
    print(f"\n📊 Visualization Processing:")
    print(f"   1. Extract chart specifications (bar chart)")
    print(f"   2. Generate SQL for chart data")
    print(f"   3. Execute SQL and get data")
    print(f"   4. Create matplotlib chart")
    print(f"   5. Save chart to outputs/plots/")
    
    # Result
    visualization_result = {
        "mode": "VISUALIZATION",
        "visualization_path": "outputs/plots/viz_20250920_123456_789012.png",
        "table_markdown": "| bakeType | total |\n|----------|-------|\n| Type A   | 150   |\n| Type B   | 200   |",
        "sql": "SELECT `bakeType`, SUM(`totalUsage`) as total FROM `production_info` GROUP BY `bakeType` LIMIT 50",
        "metadata": {
            "handler_used": "visualization",
            "classification": classification_result,
            "confidence": 0.88,
            "chart_type": "bar",
            "chart_title": "Production Volumes by Bake Type",
            "row_count": 2,
            "execution_time_ms": 3200,
            "errors": None,
            "code_decision": "continued_existing_code"
        }
    }
    
    print(f"\n✅ Visualization Result:")
    print(f"   Chart: {visualization_result['visualization_path']}")
    print(f"   Chart Type: {visualization_result['metadata']['chart_type']}")
    print(f"   Data Rows: {visualization_result['metadata']['row_count']}")
    print(f"   Execution Time: {visualization_result['metadata']['execution_time_ms']}ms")
    
    return visualization_result


def main():
    """Main demonstration function"""
    try:
        # Demonstrate simple query workflow
        simple_result = demonstrate_hybrid_workflow()
        
        # Demonstrate complex query workflow  
        complex_result = demonstrate_complex_query()
        
        # Summary
        print(f"\n\n📈 WORKFLOW SUMMARY")
        print("=" * 60)
        print(f"✅ Simple Query: {simple_result['metadata']['execution_time_ms']}ms")
        print(f"✅ Complex Query: {complex_result['metadata']['execution_time_ms']}ms")
        print(f"✅ Both queries processed successfully")
        print(f"✅ Hybrid system demonstrated successfully")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
