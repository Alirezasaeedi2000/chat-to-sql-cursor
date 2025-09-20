#!/usr/bin/env python3
"""
Test Suite for Hybrid NLP-to-SQL System
Comprehensive testing of query classification, handler routing, and result generation
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo_hybrid_system import HybridSystemDemo


class HybridSystemTester:
    """Comprehensive test suite for hybrid system"""
    
    def __init__(self):
        self.demo = HybridSystemDemo()
        self.test_results = []
        self.start_time = time.time()
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """Create comprehensive test cases"""
        return [
            # Simple scalar queries
            {
                "id": 1,
                "query": "How many workers are there?",
                "expected_handler": "simple",
                "expected_mode": "SHORT_ANSWER",
                "expected_type": "scalar",
                "description": "Simple count query"
            },
            {
                "id": 2,
                "query": "What is the total production volume for today?",
                "expected_handler": "simple",
                "expected_mode": "SHORT_ANSWER", 
                "expected_type": "scalar",
                "description": "Simple aggregation query"
            },
            
            # Simple table queries
            {
                "id": 3,
                "query": "Show me all workers",
                "expected_handler": "simple",
                "expected_mode": "TABLE",
                "expected_type": "table",
                "description": "Simple table listing"
            },
            {
                "id": 4,
                "query": "List production batches from last week",
                "expected_handler": "simple",
                "expected_mode": "TABLE",
                "expected_type": "table", 
                "description": "Date-filtered table query"
            },
            
            # Visualization queries
            {
                "id": 5,
                "query": "Show production volumes as a bar chart",
                "expected_handler": "visualization",
                "expected_mode": "VISUALIZATION",
                "expected_type": "visualization",
                "description": "Bar chart visualization"
            },
            {
                "id": 6,
                "query": "Create a pie chart of waste types",
                "expected_handler": "visualization",
                "expected_mode": "VISUALIZATION",
                "expected_type": "visualization",
                "description": "Pie chart visualization"
            },
            
            # Analytical queries
            {
                "id": 7,
                "query": "Analyze hygiene compliance trends over the last 6 months",
                "expected_handler": "analytical",
                "expected_mode": "ANALYTICAL",
                "expected_type": "analytical",
                "description": "Trend analysis query"
            },
            {
                "id": 8,
                "query": "Compare production efficiency between different bake types",
                "expected_handler": "analytical",
                "expected_mode": "ANALYTICAL",
                "expected_type": "analytical",
                "description": "Comparative analysis query"
            },
            
            # Complex queries
            {
                "id": 9,
                "query": "Show me production volumes for today and create a bar chart of the results",
                "expected_handler": "complex",
                "expected_mode": "COMBO",
                "expected_type": "complex",
                "description": "Multi-requirement complex query"
            },
            {
                "id": 10,
                "query": "Analyze hygiene compliance rates by person and identify the top 3 violators with their violation details",
                "expected_handler": "complex",
                "expected_mode": "ANALYTICAL",
                "expected_type": "complex",
                "description": "Multi-step analytical query"
            }
        ]
    
    def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        print(f"\n{'='*60}")
        print(f"TEST {test_case['id']}: {test_case['query']}")
        print(f"Expected: {test_case['expected_handler']} handler → {test_case['expected_mode']} mode")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Process query through hybrid system
            result = self.demo.system.process_query(test_case['query'])
            
            execution_time = time.time() - start_time
            
            # Analyze result
            analysis = self.analyze_test_result(test_case, result, execution_time)
            
            print(f"✅ Query processed successfully")
            print(f"⏱️ Execution time: {execution_time:.2f}s")
            
            # Show classification
            metadata = result.get('metadata', {})
            classification = metadata.get('classification', {})
            print(f"🎯 Classification: {classification.get('type', 'unknown')} ({classification.get('complexity', 'unknown')})")
            print(f"🎯 Confidence: {classification.get('confidence', 0):.2f}")
            print(f"🎯 Handler used: {metadata.get('handler_used', 'unknown')}")
            
            # Show result
            mode = result.get('mode', 'unknown')
            print(f"📊 Result mode: {mode}")
            
            if mode == 'SHORT_ANSWER' and result.get('short_answer'):
                print(f"💡 Answer: {result['short_answer']}")
            elif mode == 'TABLE' and result.get('table_markdown'):
                print("📋 Data table generated")
            elif mode == 'VISUALIZATION' and result.get('visualization_path'):
                print(f"📊 Visualization: {result['visualization_path']}")
            elif mode == 'ERROR':
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            # Show SQL
            if result.get('sql'):
                print(f"🔧 SQL: {result['sql'][:100]}...")
            
            return {
                'test_case': test_case,
                'result': result,
                'execution_time': execution_time,
                'analysis': analysis,
                'status': 'completed'
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Test failed: {e}")
            
            return {
                'test_case': test_case,
                'error': str(e),
                'execution_time': execution_time,
                'status': 'failed'
            }
    
    def analyze_test_result(self, test_case: Dict[str, Any], result: Dict[str, Any], execution_time: float) -> Dict[str, Any]:
        """Analyze test result and calculate accuracy score"""
        score = 0
        max_score = 10
        analysis_details = []
        
        # Check execution success
        if 'error' not in result and result.get('mode') != 'ERROR':
            score += 2
            analysis_details.append("✅ Query executed successfully")
        else:
            analysis_details.append(f"❌ Query failed: {result.get('error', 'Unknown error')}")
            return {
                'score': 0,
                'max_score': max_score,
                'accuracy': 0.0,
                'details': analysis_details
            }
        
        # Check handler selection
        expected_handler = test_case.get('expected_handler')
        actual_handler = result.get('metadata', {}).get('handler_used')
        if expected_handler == actual_handler:
            score += 2
            analysis_details.append(f"✅ Correct handler: {actual_handler}")
        else:
            analysis_details.append(f"⚠️ Handler mismatch: expected {expected_handler}, got {actual_handler}")
        
        # Check mode selection
        expected_mode = test_case.get('expected_mode')
        actual_mode = result.get('mode')
        if expected_mode == actual_mode:
            score += 2
            analysis_details.append(f"✅ Correct mode: {actual_mode}")
        else:
            analysis_details.append(f"⚠️ Mode mismatch: expected {expected_mode}, got {actual_mode}")
        
        # Check SQL generation
        if result.get('sql'):
            score += 1
            analysis_details.append("✅ SQL generated")
        else:
            analysis_details.append("❌ No SQL generated")
        
        # Check result content
        if result.get('short_answer') or result.get('table_markdown') or result.get('visualization_path'):
            score += 2
            analysis_details.append("✅ Result content generated")
        else:
            analysis_details.append("❌ No result content")
        
        # Check confidence
        confidence = result.get('metadata', {}).get('classification', {}).get('confidence', 0)
        if confidence > 0.7:
            score += 1
            analysis_details.append(f"✅ High confidence: {confidence:.2f}")
        elif confidence > 0.5:
            analysis_details.append(f"⚠️ Medium confidence: {confidence:.2f}")
        else:
            analysis_details.append(f"❌ Low confidence: {confidence:.2f}")
        
        return {
            'score': score,
            'max_score': max_score,
            'accuracy': score / max_score,
            'details': analysis_details
        }
    
    def run_all_tests(self):
        """Run all test cases"""
        test_cases = self.create_test_cases()
        
        print("\n🧪 HYBRID SYSTEM COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        print(f"Running {len(test_cases)} test cases...")
        
        for test_case in test_cases:
            result = self.run_single_test(test_case)
            self.test_results.append(result)
            
            # Small delay between tests
            time.sleep(0.5)
        
        # Generate summary
        self.generate_test_summary()
    
    def generate_test_summary(self):
        """Generate comprehensive test summary"""
        print(f"\n{'='*60}")
        print("📊 TEST SUMMARY")
        print(f"{'='*60}")
        
        total_tests = len(self.test_results)
        completed_tests = sum(1 for r in self.test_results if r['status'] == 'completed')
        failed_tests = sum(1 for r in self.test_results if r['status'] == 'failed')
        
        print(f"Total tests: {total_tests}")
        print(f"Completed: {completed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {completed_tests/total_tests*100:.1f}%")
        
        # Calculate average accuracy
        valid_results = [r for r in self.test_results if r['status'] == 'completed' and 'analysis' in r]
        if valid_results:
            avg_accuracy = sum(r['analysis']['accuracy'] for r in valid_results) / len(valid_results)
            print(f"Average accuracy: {avg_accuracy*100:.1f}%")
        
        # Calculate average execution time
        avg_execution_time = sum(r['execution_time'] for r in self.test_results) / len(self.test_results)
        print(f"Average execution time: {avg_execution_time:.2f}s")
        
        # Handler usage statistics
        handler_usage = {}
        for result in self.test_results:
            if result['status'] == 'completed' and 'result' in result:
                handler = result['result'].get('metadata', {}).get('handler_used', 'unknown')
                handler_usage[handler] = handler_usage.get(handler, 0) + 1
        
        print(f"\nHandler usage:")
        for handler, count in handler_usage.items():
            print(f"  {handler}: {count}")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            test_case = result['test_case']
            if result['status'] == 'completed' and 'analysis' in result:
                analysis = result['analysis']
                status_icon = "✅" if analysis['accuracy'] > 0.7 else "⚠️" if analysis['accuracy'] > 0.5 else "❌"
                print(f"{status_icon} Test {test_case['id']}: {analysis['accuracy']*100:.1f}% - {test_case['description']}")
            else:
                print(f"❌ Test {test_case['id']}: FAILED - {test_case['description']}")
        
        # Save results to file
        self.save_test_results()
    
    def save_test_results(self):
        """Save test results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hybrid_test_results_{timestamp}.json"
        
        # Prepare results for JSON serialization
        serializable_results = []
        for result in self.test_results:
            serializable_result = {
                'test_case': result['test_case'],
                'execution_time': result['execution_time'],
                'status': result['status']
            }
            
            if 'analysis' in result:
                serializable_result['analysis'] = result['analysis']
            
            if 'error' in result:
                serializable_result['error'] = result['error']
            
            if 'result' in result:
                # Simplify result for JSON
                result_data = result['result']
                serializable_result['result'] = {
                    'mode': result_data.get('mode'),
                    'short_answer': result_data.get('short_answer'),
                    'sql': result_data.get('sql'),
                    'metadata': {
                        'handler_used': result_data.get('metadata', {}).get('handler_used'),
                        'confidence': result_data.get('metadata', {}).get('classification', {}).get('confidence'),
                        'execution_time_ms': result_data.get('metadata', {}).get('execution_time_ms')
                    }
                }
            
            serializable_results.append(serializable_result)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_tests': len(self.test_results),
                'results': serializable_results
            }, f, indent=2)
        
        print(f"\n💾 Test results saved to: {filename}")


def main():
    """Main function"""
    try:
        tester = HybridSystemTester()
        tester.run_all_tests()
    
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
