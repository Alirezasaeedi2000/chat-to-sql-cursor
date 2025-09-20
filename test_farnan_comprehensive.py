#!/usr/bin/env python3
"""
Comprehensive Test Suite for Farnan NLP-to-SQL System
Tests analytical queries, visualizations, and overall accuracy
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_farnan import FarnanNL2SQLApp

class FarnanTestSuite:
    """Comprehensive test suite for Farnan NLP-to-SQL system"""
    
    def __init__(self):
        self.app = FarnanNL2SQLApp()
        self.test_results = []
        self.start_time = time.time()
        
    def create_test_questions(self) -> List[Dict[str, Any]]:
        """Create 10 comprehensive test questions covering different aspects"""
        return [
            {
                "id": 1,
                "question": "Show me production volumes for this month",
                "expected_type": "scalar",
                "expected_tables": ["production_info"],
                "expected_aggregation": True,
                "description": "Basic scalar aggregation with date filtering"
            },
            {
                "id": 2,
                "question": "Which packaging types are used most frequently? Show me a pie chart",
                "expected_type": "visualization",
                "expected_tables": ["packaging_info"],
                "expected_aggregation": True,
                "expected_chart": "pie",
                "description": "Top/most query with explicit visualization request"
            },
            {
                "id": 3,
                "question": "Analyze hygiene compliance rates by person over the last 6 months",
                "expected_type": "analytical",
                "expected_tables": ["person_hyg"],
                "expected_aggregation": True,
                "expected_grouping": True,
                "description": "Analytical query with time-based grouping"
            },
            {
                "id": 4,
                "question": "What are the most expensive ingredients and how have their prices changed over time?",
                "expected_type": "analytical",
                "expected_tables": ["prices"],
                "expected_aggregation": True,
                "expected_temporal": True,
                "description": "Complex analytical query with temporal analysis"
            },
            {
                "id": 5,
                "question": "Show me waste generation trends by type with a line chart",
                "expected_type": "visualization",
                "expected_tables": ["pack_waste"],
                "expected_chart": "line",
                "expected_temporal": True,
                "description": "Time series visualization with trend analysis"
            },
            {
                "id": 6,
                "question": "Compare production efficiency between different bake types",
                "expected_type": "analytical",
                "expected_tables": ["production_info"],
                "expected_aggregation": True,
                "expected_comparison": True,
                "description": "Comparative analysis between categories"
            },
            {
                "id": 7,
                "question": "What is the average production weight per batch by month? Show as bar chart",
                "expected_type": "visualization",
                "expected_tables": ["production_info"],
                "expected_aggregation": True,
                "expected_grouping": True,
                "expected_chart": "bar",
                "description": "Monthly aggregation with bar chart visualization"
            },
            {
                "id": 8,
                "question": "Identify employees with the most hygiene violations and analyze patterns",
                "expected_type": "analytical",
                "expected_tables": ["person_hyg"],
                "expected_aggregation": True,
                "expected_grouping": True,
                "description": "Employee analysis with pattern identification"
            },
            {
                "id": 9,
                "question": "Show me packaging waste distribution by type with a histogram",
                "expected_type": "visualization",
                "expected_tables": ["pack_waste"],
                "expected_aggregation": True,
                "expected_chart": "histogram",
                "description": "Distribution analysis with histogram visualization"
            },
            {
                "id": 10,
                "question": "Analyze the correlation between production volume and quality test results",
                "expected_type": "analytical",
                "expected_tables": ["production_info", "production_test"],
                "expected_joins": True,
                "expected_correlation": True,
                "description": "Complex analytical query with correlation analysis"
            },
            {
                "id": 11,
                "question": "How many workers are there in total?",
                "expected_type": "scalar",
                "expected_tables": ["workers"],
                "expected_aggregation": True,
                "description": "Simple count query"
            },
            {
                "id": 12,
                "question": "Show me all production batches from last week",
                "expected_type": "table",
                "expected_tables": ["production_info"],
                "expected_filtering": True,
                "description": "Date-filtered table query"
            },
            {
                "id": 13,
                "question": "What is the average ricotta usage per batch?",
                "expected_type": "scalar",
                "expected_tables": ["production_info"],
                "expected_aggregation": True,
                "description": "Average calculation query"
            },
            {
                "id": 14,
                "question": "List all hygiene violations by person with a bar chart",
                "expected_type": "visualization",
                "expected_tables": ["person_hyg"],
                "expected_chart": "bar",
                "expected_grouping": True,
                "description": "Grouped visualization query"
            },
            {
                "id": 15,
                "question": "Which bake types have the highest total usage?",
                "expected_type": "table",
                "expected_tables": ["production_info"],
                "expected_aggregation": True,
                "expected_grouping": True,
                "expected_ordering": True,
                "description": "Grouped aggregation with ordering"
            },
            {
                "id": 16,
                "question": "Show me packaging waste by type over the last 3 months",
                "expected_type": "table",
                "expected_tables": ["pack_waste"],
                "expected_grouping": True,
                "expected_temporal": True,
                "description": "Time-based grouping query"
            },
            {
                "id": 17,
                "question": "What are the top 5 most expensive ingredients?",
                "expected_type": "table",
                "expected_tables": ["prices"],
                "expected_ordering": True,
                "expected_limit": True,
                "description": "Top-N query with ordering"
            },
            {
                "id": 18,
                "question": "Analyze production efficiency trends over time",
                "expected_type": "analytical",
                "expected_tables": ["production_info"],
                "expected_temporal": True,
                "expected_aggregation": True,
                "description": "Temporal analysis query"
            },
            {
                "id": 19,
                "question": "Show me hygiene check results for today",
                "expected_type": "table",
                "expected_tables": ["person_hyg"],
                "expected_filtering": True,
                "description": "Date-filtered hygiene query"
            },
            {
                "id": 20,
                "question": "What is the total waste generated this month?",
                "expected_type": "scalar",
                "expected_tables": ["pack_waste"],
                "expected_aggregation": True,
                "expected_filtering": True,
                "description": "Filtered aggregation query"
            },
            {
                "id": 21,
                "question": "Compare oil usage between different bake types with a pie chart",
                "expected_type": "visualization",
                "expected_tables": ["production_info"],
                "expected_chart": "pie",
                "expected_grouping": True,
                "expected_comparison": True,
                "description": "Comparative visualization"
            },
            {
                "id": 22,
                "question": "List workers by section",
                "expected_type": "table",
                "expected_tables": ["workers"],
                "expected_grouping": True,
                "description": "Simple grouping query"
            },
            {
                "id": 23,
                "question": "What is the maximum production weight recorded?",
                "expected_type": "scalar",
                "expected_tables": ["production_info"],
                "expected_aggregation": True,
                "description": "Maximum value query"
            },
            {
                "id": 24,
                "question": "Show me packaging information for recent batches",
                "expected_type": "table",
                "expected_tables": ["packaging_info"],
                "expected_filtering": True,
                "description": "Recent data query"
            },
            {
                "id": 25,
                "question": "Analyze price trends for cream over the last 6 months",
                "expected_type": "analytical",
                "expected_tables": ["prices"],
                "expected_temporal": True,
                "expected_filtering": True,
                "description": "Temporal price analysis"
            },
            {
                "id": 26,
                "question": "Which ingredients have the lowest prices?",
                "expected_type": "table",
                "expected_tables": ["prices"],
                "expected_ordering": True,
                "expected_limit": True,
                "description": "Lowest value query"
            },
            {
                "id": 27,
                "question": "Show me production test results with a histogram",
                "expected_type": "visualization",
                "expected_tables": ["production_test"],
                "expected_chart": "histogram",
                "description": "Histogram visualization"
            },
            {
                "id": 28,
                "question": "What is the average humidity during production?",
                "expected_type": "scalar",
                "expected_tables": ["production_info"],
                "expected_aggregation": True,
                "description": "Average calculation"
            },
            {
                "id": 29,
                "question": "List all unique bake types used",
                "expected_type": "table",
                "expected_tables": ["production_info"],
                "expected_distinct": True,
                "description": "Distinct values query"
            },
            {
                "id": 30,
                "question": "Show me waste distribution by type with a scatter plot",
                "expected_type": "visualization",
                "expected_tables": ["pack_waste"],
                "expected_chart": "scatter",
                "expected_grouping": True,
                "description": "Scatter plot visualization"
            }
        ]
    
    def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case and return results"""
        print(f"\n{'='*60}")
        print(f"TEST {test_case['id']}: {test_case['question']}")
        print(f"Description: {test_case['description']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Process the query
            result = self.app.process_query(test_case['question'])
            
            execution_time = time.time() - start_time
            
            # Analyze the result
            analysis = self.analyze_result(test_case, result, execution_time)
            
            print(f"✅ Query executed successfully")
            print(f"⏱️ Execution time: {execution_time:.2f}s")
            print(f"📊 Mode: {result.get('mode', 'unknown')}")
            print(f"🎯 Confidence: {result.get('metadata', {}).get('confidence', 0):.2f}")
            
            if result.get('sql'):
                print(f"🔧 SQL Generated: {result['sql'][:100]}...")
            
            # Display results based on mode
            mode = result.get('mode', 'unknown').upper()
            
            if mode == 'SHORT_ANSWER':
                if result.get('short_answer'):
                    print(f"💡 SHORT ANSWER:")
                    print(f"   {result['short_answer']}")
                else:
                    print("⚠️ No short answer provided")
            
            elif mode == 'TABLE':
                if result.get('table_markdown'):
                    print(f"📋 TABLE RESULTS:")
                    print(result['table_markdown'])
                else:
                    print("⚠️ No table data provided")
            
            elif mode == 'ANALYTICAL':
                if result.get('analysis'):
                    print(f"🔍 ANALYTICAL RESULTS:")
                    print(f"   {result['analysis']}")
                else:
                    print("⚠️ No analysis provided")
                
                # Also show table data if available
                if result.get('table_markdown'):
                    print(f"📊 Supporting Data:")
                    print(result['table_markdown'])
            
            elif mode == 'VISUALIZATION':
                if result.get('visualization_path'):
                    print(f"📊 VISUALIZATION GENERATED:")
                    print(f"   File: {result['visualization_path']}")
                else:
                    print("⚠️ No visualization generated")
                
                # Also show table data if available
                if result.get('table_markdown'):
                    print(f"📋 Source Data:")
                    print(result['table_markdown'])
            
            elif mode == 'COMBO':
                print(f"🔄 COMBO MODE RESULTS:")
                
                # Show short answer if available
                if result.get('short_answer'):
                    print(f"💡 Answer: {result['short_answer']}")
                
                # Show analysis if available
                if result.get('analysis'):
                    print(f"🔍 Analysis: {result['analysis']}")
                
                # Show table data if available
                if result.get('table_markdown'):
                    print(f"📋 Data Table:")
                    print(result['table_markdown'])
                
                # Show visualization if available
                if result.get('visualization_path'):
                    print(f"📊 Visualization: {result['visualization_path']}")
            
            else:
                # Fallback: show all available results
                print(f"❓ UNKNOWN MODE ({mode}) - Showing all available results:")
                
                if result.get('short_answer'):
                    print(f"💡 Answer: {result['short_answer']}")
                
                if result.get('table_markdown'):
                    print(f"📋 Table Data:")
                    print(result['table_markdown'])
                
                if result.get('analysis'):
                    print(f"🔍 Analysis: {result['analysis']}")
                
                if result.get('visualization_path'):
                    print(f"📊 Visualization: {result['visualization_path']}")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            return {
                "test_id": test_case['id'],
                "question": test_case['question'],
                "status": "failed",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "accuracy_score": 0
            }
    
    def analyze_result(self, test_case: Dict[str, Any], result: Dict[str, Any], execution_time: float) -> Dict[str, Any]:
        """Analyze the test result and calculate accuracy score"""
        accuracy_score = 0
        max_score = 10
        analysis_details = []
        
        # Check if query executed successfully
        if 'error' not in result:
            accuracy_score += 2
            analysis_details.append("✅ Query executed without errors")
        else:
            analysis_details.append(f"❌ Query failed: {result['error']}")
            return {
                "test_id": test_case['id'],
                "question": test_case['question'],
                "status": "failed",
                "error": result['error'],
                "execution_time": execution_time,
                "accuracy_score": 0,
                "analysis_details": analysis_details
            }
        
        # Check SQL generation
        if result.get('sql'):
            accuracy_score += 1
            analysis_details.append("✅ SQL generated successfully")
            
            # Check for expected tables
            sql_lower = result['sql'].lower()
            expected_tables = test_case.get('expected_tables', [])
            found_tables = [table for table in expected_tables if table.lower() in sql_lower]
            if found_tables:
                accuracy_score += 1
                analysis_details.append(f"✅ Expected tables found: {found_tables}")
            else:
                analysis_details.append(f"⚠️ Expected tables not found: {expected_tables}")
            
            # Check for aggregation
            if test_case.get('expected_aggregation'):
                if any(agg in sql_lower for agg in ['sum(', 'count(', 'avg(', 'max(', 'min(']):
                    accuracy_score += 1
                    analysis_details.append("✅ Aggregation functions found")
                else:
                    analysis_details.append("⚠️ Expected aggregation not found")
            
            # Check for grouping
            if test_case.get('expected_grouping'):
                if 'group by' in sql_lower:
                    accuracy_score += 1
                    analysis_details.append("✅ GROUP BY clause found")
                else:
                    analysis_details.append("⚠️ Expected GROUP BY not found")
            
            # Check for joins
            if test_case.get('expected_joins'):
                if 'join' in sql_lower:
                    accuracy_score += 1
                    analysis_details.append("✅ JOIN clause found")
                else:
                    analysis_details.append("⚠️ Expected JOIN not found")
        else:
            analysis_details.append("❌ No SQL generated")
        
        # Check result mode
        mode = result.get('mode', '')
        expected_type = test_case.get('expected_type', '')
        
        if expected_type == 'scalar' and mode == 'SHORT_ANSWER':
            accuracy_score += 1
            analysis_details.append("✅ Correct mode: SHORT_ANSWER")
        elif expected_type == 'visualization' and mode in ['VISUALIZATION', 'COMBO']:
            accuracy_score += 1
            analysis_details.append("✅ Correct mode: VISUALIZATION/COMBO")
        elif expected_type == 'analytical' and mode in ['ANALYTICAL', 'COMBO']:
            accuracy_score += 1
            analysis_details.append("✅ Correct mode: ANALYTICAL/COMBO")
        else:
            analysis_details.append(f"⚠️ Mode mismatch: expected {expected_type}, got {mode}")
        
        # Check visualization
        if test_case.get('expected_chart') and result.get('visualization_path'):
            accuracy_score += 1
            analysis_details.append(f"✅ Visualization generated: {test_case['expected_chart']}")
        elif test_case.get('expected_chart'):
            analysis_details.append(f"⚠️ Expected visualization not generated: {test_case['expected_chart']}")
        
        # Check data quality
        if result.get('table_markdown') and result['table_markdown'] != '(no rows)':
            accuracy_score += 1
            analysis_details.append("✅ Data returned successfully")
        else:
            analysis_details.append("⚠️ No data returned")
        
        # Check confidence
        confidence = result.get('metadata', {}).get('confidence', 0)
        if confidence > 0.5:
            accuracy_score += 1
            analysis_details.append(f"✅ High confidence: {confidence:.2f}")
        else:
            analysis_details.append(f"⚠️ Low confidence: {confidence:.2f}")
        
        return {
            "test_id": test_case['id'],
            "question": test_case['question'],
            "status": "passed" if accuracy_score >= 7 else "partial" if accuracy_score >= 4 else "failed",
            "accuracy_score": accuracy_score,
            "max_score": max_score,
            "execution_time": execution_time,
            "mode": mode,
            "confidence": confidence,
            "sql_generated": bool(result.get('sql')),
            "data_returned": bool(result.get('table_markdown') and result['table_markdown'] != '(no rows)'),
            "visualization_generated": bool(result.get('visualization_path')),
            "analysis_details": analysis_details
        }
    
    def run_all_tests(self):
        """Run all test cases"""
        print("🧪 FARNAN NLP-to-SQL COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        print(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        test_cases = self.create_test_questions()
        
        for test_case in test_cases:
            result = self.run_single_test(test_case)
            self.test_results.append(result)
            
            # Small delay between tests
            time.sleep(1)
        
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*60}")
        print("📊 COMPREHENSIVE TEST REPORT")
        print(f"{'='*60}")
        
        # Overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['status'] == 'passed')
        partial_tests = sum(1 for r in self.test_results if r['status'] == 'partial')
        failed_tests = sum(1 for r in self.test_results if r['status'] == 'failed')
        
        total_score = sum(r['accuracy_score'] for r in self.test_results)
        max_possible_score = sum(r['max_score'] for r in self.test_results)
        overall_accuracy = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        
        print(f"📈 OVERALL STATISTICS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"   Partial: {partial_tests} ({partial_tests/total_tests*100:.1f}%)")
        print(f"   Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"   Overall Accuracy: {overall_accuracy:.1f}%")
        print(f"   Total Execution Time: {total_time:.2f}s")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            status_emoji = "✅" if result['status'] == 'passed' else "⚠️" if result['status'] == 'partial' else "❌"
            print(f"   {status_emoji} Test {result['test_id']}: {result['accuracy_score']}/{result['max_score']} ({result['accuracy_score']/result['max_score']*100:.1f}%)")
            print(f"      Question: {result['question']}")
            print(f"      Mode: {result.get('mode', 'unknown')}")
            print(f"      Confidence: {result.get('confidence', 0):.2f}")
            print(f"      Time: {result['execution_time']:.2f}s")
            
            # Show key issues
            if result['status'] != 'passed':
                print(f"      Issues:")
                for detail in result.get('analysis_details', []):
                    if detail.startswith('⚠️') or detail.startswith('❌'):
                        print(f"        {detail}")
            print()
        
        # Performance analysis
        print(f"⚡ PERFORMANCE ANALYSIS:")
        avg_execution_time = sum(r['execution_time'] for r in self.test_results) / total_tests
        print(f"   Average execution time: {avg_execution_time:.2f}s")
        
        # Mode distribution
        mode_counts = {}
        for result in self.test_results:
            mode = result.get('mode', 'unknown')
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        print(f"   Mode distribution:")
        for mode, count in mode_counts.items():
            print(f"     {mode}: {count} tests")
        
        # Save detailed report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "partial_tests": partial_tests,
            "failed_tests": failed_tests,
            "overall_accuracy": overall_accuracy,
            "total_execution_time": total_time,
            "average_execution_time": avg_execution_time,
            "mode_distribution": mode_counts,
            "detailed_results": self.test_results
        }
        
        report_file = f"farnan_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        
        # Recommendations
        print(f"\n🔧 RECOMMENDATIONS:")
        if overall_accuracy < 70:
            print("   ⚠️ Overall accuracy is below 70%. Consider:")
            print("     - Reviewing schema context and vector store")
            print("     - Improving prompt engineering")
            print("     - Adding more deterministic builders")
        
        if failed_tests > 0:
            print("   ❌ Some tests failed. Focus on:")
            print("     - SQL generation accuracy")
            print("     - Schema understanding")
            print("     - Error handling")
        
        if avg_execution_time > 10:
            print("   ⏱️ Execution time is high. Consider:")
            print("     - Optimizing vector store queries")
            print("     - Caching frequently used data")
            print("     - Using faster models for simple tasks")

def main():
    """Main function to run the test suite"""
    try:
        test_suite = FarnanTestSuite()
        test_suite.run_all_tests()
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
