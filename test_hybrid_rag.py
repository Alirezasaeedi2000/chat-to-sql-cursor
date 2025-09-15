#!/usr/bin/env python3
"""
Comprehensive test script for Hybrid Graph RAG + Vector RAG system.
Tests various capabilities including schema understanding, pattern recognition,
clarification logic, and visualization mode detection.
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from query_processor import QueryProcessor
from vector import VectorStoreManager
from sqlalchemy import create_engine

class HybridRAGTester:
    def __init__(self, db_url: str):
        """Initialize the tester with database connection."""
        self.db_url = db_url
        self.processor = None
        self.vector_manager = None
        self.results = []
        
    def setup(self):
        """Setup the query processor and vector store."""
        print("🔧 Setting up Hybrid RAG system...")
        try:
            # Create database engine
            engine = create_engine(self.db_url)
            
            # Initialize vector store
            self.vector_manager = VectorStoreManager()
            
            # Initialize query processor
            self.processor = QueryProcessor(
                engine=engine,
                vector_manager=self.vector_manager
            )
            print("✅ Setup complete!")
            return True
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def run_query(self, query: str, category: str, expected_mode: str = None) -> Dict[str, Any]:
        """Run a single query and capture results."""
        print(f"\n🔍 Testing: {query}")
        print(f"📂 Category: {category}")
        
        start_time = time.time()
        try:
            result = self.processor.process(query)
            end_time = time.time()
            
            # Handle NL2SQLOutput object properly
            mode = getattr(result, 'mode', 'UNKNOWN')
            sql = getattr(result, 'sql', '')
            error = getattr(result, 'error', '')
            table_markdown = getattr(result, 'table_markdown', '')
            short_answer = getattr(result, 'short_answer', '')
            analysis = getattr(result, 'analysis', '')
            
            # Determine success based on presence of meaningful output
            success = bool(sql or table_markdown or short_answer or analysis) and not error
            execution_time = end_time - start_time
            
            # Check if mode matches expectation
            mode_correct = expected_mode is None or mode == expected_mode
            
            test_result = {
                "query": query,
                "category": category,
                "expected_mode": expected_mode,
                "actual_mode": mode,
                "mode_correct": mode_correct,
                "sql_generated": bool(sql),
                "success": success,
                "execution_time": execution_time,
                "sql": sql[:200] + "..." if len(sql) > 200 else sql,
                "error": error,
                "has_table": bool(table_markdown),
                "has_analysis": bool(analysis),
                "has_short_answer": bool(short_answer),
                "timestamp": datetime.now().isoformat()
            }
            
            # Print result summary
            status = "✅" if success else "❌"
            mode_status = "✅" if mode_correct else f"❌ (expected {expected_mode})"
            print(f"   {status} Success: {success}")
            print(f"   🎯 Mode: {mode} {mode_status}")
            print(f"   ⏱️  Time: {execution_time:.2f}s")
            if sql:
                print(f"   🔧 SQL: {sql[:100]}...")
            
            return test_result
            
        except Exception as e:
            end_time = time.time()
            test_result = {
                "query": query,
                "category": category,
                "expected_mode": expected_mode,
                "actual_mode": "ERROR",
                "mode_correct": False,
                "sql_generated": False,
                "success": False,
                "execution_time": end_time - start_time,
                "sql": "",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            print(f"   ❌ Error: {e}")
            return test_result
    
    def run_test_suite(self):
        """Run the comprehensive test suite."""
        test_queries = [
            # Advanced Schema Understanding & Multi-hop FK Navigation
            ("Show me employee details along with their department information and current projects", "Advanced Schema", "TABLE"),
            ("Which employees work on projects managed by someone from a different department with higher salaries?", "Advanced Schema", "TABLE"),
            ("List employees whose managers earn less than them and work in the same building", "Advanced Schema", "TABLE"),
            ("Find all employees who share both department and skills with their project manager", "Advanced Schema", "TABLE"),
            
            # Enhanced Pattern Recognition & Deterministic Builders
            ("Show a pie chart of employees working in exactly 3 active projects by department", "Enhanced Patterns", "VISUALIZATION"),
            ("Count employees by department and show their average project involvement", "Enhanced Patterns", "TABLE"),
            ("Compare Engineering vs Sales headcount and budget allocation over the last 3 years", "Enhanced Patterns", "ANALYTICAL"),
            ("Which high-performing employees have managers in the same department earning similar salaries?", "Enhanced Patterns", "TABLE"),
            ("How many senior employees are working on exactly 2 high-budget ongoing projects?", "Enhanced Patterns", "SHORT_ANSWER"),
            
            # Advanced Visualization with Data Pattern Recognition
            ("Create a pie chart of project budgets showing only projects with unique team compositions", "Advanced Visualization", "VISUALIZATION"),
            ("Plot salary distribution across departments highlighting outliers and ranges", "Advanced Visualization", "VISUALIZATION"),
            ("Show a trend graph of hiring over time with department-specific forecasting", "Advanced Visualization", "VISUALIZATION"),
            ("Generate a histogram of project durations grouped by complexity and budget tiers", "Advanced Visualization", "VISUALIZATION"),
            
            # Complex Analytical & Multi-table Insights
            ("Analyze the correlation between employee salary, project budget, and performance ratings across departments", "Complex Analytics", "ANALYTICAL"),
            ("What trends do you see in hiring across departments and how do they correlate with project success rates?", "Complex Analytics", "ANALYTICAL"),
            ("Compare project success rates by department considering team size, budget, and employee experience levels", "Complex Analytics", "ANALYTICAL"),
            ("Identify departments with the best ROI based on employee costs vs project revenues", "Complex Analytics", "ANALYTICAL"),
            
            # Intelligent Clarification Logic
            ("We need more employees", "Smart Clarification", "CLARIFICATION"),
            ("Show me performance data", "Smart Clarification", "CLARIFICATION"),
            ("Compare departments", "Smart Clarification", "CLARIFICATION"),
            ("How are we doing overall?", "Smart Clarification", "CLARIFICATION"),
            ("Analyze productivity metrics", "Smart Clarification", "CLARIFICATION"),
            
            # Multi-hop Complex Join Reasoning
            ("Show employees working on the most expensive projects in each department with their skill levels and manager approval", "Multi-hop Joins", "TABLE"),
            ("Which departments have employees working on projects with budgets over 100k and what are their success rates?", "Multi-hop Joins", "TABLE"),
            ("Find employees who share skills with their project teammates and have worked together on multiple projects", "Multi-hop Joins", "TABLE"),
            ("List project managers whose teams include employees from at least 3 different departments", "Multi-hop Joins", "TABLE"),
            
            # Advanced Time-Series & Cross-temporal Analysis
            ("Show hiring trends for the last 5 years with seasonal patterns and department-specific forecasting", "Advanced Time Series", "ANALYTICAL"),
            ("Analyze seasonal patterns in employee performance correlated with project deadlines and team changes", "Advanced Time Series", "ANALYTICAL"),
            ("Compare year-over-year productivity improvements across departments with budget impact analysis", "Advanced Time Series", "ANALYTICAL"),
            
            # Semantic Understanding & Context Awareness
            ("Find high-potential employees ready for promotion based on performance, skills, and project leadership", "Semantic Intelligence", "ANALYTICAL"),
            ("Identify project bottlenecks by analyzing team composition, skill gaps, and resource allocation", "Semantic Intelligence", "ANALYTICAL"),
            ("Show me employees who might be flight risks based on workload, compensation, and career progression", "Semantic Intelligence", "ANALYTICAL"),
            
            # Advanced Edge Cases & Error Handling
            ("Show me data for employee ID 99999 and suggest similar employees if not found", "Advanced Edge Cases", "ANALYTICAL"),
            ("List projects from the year 2030 or the most recent projects if none exist", "Advanced Edge Cases", "TABLE"),
            ("Compare non-existent departments with actual department performance metrics", "Advanced Edge Cases", "CLARIFICATION"),
            ("Analyze the impact of employees who left the company on current project timelines", "Advanced Edge Cases", "ANALYTICAL"),
        ]
        
        print("🚀 Starting Enhanced Hybrid RAG Test Suite")
        print("🧠 Testing Advanced Graph RAG + Enhanced Schema Inspection")
        print(f"📊 Total advanced queries to test: {len(test_queries)}")
        print("\n🎯 NEW CAPABILITIES BEING TESTED:")
        print("   • Multi-hop FK relationship navigation")
        print("   • Semantic scoring and intelligent join paths")
        print("   • Enhanced schema inspection with data patterns")
        print("   • Context-aware clarification logic")
        print("   • Complex multi-table analytical reasoning")
        print("   • Advanced visualization with pattern recognition")
        print("=" * 60)
        
        for query, category, expected_mode in test_queries:
            result = self.run_query(query, category, expected_mode)
            self.results.append(result)
            time.sleep(0.5)  # Small delay between queries
    
    def generate_report(self):
        """Generate a comprehensive test report."""
        if not self.results:
            print("❌ No test results to report!")
            return
        
        print("\n" + "=" * 60)
        print("📊 ENHANCED HYBRID RAG TEST REPORT")
        print("🧠 Advanced Graph RAG + Enhanced Schema Inspection")
        print("=" * 60)
        
        # Overall statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r["success"])
        mode_correct_tests = sum(1 for r in self.results if r["mode_correct"])
        avg_execution_time = sum(r["execution_time"] for r in self.results) / total_tests
        
        # Advanced metrics
        complex_queries = sum(1 for r in self.results if any(keyword in r["category"].lower() 
                             for keyword in ["multi-hop", "complex", "advanced", "semantic"]))
        complex_success = sum(1 for r in self.results if r["success"] and any(keyword in r["category"].lower() 
                             for keyword in ["multi-hop", "complex", "advanced", "semantic"]))
        
        visualization_queries = sum(1 for r in self.results if "visualization" in r["category"].lower())
        viz_success = sum(1 for r in self.results if r["success"] and "visualization" in r["category"].lower())
        
        print(f"\n📈 ENHANCED INTELLIGENCE METRICS:")
        print(f"   Total Advanced Tests: {total_tests}")
        print(f"   Overall Success Rate: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        print(f"   Mode Detection Accuracy: {mode_correct_tests} ({mode_correct_tests/total_tests*100:.1f}%)")
        print(f"   Average Query Time: {avg_execution_time:.2f}s")
        print(f"   Complex Multi-table Success: {complex_success}/{complex_queries} ({complex_success/complex_queries*100:.1f}% if complex_queries > 0 else 0)")
        print(f"   Advanced Visualization Success: {viz_success}/{visualization_queries} ({viz_success/visualization_queries*100:.1f}% if visualization_queries > 0 else 0)")
        
        # Intelligence assessment
        print(f"\n🧠 INTELLIGENCE ASSESSMENT:")
        if successful_tests/total_tests >= 0.95:
            intelligence_level = "🌟 EXCEPTIONAL"
        elif successful_tests/total_tests >= 0.90:
            intelligence_level = "🚀 EXCELLENT" 
        elif successful_tests/total_tests >= 0.85:
            intelligence_level = "✅ VERY GOOD"
        elif successful_tests/total_tests >= 0.75:
            intelligence_level = "👍 GOOD"
        else:
            intelligence_level = "⚠️  NEEDS IMPROVEMENT"
            
        print(f"   Overall Intelligence Level: {intelligence_level}")
        print(f"   Schema Understanding: {'🟢 Advanced' if any('Advanced Schema' in r['category'] and r['success'] for r in self.results) else '🟡 Basic'}")
        print(f"   Graph RAG Navigation: {'🟢 Multi-hop' if any('Multi-hop' in r['category'] and r['success'] for r in self.results) else '🟡 Single-hop'}")
        print(f"   Semantic Intelligence: {'🟢 Context-aware' if any('Semantic' in r['category'] and r['success'] for r in self.results) else '🟡 Pattern-based'}")
        print(f"   Clarification Logic: {'🟢 Intelligent' if any('Smart Clarification' in r['category'] and r['actual_mode'] == 'CLARIFICATION' for r in self.results) else '🟡 Basic'}")
        
        # Category breakdown
        categories = {}
        for result in self.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "success": 0, "mode_correct": 0}
            categories[cat]["total"] += 1
            if result["success"]:
                categories[cat]["success"] += 1
            if result["mode_correct"]:
                categories[cat]["mode_correct"] += 1
        
        print(f"\n📂 CATEGORY BREAKDOWN:")
        for cat, stats in categories.items():
            success_rate = stats["success"] / stats["total"] * 100
            mode_rate = stats["mode_correct"] / stats["total"] * 100
            print(f"   {cat}:")
            print(f"      Success: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
            print(f"      Mode Accuracy: {stats['mode_correct']}/{stats['total']} ({mode_rate:.1f}%)")
        
        # Failed tests
        failed_tests = [r for r in self.results if not r["success"]]
        if failed_tests:
            print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   • {test['query']}")
                print(f"     Error: {test['error']}")
        
        # Mode detection issues
        mode_issues = [r for r in self.results if not r["mode_correct"] and r["expected_mode"]]
        if mode_issues:
            print(f"\n🎯 MODE DETECTION ISSUES ({len(mode_issues)}):")
            for test in mode_issues:
                print(f"   • {test['query']}")
                print(f"     Expected: {test['expected_mode']}, Got: {test['actual_mode']}")
        
        # Save detailed results to file
        report_file = f"outputs/hybrid_rag_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("outputs", exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "mode_correct_tests": mode_correct_tests,
                    "success_rate": successful_tests/total_tests*100,
                    "mode_accuracy": mode_correct_tests/total_tests*100,
                    "avg_execution_time": avg_execution_time
                },
                "category_stats": categories,
                "detailed_results": self.results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed report saved to: {report_file}")

def main():
    """Main test execution."""
    db_url = "mysql+pymysql://root:@localhost:3306/test_01"
    
    print("🧪 Hybrid Graph RAG + Vector RAG Comprehensive Test")
    print("=" * 60)
    
    tester = HybridRAGTester(db_url)
    
    if not tester.setup():
        print("❌ Failed to setup test environment!")
        return 1
    
    try:
        tester.run_test_suite()
        tester.generate_report()
        print("\n🎉 Test suite completed successfully!")
        return 0
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        tester.generate_report()
        return 1
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
