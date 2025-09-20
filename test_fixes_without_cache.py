#!/usr/bin/env python3
"""
Test the fixes with caching disabled to see true accuracy
"""

from app_farnan import FarnanNL2SQLApp

def test_fixes_without_cache():
    """Test improvements with caching disabled"""
    
    print("🧪 TESTING FIXES WITHOUT CACHE")
    print("=" * 50)
    
    # Test queries that were failing before
    test_cases = [
        {
            'query': 'List all unique bake types used',
            'expected_mode': 'TABLE',
            'description': 'Should detect TABLE for listing query with enhanced patterns'
        },
        {
            'query': 'What is the average humidity during production?',
            'expected_mode': 'SHORT_ANSWER',
            'description': 'Should detect SHORT_ANSWER for simple aggregation'
        },
        {
            'query': 'Show me recent batches',
            'expected_mode': 'TABLE',
            'description': 'Should detect TABLE for recent data query'
        },
        {
            'query': 'How many workers are there?',
            'expected_mode': 'SHORT_ANSWER',
            'description': 'Should detect SHORT_ANSWER for count query'
        },
        {
            'query': 'Analyze price trends for cream over the last 6 months',
            'expected_mode': 'ANALYTICAL',
            'description': 'Should use prices table for price queries'
        }
    ]
    
    app = FarnanNL2SQLApp()
    
    # Disable caching to test true accuracy
    app.query_processor.safe_exec.cache.disable()
    print("✅ Caching disabled for testing")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 TEST {i}: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        
        try:
            result = app.process_query(test_case['query'])
            
            mode = result.get('mode', 'UNKNOWN')
            confidence = result.get('metadata', {}).get('confidence', 0.0)
            sql = result.get('sql', 'No SQL')
            
            # Check mode detection
            mode_correct = mode == test_case['expected_mode']
            confidence_good = confidence >= 0.8
            
            # Check table selection for price queries
            table_correct = True
            if 'price' in test_case['query'].lower():
                table_correct = 'prices' in sql.lower() and 'production_info' not in sql.lower()
            
            # Check if using correct deterministic builders
            deterministic_used = False
            if 'list all unique bake types' in test_case['query'].lower():
                deterministic_used = 'packaging_info' in sql and 'distinct' in sql.lower()
            elif 'how many workers' in test_case['query'].lower():
                deterministic_used = 'count(*)' in sql.lower() and 'workers' in sql.lower()
            elif 'average humidity' in test_case['query'].lower():
                deterministic_used = 'avg(humidity)' in sql.lower() and 'production_info' in sql
            
            score = 0
            if mode_correct:
                score += 4
                print(f"✅ Mode: {mode} (Expected: {test_case['expected_mode']})")
            else:
                print(f"❌ Mode: {mode} (Expected: {test_case['expected_mode']})")
            
            if confidence_good:
                score += 3
                print(f"✅ Confidence: {confidence}")
            else:
                print(f"❌ Confidence: {confidence}")
            
            if table_correct:
                score += 3
                print(f"✅ Table Selection: Correct")
            else:
                print(f"❌ Table Selection: Incorrect")
            
            if deterministic_used:
                score += 2
                print(f"✅ Deterministic Builder: Used")
            else:
                print(f"⚠️ Deterministic Builder: Not used")
            
            print(f"📊 Score: {score}/12")
            print(f"🔧 SQL: {sql[:100]}...")
            
            results.append({
                'query': test_case['query'],
                'mode': mode,
                'expected_mode': test_case['expected_mode'],
                'confidence': confidence,
                'score': score,
                'mode_correct': mode_correct,
                'confidence_good': confidence_good,
                'table_correct': table_correct,
                'deterministic_used': deterministic_used
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'query': test_case['query'],
                'mode': 'ERROR',
                'expected_mode': test_case['expected_mode'],
                'confidence': 0.0,
                'score': 0,
                'mode_correct': False,
                'confidence_good': False,
                'table_correct': False,
                'deterministic_used': False
            })
    
    # Summary
    print(f"\n📊 FIXES TEST SUMMARY (NO CACHE)")
    print("=" * 50)
    
    total_tests = len(results)
    mode_correct = sum(1 for r in results if r['mode_correct'])
    confidence_good = sum(1 for r in results if r['confidence_good'])
    table_correct = sum(1 for r in results if r['table_correct'])
    deterministic_used = sum(1 for r in results if r['deterministic_used'])
    avg_score = sum(r['score'] for r in results) / total_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Mode Detection: {mode_correct}/{total_tests} ({mode_correct/total_tests:.1%})")
    print(f"Confidence Scores: {confidence_good}/{total_tests} ({confidence_good/total_tests:.1%})")
    print(f"Table Selection: {table_correct}/{total_tests} ({table_correct/total_tests:.1%})")
    print(f"Deterministic Builders: {deterministic_used}/{total_tests} ({deterministic_used/total_tests:.1%})")
    print(f"Average Score: {avg_score:.1f}/12")
    
    # Overall improvement
    overall_accuracy = (mode_correct + confidence_good + table_correct + deterministic_used) / (total_tests * 4)
    print(f"Overall Accuracy: {overall_accuracy:.1%}")
    
    if overall_accuracy >= 0.85:
        print("🎉 EXCELLENT: All fixes are working perfectly!")
    elif overall_accuracy >= 0.75:
        print("✅ VERY GOOD: Most fixes are working well!")
    elif overall_accuracy >= 0.65:
        print("✅ GOOD: Improvements are working, but can be better")
    else:
        print("⚠️ NEEDS WORK: More improvements needed")
    
    # Re-enable caching
    app.query_processor.safe_exec.cache.enable()
    print("\n✅ Caching re-enabled")
    
    return results

if __name__ == "__main__":
    test_fixes_without_cache()
