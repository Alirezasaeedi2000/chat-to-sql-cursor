#!/usr/bin/env python3
"""
Test the improvements made to mode detection and table selection
"""

from app_farnan import FarnanNL2SQLApp

def test_improvements():
    """Test specific improvements made"""
    
    print("🧪 TESTING IMPROVEMENTS")
    print("=" * 50)
    
    # Test queries that were failing before
    test_cases = [
        {
            'query': 'What is the average humidity during production?',
            'expected_mode': 'SHORT_ANSWER',
            'description': 'Should detect SHORT_ANSWER for simple aggregation'
        },
        {
            'query': 'List all unique bake types used',
            'expected_mode': 'TABLE', 
            'description': 'Should detect TABLE for listing query'
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
            
            print(f"📊 Score: {score}/10")
            print(f"🔧 SQL: {sql[:100]}...")
            
            results.append({
                'query': test_case['query'],
                'mode': mode,
                'expected_mode': test_case['expected_mode'],
                'confidence': confidence,
                'score': score,
                'mode_correct': mode_correct,
                'confidence_good': confidence_good,
                'table_correct': table_correct
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
                'table_correct': False
            })
    
    # Summary
    print(f"\n📊 IMPROVEMENT TEST SUMMARY")
    print("=" * 50)
    
    total_tests = len(results)
    mode_correct = sum(1 for r in results if r['mode_correct'])
    confidence_good = sum(1 for r in results if r['confidence_good'])
    table_correct = sum(1 for r in results if r['table_correct'])
    avg_score = sum(r['score'] for r in results) / total_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Mode Detection: {mode_correct}/{total_tests} ({mode_correct/total_tests:.1%})")
    print(f"Confidence Scores: {confidence_good}/{total_tests} ({confidence_good/total_tests:.1%})")
    print(f"Table Selection: {table_correct}/{total_tests} ({table_correct/total_tests:.1%})")
    print(f"Average Score: {avg_score:.1f}/10")
    
    # Overall improvement
    overall_accuracy = (mode_correct + confidence_good + table_correct) / (total_tests * 3)
    print(f"Overall Accuracy: {overall_accuracy:.1%}")
    
    if overall_accuracy >= 0.8:
        print("🎉 EXCELLENT: Improvements are working well!")
    elif overall_accuracy >= 0.6:
        print("✅ GOOD: Improvements are working, but can be better")
    else:
        print("⚠️ NEEDS WORK: Improvements need more refinement")
    
    return results

if __name__ == "__main__":
    test_improvements()
