#!/usr/bin/env python3
"""
Query Classifier for Hybrid NLP-to-SQL System
Intelligent query classification using LLM-based semantic understanding
"""

import json
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from langchain_ollama import ChatOllama

LOGGER = logging.getLogger(__name__)


@dataclass
class QueryClassification:
    """Structured classification result"""
    complexity: str  # simple/complex
    type: str        # scalar/table/visualization/analytical/complex
    confidence: float  # 0.0-1.0
    reasoning: str
    handler_type: str  # simple/visualization/analytical/complex
    fallback_reason: Optional[str] = None


class QueryClassifier:
    """Intelligent query classifier using LLM-based semantic understanding"""
    
    def __init__(self, fast_model: str = "llama3.2:1b"):
        self.fast_llm = ChatOllama(model=fast_model, temperature=0.0)
        self.classification_cache = {}
        self.classification_examples = self._load_classification_examples()
    
    def _load_classification_examples(self) -> Dict[str, Any]:
        """Load classification examples for better accuracy"""
        return {
            "simple_scalar": [
                "How many workers are there?",
                "What is the total production volume for today?",
                "Show me the count of hygiene violations",
                "What's the average production weight?"
            ],
            "simple_table": [
                "Show me all workers",
                "List production batches from last week",
                "Display hygiene check results",
                "Get all packaging types"
            ],
            "visualization": [
                "Show production volumes as a bar chart",
                "Create a pie chart of waste types",
                "Display hygiene compliance rates with a line chart",
                "Visualize packaging distribution"
            ],
            "analytical": [
                "Analyze production trends over the last 6 months",
                "Compare hygiene compliance rates between sections",
                "What are the insights from waste generation patterns?",
                "How has production efficiency changed over time?"
            ],
            "complex": [
                "Show me production volumes for today and create a bar chart of the results",
                "Analyze hygiene compliance trends and identify the top 3 violators",
                "Compare production efficiency between different bake types and show the results as a chart"
            ]
        }
    
    def classify(self, query: str, schema_context: str = "") -> QueryClassification:
        """Classify query with semantic understanding"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self.classification_cache:
            LOGGER.info(f"Classification cache hit for query: {query[:50]}...")
            return self.classification_cache[cache_key]
        
        try:
            # Build classification prompt
            prompt = self._build_classification_prompt(query, schema_context)
            
            # Get LLM classification
            response = self.fast_llm.invoke(prompt)
            raw_classification = self._parse_llm_response(response)
            
            # Validate and enhance classification
            classification = self._validate_and_enhance(raw_classification, query)
            
            # Cache result
            self.classification_cache[cache_key] = classification
            
            classification_time = time.time() - start_time
            LOGGER.info(f"Query classified in {classification_time:.2f}s: {classification.type} ({classification.confidence:.2f})")
            
            return classification
            
        except Exception as e:
            LOGGER.error(f"Classification failed: {e}")
            # Fallback classification
            return self._fallback_classification(query)
    
    def _build_classification_prompt(self, query: str, schema_context: str) -> str:
        """Build optimized classification prompt"""
        examples_text = self._format_examples()
        
        return f"""You are an expert query classifier for an NLP-to-SQL system.

TASK: Classify this query into the most appropriate category and complexity level.

QUERY: "{query}"

SCHEMA CONTEXT:
{schema_context[:500] if schema_context else "Food production database with tables: production_info, person_hyg, packaging_info, pack_waste, workers, prices"}

CLASSIFICATION CATEGORIES:
1. SIMPLE SCALAR: Single value requests (counts, sums, averages, totals)
   Examples: {examples_text['simple_scalar'][:2]}
   
2. SIMPLE TABLE: Multi-row data requests (lists, displays, shows)
   Examples: {examples_text['simple_table'][:2]}
   
3. VISUALIZATION: Chart/graph requests (bar charts, pie charts, plots)
   Examples: {examples_text['visualization'][:2]}
   
4. ANALYTICAL: Analysis/insights requests (trends, comparisons, patterns)
   Examples: {examples_text['analytical'][:2]}
   
5. COMPLEX: Multi-step or multi-requirement requests
   Examples: {examples_text['complex'][:2]}

COMPLEXITY LEVELS:
- SIMPLE: Straightforward single-step queries
- COMPLEX: Multi-step, multi-table, or complex reasoning queries

CLASSIFICATION RULES:
- If query asks for a chart/visualization → type=visualization
- If query asks for analysis/insights → type=analytical  
- If query asks for a single number → type=scalar
- If query asks for a list/table → type=table
- If query has multiple requirements → type=complex
- If uncertain, choose complex for safety

OUTPUT FORMAT (JSON only):
{{
    "complexity": "simple|complex",
    "type": "scalar|table|visualization|analytical|complex",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of classification decision",
    "handler_type": "simple|visualization|analytical|complex"
}}

CLASSIFY NOW:"""
    
    def _format_examples(self) -> Dict[str, list]:
        """Format examples for prompt"""
        return {
            "simple_scalar": ["How many workers?", "Total production today?"],
            "simple_table": ["Show all workers", "List production batches"],
            "visualization": ["Show as bar chart", "Create pie chart"],
            "analytical": ["Analyze trends", "Compare performance"],
            "complex": ["Show data and create chart", "Analyze and identify top performers"]
        }
    
    def _parse_llm_response(self, response) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        try:
            # Extract content from response
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to extract JSON from response
            if '```json' in content:
                json_start = content.find('```json') + 7
                json_end = content.find('```', json_start)
                json_str = content[json_start:json_end].strip()
            elif '{' in content and '}' in content:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                json_str = content[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")
            
            return json.loads(json_str)
            
        except Exception as e:
            LOGGER.error(f"Failed to parse LLM response: {e}")
            raise
    
    def _validate_and_enhance(self, raw_classification: Dict[str, Any], query: str) -> QueryClassification:
        """Validate and enhance classification result"""
        # Validate required fields
        complexity = raw_classification.get('complexity', 'complex')
        type_val = raw_classification.get('type', 'complex')
        confidence = float(raw_classification.get('confidence', 0.5))
        reasoning = raw_classification.get('reasoning', 'No reasoning provided')
        
        # Determine handler type
        handler_type = self._determine_handler_type(type_val, complexity, confidence)
        
        # Apply confidence-based routing
        if confidence < 0.5:
            LOGGER.warning(f"Low confidence classification ({confidence:.2f}), routing to complex handler")
            handler_type = "complex"
            fallback_reason = "low_confidence"
        else:
            fallback_reason = None
        
        return QueryClassification(
            complexity=complexity,
            type=type_val,
            confidence=confidence,
            reasoning=reasoning,
            handler_type=handler_type,
            fallback_reason=fallback_reason
        )
    
    def _determine_handler_type(self, type_val: str, complexity: str, confidence: float) -> str:
        """Determine appropriate handler type"""
        if type_val == "visualization":
            return "visualization"
        elif type_val == "analytical":
            return "analytical"
        elif type_val == "complex":
            return "complex"
        elif type_val in ["scalar", "table"] and complexity == "simple":
            return "simple"
        else:
            return "complex"  # Default to complex for safety
    
    def _fallback_classification(self, query: str) -> QueryClassification:
        """Fallback classification when LLM fails"""
        LOGGER.warning("Using fallback classification")
        
        # Simple heuristics as fallback
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['chart', 'plot', 'graph', 'visualize']):
            return QueryClassification(
                complexity="complex",
                type="visualization", 
                confidence=0.3,
                reasoning="Fallback: detected visualization keywords",
                handler_type="visualization",
                fallback_reason="llm_classification_failed"
            )
        elif any(word in query_lower for word in ['analyze', 'analysis', 'trends', 'compare']):
            return QueryClassification(
                complexity="complex",
                type="analytical",
                confidence=0.3,
                reasoning="Fallback: detected analytical keywords",
                handler_type="analytical",
                fallback_reason="llm_classification_failed"
            )
        else:
            return QueryClassification(
                complexity="complex",
                type="complex",
                confidence=0.2,
                reasoning="Fallback: defaulting to complex",
                handler_type="complex",
                fallback_reason="llm_classification_failed"
            )
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        import hashlib
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def clear_cache(self):
        """Clear classification cache"""
        self.classification_cache.clear()
        LOGGER.info("Classification cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.classification_cache),
            "cache_keys": list(self.classification_cache.keys())[:5]  # First 5 keys
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the classifier
    classifier = QueryClassifier()
    
    test_queries = [
        "How many workers are there?",
        "Show me production volumes as a bar chart", 
        "Analyze hygiene compliance trends",
        "List all production batches from last week",
        "Show me data and create a chart"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        classification = classifier.classify(query)
        print(f"Classification: {classification.type} ({classification.complexity})")
        print(f"Handler: {classification.handler_type}")
        print(f"Confidence: {classification.confidence:.2f}")
        print(f"Reasoning: {classification.reasoning}")
