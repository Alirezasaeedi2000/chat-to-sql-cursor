#!/usr/bin/env python3
"""
Simple Query Handler for Hybrid NLP-to-SQL System
Handles scalar and simple table queries using deterministic builders with LLM fallback
"""

import logging
import time
from typing import Dict, Any, Optional
import pandas as pd

from .base_handler import BaseHandler
from query_classifier import QueryClassification

LOGGER = logging.getLogger(__name__)


class SimpleQueryHandler(BaseHandler):
    """Handler for simple scalar and table queries"""
    
    def __init__(self, engine, vector_manager, llm, config: Optional[Dict[str, Any]] = None):
        super().__init__(engine, vector_manager, llm, config)
        self.deterministic_builders = DeterministicBuilderManager()
        self.max_execution_time = self.config.get('max_execution_time', 2.0)
        self.use_deterministic_builders = self.config.get('use_deterministic_builders', True)
        self.cache_results = self.config.get('cache_results', True)
        
        # Cache for deterministic results
        self.deterministic_cache = {}
    
    def can_handle(self, query: str, classification: QueryClassification) -> bool:
        """Check if this handler can process the query"""
        # Handle simple scalar and table queries
        return (classification.type in ['scalar', 'table'] and 
                classification.complexity == 'simple' and
                classification.confidence >= 0.5)
    
    def process(self, query: str, classification: QueryClassification) -> Dict[str, Any]:
        """Process simple query using deterministic builders with LLM fallback"""
        start_time = time.time()
        
        try:
            LOGGER.info(f"Processing simple query: {query[:50]}...")
            
            # Step 1: Try deterministic builders first
            if self.use_deterministic_builders:
                sql = self._try_deterministic_builders(query)
                if sql:
                    LOGGER.info("Using deterministic builder")
                    result = self._execute_deterministic_sql(sql, query, classification)
                    
                    execution_time = time.time() - start_time
                    self.update_stats(execution_time, True)
                    
                    return result
            
            # Step 2: Fallback to LLM generation
            LOGGER.info("Deterministic builders failed, using LLM fallback")
            result = self._fallback_to_llm(query, classification)
            
            execution_time = time.time() - start_time
            self.update_stats(execution_time, True)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.update_stats(execution_time, False)
            return self.handle_error(query, classification, e)
    
    def _try_deterministic_builders(self, query: str) -> Optional[str]:
        """Try deterministic builders for simple queries"""
        try:
            # Check cache first
            cache_key = self._get_cache_key(query)
            if cache_key in self.deterministic_cache:
                LOGGER.info("Using cached deterministic result")
                return self.deterministic_cache[cache_key]
            
            # Try different deterministic builders
            sql = self.deterministic_builders.build_sql(query)
            
            if sql:
                # Cache the result
                self.deterministic_cache[cache_key] = sql
                return sql
            
            return None
            
        except Exception as e:
            LOGGER.warning(f"Deterministic builders failed: {e}")
            return None
    
    def _execute_deterministic_sql(self, sql: str, query: str, classification: QueryClassification) -> Dict[str, Any]:
        """Execute SQL from deterministic builders"""
        try:
            # Validate SQL
            validation = self.validate_sql(sql)
            if not validation['is_valid']:
                raise ValueError(f"SQL validation failed: {validation['errors']}")
            
            # Execute SQL
            data = self.execute_sql(sql)
            
            # Determine result type
            query_type = 'scalar' if classification.type == 'scalar' else 'table'
            
            # Format result
            result = self.format_result(data, query_type, sql, 'simple', deterministic=True)
            
            # Add classification metadata
            result['metadata']['classification'] = {
                'complexity': classification.complexity,
                'type': classification.type,
                'confidence': classification.confidence
            }
            
            return result
            
        except Exception as e:
            LOGGER.error(f"Deterministic SQL execution failed: {e}")
            raise
    
    def _fallback_to_llm(self, query: str, classification: QueryClassification) -> Dict[str, Any]:
        """Fallback to LLM generation"""
        try:
            # Get schema context
            schema_context = self.get_schema_context()
            
            # Build LLM prompt for simple queries
            prompt = self._build_simple_llm_prompt(query, classification, schema_context)
            
            # Generate SQL with LLM
            response = self.llm.invoke(prompt)
            sql = self._extract_sql_from_response(response)
            
            if not sql:
                raise ValueError("No SQL generated by LLM")
            
            # Execute generated SQL
            data = self.execute_sql(sql)
            
            # Determine result type
            query_type = 'scalar' if classification.type == 'scalar' else 'table'
            
            # Format result
            result = self.format_result(data, query_type, sql, 'simple', deterministic=False)
            
            # Add classification metadata
            result['metadata']['classification'] = {
                'complexity': classification.complexity,
                'type': classification.type,
                'confidence': classification.confidence
            }
            
            # Add LLM metadata
            result['metadata']['llm_fallback'] = True
            
            return result
            
        except Exception as e:
            LOGGER.error(f"LLM fallback failed: {e}")
            raise
    
    def _build_simple_llm_prompt(self, query: str, classification: QueryClassification, schema_context: str) -> str:
        """Build optimized LLM prompt for simple queries"""
        return f"""You are an expert SQL generator for simple queries.

QUERY: "{query}"
QUERY TYPE: {classification.type.upper()}
COMPLEXITY: {classification.complexity.upper()}

SCHEMA CONTEXT:
{schema_context}

TASK: Generate a simple, efficient SQL query for this request.

RULES FOR SIMPLE QUERIES:
- Use only basic SELECT statements
- Avoid complex JOINs or subqueries
- Always include LIMIT 50
- Use backticks around table/column names
- For scalar queries: return single value (COUNT, SUM, AVG, etc.)
- For table queries: return multiple rows

EXAMPLES:
- "How many workers?" → SELECT COUNT(*) FROM `workers` LIMIT 50
- "Show all workers" → SELECT * FROM `workers` LIMIT 50
- "Total production today" → SELECT SUM(`totalUsage`) FROM `production_info` WHERE `date` = CURDATE() LIMIT 50

Generate SQL for the query above. Return ONLY the SQL statement wrapped in ```sql``` fences."""

    def _extract_sql_from_response(self, response) -> Optional[str]:
        """Extract SQL from LLM response"""
        try:
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Look for SQL in code fences
            if '```sql' in content:
                start = content.find('```sql') + 6
                end = content.find('```', start)
                sql = content[start:end].strip()
            elif '```' in content:
                start = content.find('```') + 3
                end = content.find('```', start)
                sql = content[start:end].strip()
            else:
                # Look for SELECT statement
                import re
                select_match = re.search(r'SELECT.*?(?=\n\n|\n$|$)', content, re.IGNORECASE | re.DOTALL)
                if select_match:
                    sql = select_match.group(0).strip()
                else:
                    return None
            
            return sql if sql else None
            
        except Exception as e:
            LOGGER.error(f"Failed to extract SQL from response: {e}")
            return None
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        import hashlib
        return hashlib.md5(f"simple_{query.lower().strip()}".encode()).hexdigest()
    
    def clear_cache(self):
        """Clear deterministic cache"""
        self.deterministic_cache.clear()
        LOGGER.info("Simple handler cache cleared")


class DeterministicBuilderManager:
    """Manages deterministic SQL builders for simple queries"""
    
    def __init__(self):
        self.builders = self._initialize_builders()
    
    def _initialize_builders(self) -> Dict[str, callable]:
        """Initialize deterministic builders"""
        return {
            'count_queries': self._build_count_queries,
            'scalar_aggregations': self._build_scalar_aggregations,
            'simple_tables': self._build_simple_tables,
            'date_filtered': self._build_date_filtered_queries
        }
    
    def build_sql(self, query: str) -> Optional[str]:
        """Try to build SQL using deterministic builders"""
        query_lower = query.lower().strip()
        
        # Try each builder in order
        for builder_name, builder_func in self.builders.items():
            try:
                sql = builder_func(query_lower)
                if sql:
                    LOGGER.info(f"Deterministic builder '{builder_name}' generated SQL")
                    return sql
            except Exception as e:
                LOGGER.warning(f"Builder '{builder_name}' failed: {e}")
                continue
        
        return None
    
    def _build_count_queries(self, query_lower: str) -> Optional[str]:
        """Build SQL for count queries"""
        count_patterns = [
            ('how many workers', 'SELECT COUNT(*) FROM `workers` LIMIT 50'),
            ('how many employees', 'SELECT COUNT(*) FROM `workers` LIMIT 50'),
            ('number of workers', 'SELECT COUNT(*) FROM `workers` LIMIT 50'),
            ('count of workers', 'SELECT COUNT(*) FROM `workers` LIMIT 50'),
            ('total workers', 'SELECT COUNT(*) FROM `workers` LIMIT 50'),
            ('how many production', 'SELECT COUNT(*) FROM `production_info` LIMIT 50'),
            ('how many batches', 'SELECT COUNT(*) FROM `production_info` LIMIT 50'),
            ('how many hygiene', 'SELECT COUNT(*) FROM `person_hyg` LIMIT 50'),
            ('how many waste', 'SELECT COUNT(*) FROM `pack_waste` LIMIT 50')
        ]
        
        for pattern, sql in count_patterns:
            if pattern in query_lower:
                return sql
        
        return None
    
    def _build_scalar_aggregations(self, query_lower: str) -> Optional[str]:
        """Build SQL for scalar aggregation queries"""
        # Production volume queries
        if 'production volume' in query_lower and 'today' in query_lower:
            return "SELECT SUM(`totalUsage`) FROM `production_info` WHERE `date` = CURDATE() LIMIT 50"
        elif 'production volume' in query_lower and 'this month' in query_lower:
            return "SELECT SUM(`totalUsage`) FROM `production_info` WHERE MONTH(`date`) = MONTH(CURDATE()) LIMIT 50"
        elif 'total production' in query_lower and 'today' in query_lower:
            return "SELECT SUM(`totalUsage`) FROM `production_info` WHERE `date` = CURDATE() LIMIT 50"
        
        # Waste queries
        elif 'total waste' in query_lower and 'today' in query_lower:
            return "SELECT SUM(`value`) FROM `pack_waste` WHERE DATE(`date`) = CURDATE() LIMIT 50"
        elif 'total waste' in query_lower and 'this week' in query_lower:
            return "SELECT SUM(`value`) FROM `pack_waste` WHERE `date` >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) LIMIT 50"
        
        # Average queries
        elif 'average production' in query_lower:
            return "SELECT AVG(`totalUsage`) FROM `production_info` LIMIT 50"
        elif 'average waste' in query_lower:
            return "SELECT AVG(`value`) FROM `pack_waste` LIMIT 50"
        
        return None
    
    def _build_simple_tables(self, query_lower: str) -> Optional[str]:
        """Build SQL for simple table queries"""
        # Worker queries
        if 'show workers' in query_lower or 'list workers' in query_lower or 'all workers' in query_lower:
            return "SELECT * FROM `workers` LIMIT 50"
        elif 'show employees' in query_lower or 'list employees' in query_lower:
            return "SELECT * FROM `workers` LIMIT 50"
        
        # Production queries
        elif 'show production' in query_lower and 'batch' in query_lower:
            return "SELECT * FROM `production_info` ORDER BY `date` DESC LIMIT 50"
        elif 'show batches' in query_lower or 'list batches' in query_lower:
            return "SELECT * FROM `production_info` ORDER BY `date` DESC LIMIT 50"
        
        # Hygiene queries
        elif 'show hygiene' in query_lower or 'hygiene check' in query_lower:
            return "SELECT * FROM `person_hyg` ORDER BY `date` DESC LIMIT 50"
        
        # Waste queries
        elif 'show waste' in query_lower or 'waste data' in query_lower:
            return "SELECT * FROM `pack_waste` ORDER BY `date` DESC LIMIT 50"
        
        return None
    
    def _build_date_filtered_queries(self, query_lower: str) -> Optional[str]:
        """Build SQL for date-filtered queries"""
        # Today queries
        if 'today' in query_lower:
            if 'production' in query_lower:
                return "SELECT * FROM `production_info` WHERE DATE(`date`) = CURDATE() LIMIT 50"
            elif 'hygiene' in query_lower:
                return "SELECT * FROM `person_hyg` WHERE DATE(`date`) = CURDATE() LIMIT 50"
            elif 'waste' in query_lower:
                return "SELECT * FROM `pack_waste` WHERE DATE(`date`) = CURDATE() LIMIT 50"
        
        # This week queries
        elif 'this week' in query_lower:
            if 'production' in query_lower:
                return "SELECT * FROM `production_info` WHERE `date` >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) LIMIT 50"
            elif 'waste' in query_lower:
                return "SELECT * FROM `pack_waste` WHERE `date` >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) LIMIT 50"
        
        # This month queries
        elif 'this month' in query_lower:
            if 'production' in query_lower:
                return "SELECT * FROM `production_info` WHERE MONTH(`date`) = MONTH(CURDATE()) LIMIT 50"
            elif 'waste' in query_lower:
                return "SELECT * FROM `pack_waste` WHERE MONTH(`date`) = MONTH(CURDATE()) LIMIT 50"
        
        return None


# Example usage and testing
if __name__ == "__main__":
    # Test the simple handler
    print("Simple Query Handler Test")
    print("=" * 40)
    
    # This would be used with actual engine, vector_manager, and llm
    # handler = SimpleQueryHandler(engine, vector_manager, llm)
    # result = handler.process("How many workers are there?", classification)
    print("Handler implementation complete")
