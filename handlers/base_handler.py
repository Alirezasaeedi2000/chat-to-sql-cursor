#!/usr/bin/env python3
"""
Base Handler for Hybrid NLP-to-SQL System
Abstract base class for all query handlers
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import text

from vector import VectorStoreManager, RetrievedContext
from query_classifier import QueryClassification

LOGGER = logging.getLogger(__name__)


class BaseHandler(ABC):
    """Abstract base class for all query handlers"""
    
    def __init__(self, engine: Engine, vector_manager: VectorStoreManager, llm, config: Optional[Dict[str, Any]] = None):
        self.engine = engine
        self.vector_manager = vector_manager
        self.llm = llm
        self.config = config or {}
        self.fallback_handler: Optional['BaseHandler'] = None
        self.execution_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'average_execution_time': 0.0
        }
    
    @abstractmethod
    def can_handle(self, query: str, classification: QueryClassification) -> bool:
        """Check if this handler can process the query"""
        pass
    
    @abstractmethod
    def process(self, query: str, classification: QueryClassification) -> Dict[str, Any]:
        """Process the query and return results"""
        pass
    
    def set_fallback_handler(self, handler: 'BaseHandler'):
        """Set fallback handler for error cases"""
        self.fallback_handler = handler
    
    def execute_sql(self, sql: str, timeout: int = 30) -> pd.DataFrame:
        """Execute SQL with error handling and timeout"""
        start_time = time.time()
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                execution_time = time.time() - start_time
                LOGGER.info(f"SQL executed successfully in {execution_time:.2f}s")
                
                return df
                
        except Exception as e:
            execution_time = time.time() - start_time
            LOGGER.error(f"SQL execution failed after {execution_time:.2f}s: {e}")
            raise
    
    def validate_sql(self, sql: str) -> Dict[str, Any]:
        """Basic SQL validation"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check for dangerous operations
            dangerous_patterns = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE']
            sql_upper = sql.upper()
            
            for pattern in dangerous_patterns:
                if pattern in sql_upper:
                    validation_result['is_valid'] = False
                    validation_result['errors'].append(f"Dangerous operation detected: {pattern}")
            
            # Check for LIMIT clause
            if 'SELECT' in sql_upper and 'LIMIT' not in sql_upper:
                validation_result['warnings'].append("Query missing LIMIT clause")
            
            # Check for SELECT * 
            if 'SELECT *' in sql_upper:
                validation_result['warnings'].append("SELECT * detected - consider specifying columns")
            
            return validation_result
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation error: {e}")
            return validation_result
    
    def format_result(self, data: pd.DataFrame, query_type: str, sql: str, 
                     handler_name: str, deterministic: bool = False) -> Dict[str, Any]:
        """Format result in standardized format"""
        result = {
            'sql': sql,
            'metadata': {
                'handler_used': handler_name,
                'deterministic': deterministic,
                'row_count': len(data),
                'execution_time_ms': 0,  # Will be set by caller
                'errors': None
            }
        }
        
        if query_type == 'scalar' and len(data) == 1 and len(data.columns) == 1:
            # Scalar result
            result['mode'] = 'SHORT_ANSWER'
            result['short_answer'] = str(data.iloc[0, 0])
        else:
            # Table result
            result['mode'] = 'TABLE'
            result['table_markdown'] = data.to_markdown(index=False) if not data.empty else "(no rows)"
        
        return result
    
    def handle_error(self, query: str, classification: QueryClassification, error: Exception) -> Dict[str, Any]:
        """Handle errors with fallback mechanisms"""
        LOGGER.error(f"Handler {self.__class__.__name__} failed: {error}")
        
        # Update stats
        self.execution_stats['failed_queries'] += 1
        
        # Try fallback handler if available
        if self.fallback_handler:
            LOGGER.info(f"Attempting fallback to {self.fallback_handler.__class__.__name__}")
            try:
                return self.fallback_handler.process(query, classification)
            except Exception as fallback_error:
                LOGGER.error(f"Fallback handler also failed: {fallback_error}")
        
        # Return error result
        return {
            'mode': 'ERROR',
            'error': str(error),
            'sql': None,
            'metadata': {
                'handler_used': self.__class__.__name__,
                'error_type': type(error).__name__,
                'fallback_attempted': self.fallback_handler is not None,
                'errors': [str(error)]
            }
        }
    
    def update_stats(self, execution_time: float, success: bool):
        """Update handler execution statistics"""
        self.execution_stats['total_queries'] += 1
        
        if success:
            self.execution_stats['successful_queries'] += 1
        else:
            self.execution_stats['failed_queries'] += 1
        
        # Update average execution time
        total_time = self.execution_stats['average_execution_time'] * (self.execution_stats['total_queries'] - 1)
        self.execution_stats['average_execution_time'] = (total_time + execution_time) / self.execution_stats['total_queries']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler execution statistics"""
        stats = self.execution_stats.copy()
        if stats['total_queries'] > 0:
            stats['success_rate'] = stats['successful_queries'] / stats['total_queries']
        else:
            stats['success_rate'] = 0.0
        return stats
    
    def get_schema_context(self) -> str:
        """Get relevant schema context for the query"""
        try:
            context = self.vector_manager.similarity_search("schema tables columns", top_k=3)
            return "\n\n".join(context.texts[:2]) if context.texts else "Schema context not available"
        except Exception as e:
            LOGGER.warning(f"Failed to get schema context: {e}")
            return "Schema context not available"
    
    def extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL query"""
        import re
        # Simple regex to find table names after FROM and JOIN
        table_pattern = r'(?:FROM|JOIN)\s+([`\w]+)'
        matches = re.findall(table_pattern, sql, re.IGNORECASE)
        return [match.strip('`') for match in matches]
