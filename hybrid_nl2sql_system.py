#!/usr/bin/env python3
"""
Hybrid NLP-to-SQL System
Main orchestrator for the hybrid agent-based routing architecture
"""

import logging
import time
from typing import Dict, Any, Optional, List
from sqlalchemy.engine import Engine

from query_classifier import QueryClassifier, QueryClassification
from vector import VectorStoreManager
from handlers import (
    SimpleQueryHandler, 
    VisualizationHandler, 
    AnalyticalHandler, 
    ComplexQueryHandler
)

LOGGER = logging.getLogger(__name__)


class HybridNL2SQLSystem:
    """Main hybrid system orchestrating query classification and handler routing"""
    
    def __init__(self, engine: Engine, vector_manager: VectorStoreManager, 
                 model_name: str = "llama3.1:8b-instruct-q4_K_M"):
        self.engine = engine
        self.vector_manager = vector_manager
        
        # Initialize LLM
        from langchain_ollama import ChatOllama
        self.llm = ChatOllama(model=model_name, temperature=0.0)
        
        # Initialize components
        self.query_classifier = QueryClassifier()
        self.handlers = self._initialize_handlers()
        
        # Set up fallback chains
        self._setup_fallback_chains()
        
        # System statistics
        self.system_stats = {
            'total_queries': 0,
            'handler_usage': {},
            'average_execution_time': 0.0,
            'success_rate': 0.0,
            'classification_accuracy': 0.0
        }
        
        LOGGER.info("Hybrid NL2SQL System initialized")
    
    def _initialize_handlers(self) -> Dict[str, Any]:
        """Initialize all handlers with configurations"""
        handler_configs = {
            'simple': {
                'max_execution_time': 2.0,
                'use_deterministic_builders': True,
                'fallback_to_llm': True,
                'cache_results': True
            },
            'visualization': {
                'max_execution_time': 5.0,
                'supported_chart_types': ['bar', 'pie', 'line', 'scatter'],
                'default_chart_size': (10, 6),
                'save_charts': True,
                'chart_output_dir': 'outputs/plots'
            },
            'analytical': {
                'max_execution_time': 8.0,
                'use_multi_step_analysis': True,
                'generate_insights': True,
                'max_analysis_depth': 3
            },
            'complex': {
                'max_execution_time': 15.0,
                'use_planning': True,
                'validate_sql': True,
                'max_retries': 3
            }
        }
        
        handlers = {}
        for handler_name, config in handler_configs.items():
            if handler_name == 'simple':
                handlers[handler_name] = SimpleQueryHandler(self.engine, self.vector_manager, self.llm, config)
            elif handler_name == 'visualization':
                handlers[handler_name] = VisualizationHandler(self.engine, self.vector_manager, self.llm, config)
            elif handler_name == 'analytical':
                # For now, use complex handler as analytical handler
                handlers[handler_name] = ComplexQueryHandler(self.engine, self.vector_manager, self.llm, config)
            elif handler_name == 'complex':
                handlers[handler_name] = ComplexQueryHandler(self.engine, self.vector_manager, self.llm, config)
        
        return handlers
    
    def _setup_fallback_chains(self):
        """Set up fallback chains between handlers"""
        # Simple handler falls back to complex handler
        if 'simple' in self.handlers and 'complex' in self.handlers:
            self.handlers['simple'].set_fallback_handler(self.handlers['complex'])
        
        # Visualization handler falls back to complex handler
        if 'visualization' in self.handlers and 'complex' in self.handlers:
            self.handlers['visualization'].set_fallback_handler(self.handlers['complex'])
        
        # Analytical handler falls back to complex handler
        if 'analytical' in self.handlers and 'complex' in self.handlers:
            self.handlers['analytical'].set_fallback_handler(self.handlers['complex'])
        
        LOGGER.info("Fallback chains established")
    
    def process_query(self, query: str, prefer_mode: Optional[str] = None) -> Dict[str, Any]:
        """Process query using hybrid routing system"""
        start_time = time.time()
        
        try:
            LOGGER.info(f"Processing query: {query[:50]}...")
            
            # Step 1: Classify query
            classification = self._classify_query(query, prefer_mode)
            
            # Step 2: Route to appropriate handler
            handler_result = self._route_to_handler(query, classification)
            
            # Step 3: Add system metadata
            result = self._add_system_metadata(handler_result, classification, start_time)
            
            # Step 4: Update system statistics
            self._update_system_stats(result, time.time() - start_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            LOGGER.error(f"System processing failed: {e}")
            return self._handle_system_error(query, e, execution_time)
    
    def _classify_query(self, query: str, prefer_mode: Optional[str] = None) -> QueryClassification:
        """Classify query with optional mode preference"""
        if prefer_mode:
            # Override classification with preferred mode
            LOGGER.info(f"Using preferred mode: {prefer_mode}")
            return QueryClassification(
                complexity='complex',
                type=prefer_mode.lower(),
                confidence=1.0,
                reasoning=f"User preferred mode: {prefer_mode}",
                handler_type=prefer_mode.lower()
            )
        
        # Get schema context for classification
        schema_context = self._get_schema_context()
        
        # Classify using intelligent classifier
        classification = self.query_classifier.classify(query, schema_context)
        
        LOGGER.info(f"Query classified as: {classification.type} ({classification.complexity}) - {classification.confidence:.2f} confidence")
        
        return classification
    
    def _route_to_handler(self, query: str, classification: QueryClassification) -> Dict[str, Any]:
        """Route query to appropriate handler"""
        handler_type = classification.handler_type
        
        # Check if handler exists
        if handler_type not in self.handlers:
            LOGGER.warning(f"Handler {handler_type} not found, using complex handler")
            handler_type = 'complex'
        
        # Get handler
        handler = self.handlers[handler_type]
        
        # Check if handler can handle the query
        if not handler.can_handle(query, classification):
            LOGGER.warning(f"Handler {handler_type} cannot handle query, using complex handler")
            handler = self.handlers['complex']
            handler_type = 'complex'
        
        LOGGER.info(f"Routing to {handler_type} handler")
        
        # Process with handler
        try:
            result = handler.process(query, classification)
            result['metadata']['handler_used'] = handler_type
            return result
            
        except Exception as e:
            LOGGER.error(f"Handler {handler_type} failed: {e}")
            
            # Try fallback to complex handler if not already using it
            if handler_type != 'complex':
                LOGGER.info("Attempting fallback to complex handler")
                try:
                    fallback_result = self.handlers['complex'].process(query, classification)
                    fallback_result['metadata']['handler_used'] = 'complex_fallback'
                    fallback_result['metadata']['fallback_reason'] = f"{handler_type}_handler_failed"
                    return fallback_result
                except Exception as fallback_error:
                    LOGGER.error(f"Fallback handler also failed: {fallback_error}")
            
            # Final fallback: generic LLM processing
            return self._generic_fallback(query, classification, e)
    
    def _generic_fallback(self, query: str, classification: QueryClassification, error: Exception) -> Dict[str, Any]:
        """Generic LLM fallback when all handlers fail"""
        LOGGER.warning("Using generic LLM fallback")
        
        try:
            # Simple LLM-based processing as last resort
            schema_context = self._get_schema_context()
            
            prompt = f"""Generate SQL for this query:

QUERY: "{query}"

SCHEMA CONTEXT:
{schema_context}

Generate a simple SQL query. Return ONLY the SQL statement wrapped in ```sql``` fences."""

            response = self.llm.invoke(prompt)
            sql = self._extract_sql_from_response(response)
            
            if sql:
                # Execute SQL
                with self.engine.connect() as conn:
                    result = conn.execute(sql)
                    data = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                return {
                    'mode': 'TABLE',
                    'table_markdown': data.to_markdown(index=False) if not data.empty else "(no rows)",
                    'sql': sql,
                    'metadata': {
                        'handler_used': 'generic_fallback',
                        'classification': {
                            'complexity': classification.complexity,
                            'type': classification.type,
                            'confidence': classification.confidence
                        },
                        'confidence': 0.3,
                        'errors': [str(error)],
                        'fallback_reason': 'all_handlers_failed'
                    }
                }
            else:
                raise ValueError("No SQL generated in fallback")
                
        except Exception as fallback_error:
            LOGGER.error(f"Generic fallback also failed: {fallback_error}")
            return {
                'mode': 'ERROR',
                'error': f"All processing methods failed. Original error: {error}. Fallback error: {fallback_error}",
                'sql': None,
                'metadata': {
                    'handler_used': 'error',
                    'classification': {
                        'complexity': classification.complexity,
                        'type': classification.type,
                        'confidence': classification.confidence
                    },
                    'confidence': 0.0,
                    'errors': [str(error), str(fallback_error)],
                    'fallback_reason': 'complete_system_failure'
                }
            }
    
    def _add_system_metadata(self, result: Dict[str, Any], classification: QueryClassification, 
                           start_time: float) -> Dict[str, Any]:
        """Add system-level metadata to result"""
        execution_time = time.time() - start_time
        
        # Ensure metadata exists
        if 'metadata' not in result:
            result['metadata'] = {}
        
        # Add classification metadata
        result['metadata']['classification'] = {
            'complexity': classification.complexity,
            'type': classification.type,
            'confidence': classification.confidence,
            'reasoning': classification.reasoning
        }
        
        # Add execution metadata
        result['metadata']['execution_time_ms'] = execution_time * 1000
        result['metadata']['system_version'] = 'hybrid_v1.0'
        result['metadata']['code_decision'] = 'continued_existing_code'
        
        # Add fallback information if present
        if classification.fallback_reason:
            result['metadata']['classification_fallback'] = classification.fallback_reason
        
        return result
    
    def _update_system_stats(self, result: Dict[str, Any], execution_time: float):
        """Update system statistics"""
        self.system_stats['total_queries'] += 1
        
        # Update handler usage
        handler_used = result.get('metadata', {}).get('handler_used', 'unknown')
        self.system_stats['handler_usage'][handler_used] = self.system_stats['handler_usage'].get(handler_used, 0) + 1
        
        # Update average execution time
        total_time = self.system_stats['average_execution_time'] * (self.system_stats['total_queries'] - 1)
        self.system_stats['average_execution_time'] = (total_time + execution_time) / self.system_stats['total_queries']
        
        # Update success rate
        success = 'error' not in result and result.get('mode') != 'ERROR'
        if success:
            successful_queries = self.system_stats.get('successful_queries', 0) + 1
        else:
            successful_queries = self.system_stats.get('successful_queries', 0)
        
        self.system_stats['successful_queries'] = successful_queries
        self.system_stats['success_rate'] = successful_queries / self.system_stats['total_queries']
    
    def _handle_system_error(self, query: str, error: Exception, execution_time: float) -> Dict[str, Any]:
        """Handle system-level errors"""
        LOGGER.error(f"System error processing query '{query}': {error}")
        
        return {
            'mode': 'ERROR',
            'error': f"System error: {str(error)}",
            'sql': None,
            'metadata': {
                'handler_used': 'system_error',
                'classification': {
                    'complexity': 'unknown',
                    'type': 'error',
                    'confidence': 0.0
                },
                'confidence': 0.0,
                'execution_time_ms': execution_time * 1000,
                'errors': [str(error)],
                'system_version': 'hybrid_v1.0',
                'code_decision': 'continued_existing_code'
            }
        }
    
    def _get_schema_context(self) -> str:
        """Get schema context for queries"""
        try:
            context = self.vector_manager.similarity_search("schema tables columns", top_k=3)
            return "\n\n".join(context.texts[:2]) if context.texts else "Schema context not available"
        except Exception as e:
            LOGGER.warning(f"Failed to get schema context: {e}")
            return "Schema context not available"
    
    def _extract_sql_from_response(self, response) -> Optional[str]:
        """Extract SQL from LLM response"""
        try:
            content = response.content if hasattr(response, 'content') else str(response)
            
            if '```sql' in content:
                start = content.find('```sql') + 6
                end = content.find('```', start)
                sql = content[start:end].strip()
            elif '```' in content:
                start = content.find('```') + 3
                end = content.find('```', start)
                sql = content[start:end].strip()
            else:
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
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = self.system_stats.copy()
        
        # Add handler-specific stats
        stats['handler_stats'] = {}
        for handler_name, handler in self.handlers.items():
            if hasattr(handler, 'get_stats'):
                stats['handler_stats'][handler_name] = handler.get_stats()
        
        # Add classification cache stats
        if hasattr(self.query_classifier, 'get_cache_stats'):
            stats['classification_cache'] = self.query_classifier.get_cache_stats()
        
        return stats
    
    def clear_caches(self):
        """Clear all caches"""
        # Clear classification cache
        if hasattr(self.query_classifier, 'clear_cache'):
            self.query_classifier.clear_cache()
        
        # Clear handler caches
        for handler in self.handlers.values():
            if hasattr(handler, 'clear_cache'):
                handler.clear_cache()
        
        LOGGER.info("All caches cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health_status = {
            'overall': 'healthy',
            'components': {},
            'timestamp': time.time()
        }
        
        # Check database connection
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            health_status['components']['database'] = 'healthy'
        except Exception as e:
            health_status['components']['database'] = f'unhealthy: {e}'
            health_status['overall'] = 'unhealthy'
        
        # Check LLM availability
        try:
            response = self.llm.invoke("Test")
            health_status['components']['llm'] = 'healthy'
        except Exception as e:
            health_status['components']['llm'] = f'unhealthy: {e}'
            health_status['overall'] = 'unhealthy'
        
        # Check handlers
        for handler_name, handler in self.handlers.items():
            try:
                # Simple health check
                handler.can_handle("test query", QueryClassification(
                    complexity='simple', type='table', confidence=1.0, reasoning='test', handler_type='simple'
                ))
                health_status['components'][f'handler_{handler_name}'] = 'healthy'
            except Exception as e:
                health_status['components'][f'handler_{handler_name}'] = f'unhealthy: {e}'
                health_status['overall'] = 'unhealthy'
        
        return health_status


# Example usage and testing
if __name__ == "__main__":
    print("Hybrid NL2SQL System")
    print("=" * 40)
    print("System implementation complete")
