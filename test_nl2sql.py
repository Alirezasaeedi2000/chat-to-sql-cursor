#!/usr/bin/env python3
"""
Test suite for the Natural Language to SQL Assistant.
Comprehensive tests for all major components.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import json
from datetime import datetime

# Import our modules
from query_processor import (
    SafeSqlExecutor, QueryProcessor, SqlValidationError, QueryCache,
    _strip_code_fences, _extract_sql_from_text, _parse_json_block,
    _stringify_llm_content, create_engine_from_url
)
from query_history import QueryHistoryManager, QueryHistoryEntry
from vector import VectorStoreManager, QueryExpander, RetrievedContext


class TestSqlValidation(unittest.TestCase):
    """Test SQL validation and safety features."""
    
    def setUp(self):
        # Mock engine for testing
        self.mock_engine = Mock()
        self.executor = SafeSqlExecutor(self.mock_engine, enable_cache=False)
    
    def test_valid_select_queries(self):
        """Test that valid SELECT queries pass validation."""
        valid_queries = [
            "SELECT * FROM users",
            "SELECT id, name FROM customers WHERE active = 1",
            "WITH recent AS (SELECT * FROM orders WHERE date > '2023-01-01') SELECT * FROM recent",
            "select count(*) from products",  # case insensitive
        ]
        
        for query in valid_queries:
            with self.subTest(query=query):
                try:
                    self.executor.validate_select_only(query)
                except SqlValidationError:
                    self.fail(f"Valid query failed validation: {query}")
    
    def test_invalid_queries_rejected(self):
        """Test that non-SELECT queries are rejected."""
        invalid_queries = [
            "INSERT INTO users VALUES (1, 'test')",
            "UPDATE users SET name = 'new'",
            "DELETE FROM users",
            "DROP TABLE users",
            "CREATE TABLE test (id INT)",
            "SELECT * FROM users; DROP TABLE users;",
            "SELECT * FROM users INTO OUTFILE '/tmp/file'",
            "LOAD DATA INFILE '/tmp/file' INTO TABLE users",
        ]
        
        for query in invalid_queries:
            with self.subTest(query=query):
                with self.assertRaises(SqlValidationError):
                    self.executor.validate_select_only(query)
    
    def test_limit_injection(self):
        """Test automatic LIMIT injection and clamping."""
        # Test injection when no LIMIT exists
        sql = "SELECT * FROM users"
        result = self.executor._clamp_or_inject_limit(sql)
        self.assertIn("LIMIT 50", result)
        
        # Test clamping when LIMIT exceeds max
        sql = "SELECT * FROM users LIMIT 5000"
        result = self.executor._clamp_or_inject_limit(sql)
        self.assertIn("LIMIT 1000", result)
        
        # Test valid LIMIT passes through
        sql = "SELECT * FROM users LIMIT 25"
        result = self.executor._clamp_or_inject_limit(sql)
        self.assertIn("LIMIT 25", result)


class TestQueryCache(unittest.TestCase):
    """Test query result caching functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache = QueryCache(cache_dir=self.temp_dir, ttl_seconds=3600)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_set_get(self):
        """Test basic cache set and get operations."""
        sql = "SELECT * FROM test"
        df = pd.DataFrame({'id': [1, 2], 'name': ['a', 'b']})
        result = (df, sql)
        
        # Set cache
        self.cache.set(sql, result)
        
        # Get from cache
        cached = self.cache.get(sql)
        self.assertIsNotNone(cached)
        
        cached_df, cached_sql = cached
        pd.testing.assert_frame_equal(cached_df, df)
        self.assertEqual(cached_sql, sql)
    
    def test_cache_miss(self):
        """Test cache miss for non-existent queries."""
        result = self.cache.get("SELECT * FROM nonexistent")
        self.assertIsNone(result)
    
    def test_cache_normalization(self):
        """Test that similar queries hit the same cache."""
        sql1 = "SELECT * FROM test"
        sql2 = "  SELECT   *   FROM   test  "
        
        df = pd.DataFrame({'id': [1]})
        self.cache.set(sql1, (df, sql1))
        
        # Should hit the same cache entry
        cached = self.cache.get(sql2)
        self.assertIsNotNone(cached)


class TestStringParsing(unittest.TestCase):
    """Test string parsing utilities."""
    
    def test_strip_code_fences(self):
        """Test removal of code fence markers."""
        test_cases = [
            ("```sql\nSELECT * FROM test\n```", "SELECT * FROM test"),
            ("```\nsome code\n```", "some code"),
            ("no fences", "no fences"),
            ("```json\n{\"key\": \"value\"}\n```", "{\"key\": \"value\"}"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = _strip_code_fences(input_text)
                self.assertEqual(result, expected)
    
    def test_extract_sql_from_text(self):
        """Test SQL extraction from mixed text."""
        text_with_sql = """
        Here's the query you need:
        ```sql
        SELECT id, name FROM users WHERE active = 1
        ```
        This should work great!
        """
        
        result = _extract_sql_from_text(text_with_sql)
        expected = "SELECT id, name FROM users WHERE active = 1"
        self.assertEqual(result, expected)
    
    def test_parse_json_block(self):
        """Test JSON parsing from text blocks."""
        json_text = '```json\n{"mode": "TABLE", "reason": "Show data"}\n```'
        result = _parse_json_block(json_text)
        expected = {"mode": "TABLE", "reason": "Show data"}
        self.assertEqual(result, expected)


class TestQueryHistory(unittest.TestCase):
    """Test query history management."""
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_file.close()
        self.history = QueryHistoryManager(self.temp_file.name, max_entries=100)
    
    def tearDown(self):
        os.unlink(self.temp_file.name)
    
    def test_add_entry(self):
        """Test adding entries to history."""
        entry_id = self.history.add_entry(
            user_query="Show all users",
            sql_query="SELECT * FROM users",
            mode="TABLE",
            execution_time_ms=123.45,
            row_count=10
        )
        
        self.assertIsNotNone(entry_id)
        self.assertTrue(entry_id.startswith("q_"))
        
        # Check entry was added
        entry = self.history.get_entry(entry_id)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.user_query, "Show all users")
        self.assertEqual(entry.sql_query, "SELECT * FROM users")
        self.assertEqual(entry.mode, "TABLE")
        self.assertEqual(entry.execution_time_ms, 123.45)
        self.assertEqual(entry.row_count, 10)
    
    def test_search_history(self):
        """Test searching through history."""
        # Add test entries
        self.history.add_entry("Show users", "SELECT * FROM users", "TABLE")
        self.history.add_entry("Count orders", "SELECT COUNT(*) FROM orders", "SHORT_ANSWER")
        self.history.add_entry("Show products", "SELECT * FROM products", "TABLE")
        
        # Search by query text
        results = self.history.search_history("users")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].user_query, "Show users")
        
        # Search by mode
        results = self.history.search_history(mode="TABLE")
        self.assertEqual(len(results), 2)
        
        # Search for non-existent
        results = self.history.search_history("nonexistent")
        self.assertEqual(len(results), 0)
    
    def test_favorites(self):
        """Test favorite functionality."""
        entry_id = self.history.add_entry("Test query", "SELECT 1", "TABLE")
        
        # Initially not favorite
        entry = self.history.get_entry(entry_id)
        self.assertFalse(entry.is_favorite)
        
        # Toggle to favorite
        is_fav = self.history.toggle_favorite(entry_id)
        self.assertTrue(is_fav)
        
        entry = self.history.get_entry(entry_id)
        self.assertTrue(entry.is_favorite)
        
        # Toggle back
        is_fav = self.history.toggle_favorite(entry_id)
        self.assertFalse(is_fav)
    
    def test_statistics(self):
        """Test statistics generation."""
        # Add some test data
        self.history.add_entry("Query 1", "SELECT 1", "TABLE", 100, 5)
        self.history.add_entry("Query 2", "SELECT 2", "SHORT_ANSWER", 200, 1)
        self.history.add_entry("Query 3", None, "TABLE", None, None, "Some error")
        
        stats = self.history.get_statistics()
        
        self.assertEqual(stats['total_queries'], 3)
        self.assertEqual(stats['successful_queries'], 2)
        self.assertEqual(stats['failed_queries'], 1)
        self.assertEqual(stats['success_rate'], 2/3)
        self.assertEqual(stats['average_execution_time_ms'], 150)  # (100 + 200) / 2


class TestQueryExpander(unittest.TestCase):
    """Test query expansion functionality."""
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        synonyms = {
            "customer": ["client", "user", "buyer"],
            "order": ["purchase", "transaction"],
        }
        json.dump(synonyms, self.temp_file)
        self.temp_file.close()
        
        self.expander = QueryExpander(self.temp_file.name)
    
    def tearDown(self):
        os.unlink(self.temp_file.name)
    
    def test_expansion(self):
        """Test query expansion with synonyms."""
        query = "Show me customer orders"
        expanded = self.expander.expand(query)
        
        self.assertIn("customer", expanded)
        self.assertIn("client", expanded)
        self.assertIn("user", expanded)
        self.assertIn("buyer", expanded)
        self.assertIn("purchase", expanded)
        self.assertIn("transaction", expanded)
    
    def test_no_expansion_needed(self):
        """Test that queries without matching terms are unchanged."""
        query = "Show me products"
        expanded = self.expander.expand(query)
        self.assertEqual(query, expanded)


class TestVectorStoreManager(unittest.TestCase):
    """Test vector store functionality."""
    
    def setUp(self):
        # Create temporary directory for vector store
        self.temp_dir = tempfile.mkdtemp()
        self.vector_manager = VectorStoreManager(persist_directory=self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('vector.inspect')
    def test_schema_snippet_generation(self, mock_inspect):
        """Test schema snippet generation."""
        # Mock database inspection
        mock_inspector = Mock()
        mock_inspect.return_value = mock_inspector
        
        mock_inspector.get_columns.return_value = [
            {'name': 'id', 'type': 'INTEGER'},
            {'name': 'name', 'type': 'VARCHAR(100)'},
        ]
        mock_inspector.get_pk_constraint.return_value = {'constrained_columns': ['id']}
        mock_inspector.get_foreign_keys.return_value = []
        
        mock_engine = Mock()
        
        snippet = self.vector_manager._schema_snippet(mock_engine, 'test_table')
        
        self.assertIn('test_table', snippet)
        self.assertIn('id(INTEGER)', snippet)
        self.assertIn('name(VARCHAR(100))', snippet)
        self.assertIn('Primary key: id', snippet)


class TestWebAPI(unittest.TestCase):
    """Test web API endpoints."""
    
    def setUp(self):
        # Import here to avoid issues if Flask not available
        from web_server import app
        app.config['TESTING'] = True
        self.client = app.test_client()
        
        # Mock the global instances
        with patch('web_server.qp') as mock_qp, patch('web_server.history') as mock_history:
            mock_result = Mock()
            mock_result.mode = "TABLE"
            mock_result.sql = "SELECT * FROM test"
            mock_result.table_markdown = "|id|name|\n|1|test|"
            mock_result.short_answer = None
            mock_result.analysis = None
            mock_result.visualization_path = None
            mock_result.metadata = {"row_count": 1}
            
            mock_qp.process.return_value = mock_result
            mock_history.search_history.return_value = []
            
            self.mock_qp = mock_qp
            self.mock_history = mock_history
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('status', data)
        self.assertIn('timestamp', data)
    
    def test_query_endpoint_missing_data(self):
        """Test query endpoint with missing data."""
        response = self.client.post('/api/query', json={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)
    
    def test_history_endpoint(self):
        """Test history retrieval endpoint."""
        response = self.client.get('/api/history')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, list)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    @patch('query_processor.create_engine_from_url')
    def test_full_pipeline_mock(self, mock_create_engine):
        """Test the full pipeline with mocked database."""
        # Mock engine and connection
        mock_engine = Mock()
        mock_conn = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        # Mock pandas read_sql to return test data
        with patch('pandas.read_sql') as mock_read_sql:
            mock_df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
            mock_read_sql.return_value = mock_df
            
            # Mock the LLM responses
            with patch('query_processor.ChatOllama') as mock_llm_class:
                mock_llm = Mock()
                mock_llm_class.return_value = mock_llm
                
                # Mock mode detection response
                mock_mode_response = Mock()
                mock_mode_response.content = '{"mode": "TABLE", "reason": "Show data"}'
                
                # Mock SQL generation response
                mock_sql_response = Mock()
                mock_sql_response.content = '```sql\nSELECT id, name FROM users LIMIT 50\n```'
                
                mock_llm.invoke.side_effect = [mock_mode_response, mock_sql_response]
                
                # Create vector manager and processor
                vector_manager = VectorStoreManager()
                processor = QueryProcessor(mock_engine, vector_manager)
                
                # Process a query
                result = processor.process("Show me all users")
                
                # Verify results
                self.assertEqual(result.mode, "TABLE")
                self.assertIsNotNone(result.sql)
                self.assertIsNotNone(result.table_markdown)
                self.assertEqual(result.metadata["row_count"], 2)


def run_tests():
    """Run all tests and return success status."""
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
