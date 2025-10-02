"""
SQL utility functions and SafeSqlExecutor class.
Extracted from query_processor.py for better modularity.
"""

import json
import logging
import os
import re
import hashlib
import pickle
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import sqlparse
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .database import ensure_dirs

LOGGER = logging.getLogger(__name__)


def _strip_code_fences(text_value: str) -> str:
    """Remove code fence markers from text."""
    fenced = re.sub(
        r"^```(?:sql|json|\w+)?\n|\n```$",
        "",
        text_value.strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if fenced != text_value:
        return fenced.strip()
    return text_value.strip()


def _extract_sql_from_text(text_value: str) -> Optional[str]:
    """Extract SQL from text, handling code fences and fallback patterns."""
    pattern = re.compile(r"```sql\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
    m = pattern.search(text_value)
    if m:
        return m.group(1).strip()
    # fallback: look for SELECT start
    start = re.search(r"\bSELECT\b", text_value, flags=re.IGNORECASE)
    if start:
        return text_value[start.start() :].strip().rstrip("`")
    return None


def _parse_json_block(text_value: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from text block."""
    try:
        cleaned = _strip_code_fences(text_value)
        return json.loads(cleaned)
    except Exception:
        return None


def _stringify_llm_content(value: Any) -> str:
    """Convert LLM response content into a plain string for parsing."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"]) 
                elif "json" in item:
                    try:
                        parts.append(json.dumps(item["json"]))
                    except Exception:
                        parts.append(str(item["json"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))

        return "\n".join(parts)
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        try:
            return json.dumps(value)
        except Exception:
            return str(value)
    # Fallback
    return str(value)


class SqlValidationError(Exception):
    """Exception raised for SQL validation errors."""
    pass


class QueryCache:
    """Simple file-based cache for query results with TTL support."""

    def __init__(self, cache_dir: str = "outputs/cache", ttl_seconds: int = 3600, enabled: bool = True):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        ensure_dirs()

    def _get_cache_key(self, sql: str) -> str:
        """Generate a cache key from SQL query."""
        normalized = re.sub(r"\s+", " ", sql.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def get(self, sql: str) -> Optional[Tuple[pd.DataFrame, str]]:
        """Get cached result if exists and not expired."""
        if not self.enabled:
            return None
            
        cache_key = self._get_cache_key(sql)
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            # Check if cache is expired
            cache_time = os.path.getmtime(cache_path)
            if time.time() - cache_time > self.ttl_seconds:
                os.remove(cache_path)
                return None
                
            # Load cached result
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            LOGGER.warning(f"Failed to load cache: {e}")
            return None

    def set(self, sql: str, result: Tuple[pd.DataFrame, str]) -> None:
        """Cache a query result."""
        if not self.enabled:
            return
            
        cache_key = self._get_cache_key(sql)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            LOGGER.warning(f"Failed to save cache: {e}")

    def clear(self) -> None:
        """Clear all cached results."""
        if not os.path.exists(self.cache_dir):
            return
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
        except Exception as e:
            LOGGER.warning(f"Failed to clear cache: {e}")


class SafeSqlExecutor:
    """Safe SQL executor with validation, caching, and timeout protection."""

    def __init__(
        self,
        engine: Engine,
        default_limit: int = 50,
        max_limit: int = 1000,
        timeout_secs: int = 30,
        enable_cache: bool = True,
    ) -> None:
        self.engine = engine
        self.default_limit = default_limit
        self.max_limit = max_limit
        self.timeout_secs = timeout_secs
        self.cache = QueryCache() if enable_cache else None

    def validate_select_only(self, sql: str) -> None:
        """Validate that SQL is a safe SELECT-only query."""
        stripped = sql.strip().rstrip(";")
        if ";" in stripped:
            raise SqlValidationError("Multiple statements are not allowed.")
        parsed = sqlparse.parse(stripped)
        if not parsed:
            raise SqlValidationError("Invalid SQL.")
        if len(parsed) != 1:
            raise SqlValidationError("Only a single SELECT statement is allowed.")
        stmt = parsed[0]
        # Allow WITH ... SELECT
        tokens = [t for t in stmt.tokens if not t.is_whitespace]
        keywords = " ".join(
            [
                t.value.upper()
                for t in tokens
                if t.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.DML)
            ]
        )
        forbidden = [
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "DROP",
            "ALTER",
            "TRUNCATE",
            "REPLACE",
            "MERGE",
            "GRANT",
            "REVOKE",
            "CALL",
            "USE",
            "SET",
            "SHOW",
            "DESCRIBE",
            "EXPLAIN ",
            # Exfiltration / file I/O vectors
            "OUTFILE",
            "DUMPFILE",
            "LOAD_FILE",
            "LOAD DATA",
            "INFILE",
            "INTO OUTFILE",
            "INTO DUMPFILE",
        ]
        # Also check for these patterns in the original SQL text
        sql_upper = sql.upper()
        if any(
            f in sql_upper
            for f in ["INTO OUTFILE", "INTO DUMPFILE", "LOAD_FILE", "LOAD DATA"]
        ):
            raise SqlValidationError("File operations are not allowed.")
        if any(f in keywords for f in forbidden):
            raise SqlValidationError("Only SELECT queries are allowed.")
        if not (
            " SELECT " in f" {keywords} "
            or keywords.strip().startswith("SELECT")
            or "WITH" in keywords
        ):
            raise SqlValidationError("Query must be a SELECT.")

    def _inject_exec_timeout_hint(self, sql: str) -> str:
        """Inject MySQL MAX_EXECUTION_TIME hint after the first SELECT if absent."""
        try:
            if re.search(r"MAX_EXECUTION_TIME\s*\(\s*\d+\s*\)", sql, re.IGNORECASE):
                return sql
            # Find the first SELECT and inject the hint
            select_match = re.search(r"\bSELECT\b", sql, re.IGNORECASE)
            if select_match:
                # Insert the hint after SELECT
                before_select = sql[: select_match.end()]
                after_select = sql[select_match.end() :]
                return f"{before_select} /*+ MAX_EXECUTION_TIME({self.timeout_secs * 1000}) */ {after_select}"
            return sql
        except Exception:
            return sql

    def _clamp_or_inject_limit(self, sql: str) -> str:
        """Add or modify LIMIT clause to prevent large result sets."""
        sql_upper = sql.upper()
        limit_match = re.search(r"LIMIT\s+(\d+)", sql_upper)
        
        if limit_match:
            current_limit = int(limit_match.group(1))
            if current_limit > self.max_limit:
                # Replace with max limit
                return re.sub(
                    r"LIMIT\s+\d+",
                    f"LIMIT {self.max_limit}",
                    sql,
                    flags=re.IGNORECASE,
                )
            return sql
        else:
            # No LIMIT found, add one
            return f"{sql.rstrip(';')} LIMIT {self.default_limit}"

    def execute_select(self, sql: str, user_query: str = None, query_processor=None) -> Tuple[pd.DataFrame, str]:
        """Execute a SELECT query safely with caching and validation."""
        self.validate_select_only(sql)
        hinted = self._inject_exec_timeout_hint(sql)
        safe_sql = self._clamp_or_inject_limit(hinted)

        # Apply fixes before cache check
        if user_query and query_processor and hasattr(query_processor, '_apply_sql_fixes'):
            original_sql = safe_sql
            safe_sql = query_processor._apply_sql_fixes(safe_sql, user_query, None)
            if safe_sql != original_sql:
                LOGGER.info(f"Applied fixes: {original_sql} -> {safe_sql}")

        # Check cache first (after applying fixes)
        if self.cache:
            cached_result = self.cache.get(safe_sql)
            if cached_result is not None:
                LOGGER.info("Using cached result for SQL: %s", safe_sql)
                return cached_result

        # Execute query
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(safe_sql))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                # Cache the result
                if self.cache:
                    self.cache.set(safe_sql, (df, safe_sql))
                    
                return df, safe_sql
                
        except Exception as e:
            LOGGER.error(f"SQL execution failed: {e}")
            raise
