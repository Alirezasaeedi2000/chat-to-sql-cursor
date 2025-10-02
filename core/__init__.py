"""
Core utilities for the NLP-to-SQL system.
Safe modular components that can be imported without breaking existing functionality.
"""

from .database import ensure_dirs, create_engine_from_url, create_engine_from_env
from .sql_utils import (
    SafeSqlExecutor, 
    _strip_code_fences, 
    _extract_sql_from_text,
    _parse_json_block,
    _stringify_llm_content,
    SqlValidationError,
    QueryCache
)

__all__ = [
    'ensure_dirs',
    'create_engine_from_url', 
    'create_engine_from_env',
    'SafeSqlExecutor',
    '_strip_code_fences',
    '_extract_sql_from_text',
    '_parse_json_block',
    '_stringify_llm_content',
    'SqlValidationError',
    'QueryCache'
]
