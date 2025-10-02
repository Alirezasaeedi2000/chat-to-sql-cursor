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

from langchain_ollama import ChatOllama

from vector import VectorStoreManager, RetrievedContext
import mcp_handler
from query_history import QueryHistoryManager
import networkx as nx


LOGGER = logging.getLogger(__name__)


def ensure_dirs() -> None:
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/exports", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/cache", exist_ok=True)


def create_engine_from_url(db_url: str) -> Engine:
    connect_args: Dict[str, Any] = {}
    if db_url.startswith("mysql+"):
        connect_args["connect_timeout"] = 10
        connect_args["charset"] = "utf8mb4"
    elif db_url.startswith("postgresql+"):
        connect_args["connect_timeout"] = 10
    elif db_url.startswith("sqlite+"):
        # SQLite specific optimizations
        connect_args["check_same_thread"] = False

    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=10,
        connect_args=connect_args,
        echo=False,  # Set to True for SQL debugging
    )
    return engine


def create_engine_from_env() -> Engine:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    # Test the connection with better error messaging
    try:
        engine = create_engine_from_url(db_url)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        if "Access denied" in str(e):
            raise RuntimeError(
                f"Database authentication failed. Please check your DATABASE_URL credentials: {e}"
            )
        elif "Can't connect" in str(e) or "Connection refused" in str(e):
            raise RuntimeError(
                f"Cannot connect to database server. Please verify the host and port in DATABASE_URL: {e}"
            )
        elif "Unknown database" in str(e):
            raise RuntimeError(
                f"Database does not exist. Please check the database name in DATABASE_URL: {e}"
            )
        else:
            raise RuntimeError(f"Database connection failed: {e}")


def _strip_code_fences(text_value: str) -> str:
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
    try:
        cleaned = _strip_code_fences(text_value)
        return json.loads(cleaned)
    except Exception:
        return None


def _stringify_llm_content(value: Any) -> str:
    """Convert LLM response content into a plain string for parsing.

    Handles cases where the content can be a list of chunks or a dict.
    """
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
        try:
            cache_key = self._get_cache_key(sql)
            cache_path = self._get_cache_path(cache_key)

            if not os.path.exists(cache_path):
                return None

            # Check if expired
            if time.time() - os.path.getmtime(cache_path) > self.ttl_seconds:
                try:
                    os.remove(cache_path)
                except Exception:
                    pass
                return None

            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            LOGGER.debug(f"Cache get failed: {e}")
            return None

    def set(self, sql: str, result: Tuple[pd.DataFrame, str]) -> None:
        """Cache the query result."""
        if not self.enabled:
            return
        try:
            cache_key = self._get_cache_key(sql)
            cache_path = self._get_cache_path(cache_key)

            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            LOGGER.debug(f"Cache set failed: {e}")

    def clear(self) -> None:
        """Clear all cached results."""
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith(".pkl"):
                    os.remove(os.path.join(self.cache_dir, file))
        except Exception as e:
            LOGGER.debug(f"Cache clear failed: {e}")
    
    def disable(self) -> None:
        """Disable caching temporarily."""
        self.enabled = False
    
    def enable(self) -> None:
        """Re-enable caching."""
        self.enabled = True


class SafeSqlExecutor:
    """Guards and executes SELECT-only SQL with LIMIT enforcement and timeouts."""

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
        """Inject MySQL MAX_EXECUTION_TIME hint after the first SELECT if absent.

        This is a no-op for databases that ignore MySQL hints; safe to include for MySQL dialects.
        """
        try:
            if re.search(r"MAX_EXECUTION_TIME\s*\(", sql, flags=re.IGNORECASE):
                return sql
            # Find the first SELECT keyword and inject the hint right after it
            match = re.search(r"\bSELECT\b", sql, flags=re.IGNORECASE)
            if not match:
                return sql
            ms = max(int(self.timeout_secs * 1000), 1)
            start, end = match.span()
            return sql[:end] + f" /*+ MAX_EXECUTION_TIME({ms}) */" + sql[end:]
        except Exception:
            return sql

    def _clamp_or_inject_limit(self, sql: str) -> str:
        # crude but effective LIMIT detection and clamping
        limit_regex = re.compile(
            r"\bLIMIT\s+(\d+)(?:\s*,\s*(\d+))?\b", flags=re.IGNORECASE
        )

        def repl(m: re.Match) -> str:
            first = int(m.group(1))
            second = m.group(2)
            if second is not None:
                second_val = int(second)
                second_val = min(second_val, self.max_limit)
                return f"LIMIT {first}, {second_val}"
            clamped = min(first, self.max_limit)
            return f"LIMIT {clamped}"

        if limit_regex.search(sql):
            return limit_regex.sub(repl, sql)
        # inject default limit at the end
        sql_no_semicolon = sql.strip().rstrip(";")
        return f"{sql_no_semicolon} LIMIT {self.default_limit}"

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
        reraise=True,
        retry=retry_if_exception_type(SQLAlchemyError),
    )
    def execute_select(self, sql: str, user_query: str = None, query_processor=None) -> Tuple[pd.DataFrame, str]:
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

        LOGGER.info("Executing SQL: %s", safe_sql)
        with self.engine.connect() as conn:
            df = pd.read_sql(text(safe_sql), conn)

        result = (df, safe_sql)

        # Cache the result
        if self.cache:
            self.cache.set(safe_sql, result)

        return result

    def explain(self, sql: str) -> pd.DataFrame:
        self.validate_select_only(sql)
        explain_sql = f"EXPLAIN {sql.strip().rstrip(';')}"
        with self.engine.connect() as conn:
            df = pd.read_sql(text(explain_sql), conn)
        return df

    def get_schema_summary(self) -> Dict[str, Any]:
        from sqlalchemy import inspect

        inspector = inspect(self.engine)
        result: Dict[str, Any] = {"tables": []}
        for t in inspector.get_table_names():
            cols = inspector.get_columns(t)
            result["tables"].append(
                {
                "name": t,
                    "columns": [
                        {"name": c.get("name"), "type": str(c.get("type"))}
                        for c in cols
                    ],
                }
            )
        return result

    def describe_table(self, table_name: str) -> Dict[str, Any]:
        from sqlalchemy import inspect

        inspector = inspect(self.engine)
        cols = inspector.get_columns(table_name)
        return {
            "name": table_name,
            "columns": [
                {"name": c.get("name"), "type": str(c.get("type"))} for c in cols
            ],
        }

    def find_tables(self, pattern: str) -> List[str]:
        from sqlalchemy import inspect

        inspector = inspect(self.engine)
        names = inspector.get_table_names()
        regex = re.compile(pattern, re.IGNORECASE)
        return [n for n in names if regex.search(n)]

    def find_columns(self, pattern: str) -> List[Tuple[str, str]]:
        from sqlalchemy import inspect

        inspector = inspect(self.engine)
        regex = re.compile(pattern, re.IGNORECASE)
        matches: List[Tuple[str, str]] = []
        for t in inspector.get_table_names():
            for c in inspector.get_columns(t):
                name = c.get("name")
                if name and regex.search(name):
                    matches.append((t, name))
        return matches

    def distinct_values(self, table: str, column: str, limit: int = 50) -> List[Any]:
        sql = f"SELECT DISTINCT `{column}` AS val FROM `{table}` LIMIT :lim"
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(sql), {"lim": min(limit, self.max_limit)}
            ).fetchall()
        return [r[0] for r in rows]


@dataclass
class NL2SQLOutput:
    mode: str
    sql: Optional[str]
    table_markdown: Optional[str]
    short_answer: Optional[str]
    analysis: Optional[str]
    visualization_path: Optional[str]
    metadata: Dict[str, Any]


class QueryProcessor:
    """Coordinates RAG retrieval, intent detection, SQL generation, execution, and multi-mode formatting."""

    def __init__(
        self,
        engine: Engine,
        vector_manager: VectorStoreManager,
        model_name: str = "llama3.1:8b-instruct-q4_K_M",
        temperature: float = 0.0,
        default_limit: int = 50,
        max_limit: int = 1000,
        timeout_secs: int = 30,
    ) -> None:
        ensure_dirs()
        self.engine = engine
        self.vector_manager = vector_manager
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        # Use a faster model for simple tasks like mode detection
        self.fast_llm = ChatOllama(
            model="llama3.2:1b",  # Always use fastest model for mode detection
            temperature=0.0,
        )
        self.safe_exec = SafeSqlExecutor(
            engine,
            default_limit=default_limit,
            max_limit=max_limit,
            timeout_secs=timeout_secs,
        )
        self.history = QueryHistoryManager()
        # Load schema identifiers for grounding and validation
        self.allowed_tables: set[str] = set()
        self.allowed_columns: set[str] = set()
        self.table_to_columns: Dict[str, List[str]] = {}
        
        # Performance optimization: cache mode detection results
        self._mode_cache: Dict[str, str] = {}
        self._intent_cache: Dict[str, Dict[str, Any]] = {}
        self._load_schema_identifiers()
        # Placeholder for a simple FK graph (table -> list of (col, ref_table, ref_col))
        self.fk_graph: Dict[str, List[Tuple[str, str, str]]] = {}
        self._load_fk_graph()
        # Graph RAG: schema graph (tables, columns, fk paths)
        self.schema_graph: nx.Graph = nx.Graph()
        self._build_schema_graph()
        
        # ADVANCED FEATURES: Community Detection and Search Strategies
        self.schema_communities: Dict[str, Any] = {}
        self.community_reports: Dict[str, Any] = {}
        self._detect_schema_communities()
        self._generate_schema_reports()

    def _can_inspect_engine(self) -> bool:
        """Return True if the engine appears to be a real SQLAlchemy Engine, not a mock.

        Guards test environments where a unittest.mock object is supplied, which
        raises 'No inspection system is available' when calling sqlalchemy.inspect.
        """
        try:
            # Must be an actual Engine instance
            if not isinstance(self.engine, Engine):
                return False
            # Avoid unittest.mock objects
            mod = getattr(self.engine.__class__, "__module__", "")
            if "unittest.mock" in mod:
                return False
            return True
        except Exception:
            return False

    def _load_schema_identifiers(self) -> None:
        if not self._can_inspect_engine():
            LOGGER.debug("Skipping schema identifier load: non-inspectable engine")
            return
        try:
            insp = inspect(self.engine)
            tables = insp.get_table_names()
            self.table_to_columns = {}
            for t in tables:
                cols = [c.get("name") for c in insp.get_columns(t) if c.get("name")]
                self.table_to_columns[t] = cols
            self.allowed_tables = set(tables)
            self.allowed_columns = set(
                [c for cols in self.table_to_columns.values() for c in cols]
            )
        except Exception as exc:
            LOGGER.debug("Failed to load schema identifiers: %s", exc)

    def _load_fk_graph(self) -> None:
        if not self._can_inspect_engine():
            LOGGER.debug("Skipping FK graph load: non-inspectable engine")
            return
        try:
            insp = inspect(self.engine)
            graph: Dict[str, List[Tuple[str, str, str]]] = {}
            for t in insp.get_table_names():
                fks = insp.get_foreign_keys(t)
                edges: List[Tuple[str, str, str]] = []
                for fk in fks:
                    cols = fk.get("constrained_columns", []) or []
                    rt = fk.get("referred_table")
                    rcols = fk.get("referred_columns", []) or []
                    if rt and cols and rcols:
                        for c, rc in zip(cols, rcols):
                            edges.append((c, rt, rc))
                graph[t] = edges
            self.fk_graph = graph
        except Exception as exc:
            LOGGER.debug("Failed to load FK graph: %s", exc)

    def _build_schema_graph(self) -> None:
        """Build an undirected schema graph with nodes for tables and columns, and edges for containment and FKs."""
        if not self._can_inspect_engine():
            LOGGER.debug("Skipping schema graph build: non-inspectable engine")
            return
        try:
            g: nx.Graph = nx.Graph()
            insp = inspect(self.engine)
            for t in insp.get_table_names():
                g.add_node(("table", t))
                cols = insp.get_columns(t)
                for c in cols:
                    col_name = c.get("name")
                    if not col_name:
                        continue
                    g.add_node(("column", t, col_name))
                    g.add_edge(("table", t), ("column", t, col_name), kind="has_column")
            for t in insp.get_table_names():
                for fk in insp.get_foreign_keys(t):
                    rt = fk.get("referred_table")
                    cols = fk.get("constrained_columns", []) or []
                    rcols = fk.get("referred_columns", []) or []
                    if not (rt and cols and rcols):
                        continue
                    for c, rc in zip(cols, rcols):
                        # Connect tables and specific columns
                        if g.has_node(("table", t)) and g.has_node(("table", rt)):
                            g.add_edge(
                                ("table", t), ("table", rt), kind="fk", via=(c, rc)
                            )
                        if g.has_node(("column", t, c)) and g.has_node(
                            ("column", rt, rc)
                        ):
                            g.add_edge(
                                ("column", t, c), ("column", rt, rc), kind="fk_col"
                            )
            # ENHANCED: Add inferred relationships since database has no foreign keys
            self._add_inferred_relationships(g, insp)
            
            self.schema_graph = g
        except Exception as exc:
            LOGGER.debug("Failed to build schema graph: %s", exc)

    def _add_inferred_relationships(self, g: nx.Graph, insp) -> None:
        """Add inferred table relationships based on column naming patterns and business logic."""
        try:
            tables = insp.get_table_names()
            
            # Define relationship patterns based on column names and business logic
            relationship_patterns = {
                # Worker relationships
                'workers': {
                    'section': ['production_info', 'person_hyg'],  # Workers belong to sections
                    'id': ['production_info'],  # Workers have production records
                },
                # Production relationships  
                'production_info': {
                    'bakeType': ['packaging_info'],  # Production batches have packaging
                    'bakeID': ['production_test'],  # Production has test results
                },
                # Packaging relationships
                'packaging_info': {
                    'tranNumber': ['transtatus'],  # Packaging has transaction status
                    'date': ['pack_waste'],  # Packaging dates match waste dates
                },
                # Transaction relationships
                'transtatus': {
                    'tranWeight': ['packaging_info'],  # Transaction weight relates to packaging
                },
                # Hygiene relationships
                'person_hyg': {
                    'personName': ['workers'],  # Hygiene records for workers
                    'date': ['production_info'],  # Hygiene checks on production dates
                },
                # Price relationships
                'prices': {
                    'date': ['production_info'],  # Prices on production dates
                },
                # User relationships
                'users': {
                    'created': ['workers'],  # Users create worker records
                }
            }
            
            # Add inferred edges
            for source_table, targets in relationship_patterns.items():
                if source_table not in tables:
                    continue
                    
                for column_pattern, target_tables in targets.items():
                    # Get source table columns
                    source_cols = insp.get_columns(source_table)
                    source_col_names = [col['name'].lower() for col in source_cols]
                    
                    # Check if column pattern exists in source table
                    if any(column_pattern.lower() in col_name for col_name in source_col_names):
                        for target_table in target_tables:
                            if target_table in tables:
                                # Add inferred relationship edge
                                if g.has_node(("table", source_table)) and g.has_node(("table", target_table)):
                                    g.add_edge(
                                        ("table", source_table), 
                                        ("table", target_table), 
                                        kind="inferred", 
                                        via=column_pattern,
                                        confidence=0.7  # Medium confidence for inferred relationships
                                    )
                                    LOGGER.debug(f"Inferred relationship: {source_table} -> {target_table} via {column_pattern}")
            
            # Add date-based relationships (common in manufacturing)
            date_tables = [t for t in tables if any('date' in col['name'].lower() for col in insp.get_columns(t))]
            for i, table1 in enumerate(date_tables):
                for table2 in date_tables[i+1:]:
                    if g.has_node(("table", table1)) and g.has_node(("table", table2)):
                        g.add_edge(
                            ("table", table1), 
                            ("table", table2), 
                            kind="date_based", 
                            via="date_columns",
                            confidence=0.5  # Lower confidence for date-based relationships
                        )
            
            LOGGER.info(f"Added inferred relationships to schema graph")
            
        except Exception as exc:
            LOGGER.warning(f"Failed to add inferred relationships: {exc}")

    def _detect_schema_communities(self) -> None:
        """ADVANCED FEATURE 1: Detect hierarchical communities in database schema using graph clustering"""
        try:
            if not self.schema_graph.nodes:
                LOGGER.debug("Skipping community detection: empty schema graph")
                return

            LOGGER.info("Starting schema community detection...")
            
            # Level 1: Table-level communities (broad categories)
            table_communities = self._extract_table_communities()
            
            # Level 2: Column-level communities (detailed categories)  
            column_communities = self._extract_column_communities()
            
            # Level 3: Relationship-based communities (FK clusters)
            fk_communities = self._detect_fk_communities()
            
            # Level 4: Semantic communities (based on naming patterns)
            semantic_communities = self._detect_semantic_communities()
            
            self.schema_communities = {
                'table_communities': table_communities,
                'column_communities': column_communities,
                'fk_communities': fk_communities,
                'semantic_communities': semantic_communities,
                'hierarchy_levels': {
                    'level_1': 'broad_domain_categories',
                    'level_2': 'detailed_functional_groups', 
                    'level_3': 'relationship_clusters',
                    'level_4': 'semantic_patterns'
                }
            }
            
            LOGGER.info(f"Schema community detection completed: {len(table_communities)} table communities, {len(column_communities)} column communities")
            
        except Exception as exc:
            LOGGER.error(f"Failed to detect schema communities: {exc}")
            self.schema_communities = {}

    def _extract_table_communities(self) -> Dict[str, List[str]]:
        """Extract table-level communities using graph clustering"""
        try:
            # Create table-only subgraph
            table_graph = nx.Graph()
            
            # Add table nodes
            for node in self.schema_graph.nodes:
                if node[0] == "table":
                    table_graph.add_node(node[1])
            
            # Add table relationships (foreign keys)
            for edge in self.schema_graph.edges(data=True):
                source, target, data = edge
                if (source[0] == "table" and target[0] == "table" and 
                    data.get('kind') == 'fk'):
                    table_graph.add_edge(source[1], target[1], weight=1.0)
            
            # Apply community detection algorithm
            if table_graph.number_of_nodes() > 1:
                # Use greedy modularity communities
                communities = list(nx.algorithms.community.greedy_modularity_communities(table_graph))
                
                # Convert to named communities
                named_communities = {}
                for i, community in enumerate(communities):
                    community_name = f"community_{i+1}"
                    named_communities[community_name] = list(community)
                
                # Apply domain-specific naming based on table patterns
                named_communities = self._apply_domain_naming(named_communities)
                
                return named_communities
            else:
                return {'single_table_community': list(table_graph.nodes())}
                
        except Exception as exc:
            LOGGER.error(f"Failed to extract table communities: {exc}")
            return {}

    def _extract_column_communities(self) -> Dict[str, List[str]]:
        """Extract column-level communities based on naming patterns and relationships"""
        try:
            column_communities = {
                'identifier_community': [],
                'date_time_community': [],
                'weight_measurement_community': [],
                'status_community': [],
                'name_text_community': [],
                'foreign_key_community': []
            }
            
            # Analyze all columns and categorize them
            for table_name, columns in self.table_to_columns.items():
                for column in columns:
                    column_lower = column.lower()
                    
                    # Identifier patterns
                    if any(pattern in column_lower for pattern in ['id', 'key', 'pk']):
                        column_communities['identifier_community'].append(f"{table_name}.{column}")
                    
                    # Date/time patterns
                    elif any(pattern in column_lower for pattern in ['date', 'time', 'created', 'updated', 'archived']):
                        column_communities['date_time_community'].append(f"{table_name}.{column}")
                    
                    # Weight/measurement patterns
                    elif any(pattern in column_lower for pattern in ['weight', 'amount', 'quantity', 'volume', 'usage']):
                        column_communities['weight_measurement_community'].append(f"{table_name}.{column}")
                    
                    # Status patterns
                    elif any(pattern in column_lower for pattern in ['status', 'state', 'free', 'active', 'inactive']):
                        column_communities['status_community'].append(f"{table_name}.{column}")
                    
                    # Name/text patterns
                    elif any(pattern in column_lower for pattern in ['name', 'title', 'description', 'type']):
                        column_communities['name_text_community'].append(f"{table_name}.{column}")
            
            # Remove empty communities
            column_communities = {k: v for k, v in column_communities.items() if v}
            
            return column_communities
            
        except Exception as exc:
            LOGGER.error(f"Failed to extract column communities: {exc}")
            return {}

    def _detect_fk_communities(self) -> Dict[str, List[str]]:
        """Detect communities based on foreign key relationships"""
        try:
            fk_communities = {}
            
            # Build FK relationship graph
            fk_graph = nx.Graph()
            
            for table, fk_list in self.fk_graph.items():
                fk_graph.add_node(table)
                for col, ref_table, ref_col in fk_list:
                    fk_graph.add_edge(table, ref_table, weight=1.0)
            
            if fk_graph.number_of_nodes() > 1:
                # Find connected components (FK clusters)
                components = list(nx.connected_components(fk_graph))
                
                for i, component in enumerate(components):
                    community_name = f"fk_cluster_{i+1}"
                    fk_communities[community_name] = list(component)
            else:
                fk_communities['single_fk_cluster'] = list(fk_graph.nodes())
            
            return fk_communities
            
        except Exception as exc:
            LOGGER.error(f"Failed to detect FK communities: {exc}")
            return {}

    def _detect_semantic_communities(self) -> Dict[str, List[str]]:
        """Detect communities based on semantic naming patterns"""
        try:
            semantic_communities = {
                'production_semantic': [],
                'hygiene_semantic': [],
                'packaging_semantic': [],
                'user_semantic': [],
                'worker_semantic': []
            }
            
            # Analyze table names for semantic patterns
            for table_name in self.table_to_columns.keys():
                table_lower = table_name.lower()
                
                if 'production' in table_lower:
                    semantic_communities['production_semantic'].append(table_name)
                elif 'hyg' in table_lower or 'hygiene' in table_lower:
                    semantic_communities['hygiene_semantic'].append(table_name)
                elif 'packaging' in table_lower or 'package' in table_lower:
                    semantic_communities['packaging_semantic'].append(table_name)
                elif 'user' in table_lower:
                    semantic_communities['user_semantic'].append(table_name)
                elif 'worker' in table_lower or 'person' in table_lower:
                    semantic_communities['worker_semantic'].append(table_name)
            
            # Remove empty communities
            semantic_communities = {k: v for k, v in semantic_communities.items() if v}
            
            return semantic_communities
            
        except Exception as exc:
            LOGGER.error(f"Failed to detect semantic communities: {exc}")
            return {}

    def _apply_domain_naming(self, communities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Apply domain-specific naming to communities based on table content"""
        try:
            named_communities = {}
            
            for community_name, tables in communities.items():
                # Analyze table names to determine domain
                domain_keywords = {
                    'production': ['production', 'bake', 'batch', 'manufacturing'],
                    'hygiene': ['hyg', 'hygiene', 'compliance', 'violation'],
                    'packaging': ['packaging', 'package', 'waste', 'weight'],
                    'user': ['user', 'account', 'access', 'level'],
                    'worker': ['worker', 'person', 'employee', 'staff'],
                    'transaction': ['tran', 'transaction', 'status', 'repo']
                }
                
                # Count keyword matches
                domain_scores = {}
                for domain, keywords in domain_keywords.items():
                    score = 0
                    for table in tables:
                        table_lower = table.lower()
                        for keyword in keywords:
                            if keyword in table_lower:
                                score += 1
                    domain_scores[domain] = score
                
                # Assign domain name to community
                if domain_scores:
                    best_domain = max(domain_scores, key=domain_scores.get)
                    if domain_scores[best_domain] > 0:
                        new_name = f"{best_domain}_community"
                    else:
                        new_name = f"general_{community_name}"
                else:
                    new_name = f"general_{community_name}"
                
                named_communities[new_name] = tables
            
            return named_communities
            
        except Exception as exc:
            LOGGER.error(f"Failed to apply domain naming: {exc}")
            return communities

    def _global_schema_search(self, query: str) -> Dict[str, Any]:
        """ADVANCED FEATURE 2A: Global schema search using community-level summaries"""
        try:
            LOGGER.info(f"Performing global schema search for: {query}")
            
            # Analyze query for domain keywords
            query_lower = query.lower()
            domain_scores = {}
            
            for community_name, community_data in self.schema_communities.get('semantic_communities', {}).items():
                score = 0
                domain = community_name.replace('_semantic', '')
                
                # Score based on domain keywords in query
                domain_keywords = {
                    'production': ['production', 'bake', 'batch', 'manufacturing', 'efficiency'],
                    'hygiene': ['hygiene', 'hyg', 'compliance', 'violation', 'clean'],
                    'packaging': ['packaging', 'package', 'waste', 'weight', 'material'],
                    'user': ['user', 'account', 'access', 'level', 'permission'],
                    'worker': ['worker', 'person', 'employee', 'staff', 'name']
                }
                
                if domain in domain_keywords:
                    for keyword in domain_keywords[domain]:
                        if keyword in query_lower:
                            score += query_lower.count(keyword)
                
                domain_scores[domain] = score
            
            # Get top domains
            top_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            
            # Build global context
            global_context = {
                'search_type': 'global',
                'top_domains': [domain for domain, score in top_domains if score > 0],
                'domain_scores': domain_scores,
                'relevant_communities': [],
                'schema_summary': {}
            }
            
            # Add relevant community information
            for domain, score in top_domains:
                if score > 0:
                    community_key = f"{domain}_semantic"
                    if community_key in self.schema_communities.get('semantic_communities', {}):
                        global_context['relevant_communities'].append({
                            'domain': domain,
                            'tables': self.schema_communities['semantic_communities'][community_key],
                            'score': score
                        })
            
            # Generate schema summary
            global_context['schema_summary'] = {
                'total_tables': len(self.table_to_columns),
                'total_communities': len(self.schema_communities.get('table_communities', {})),
                'primary_domains': [domain for domain, score in top_domains[:3]],
                'complexity_level': 'high' if len(global_context['relevant_communities']) > 2 else 'medium'
            }
            
            LOGGER.info(f"Global search completed: {len(global_context['relevant_communities'])} relevant communities found")
            return global_context
            
        except Exception as exc:
            LOGGER.error(f"Failed to perform global schema search: {exc}")
            return {'search_type': 'global', 'error': str(exc)}

    def _local_schema_search(self, query: str) -> Dict[str, Any]:
        """ADVANCED FEATURE 2B: Local schema search using entity-focused retrieval"""
        try:
            LOGGER.info(f"Performing local schema search for: {query}")
            
            # Extract entities from query
            entities = self._extract_query_entities(query)
            
            # Find specific tables and columns
            relevant_tables = set()
            relevant_columns = set()
            
            # Match query entities to schema entities
            for entity in entities['tables']:
                for table in self.table_to_columns.keys():
                    if entity.lower() in table.lower() or table.lower() in entity.lower():
                        relevant_tables.add(table)
            
            # ENHANCED: Use inferred relationships to find related tables
            if relevant_tables:
                for table in list(relevant_tables):
                    # Find tables connected via inferred relationships
                    if self.schema_graph.has_node(("table", table)):
                        for neighbor in self.schema_graph.neighbors(("table", table)):
                            if neighbor[0] == "table":  # Only table neighbors
                                neighbor_table = neighbor[1]
                                edge_data = self.schema_graph.get_edge_data(("table", table), neighbor)
                                if edge_data and edge_data.get('kind') in ['inferred', 'date_based']:
                                    relevant_tables.add(neighbor_table)
                                    LOGGER.debug(f"Added related table {neighbor_table} via {edge_data.get('kind')} relationship")
            
            # FORCE MULTI-TABLE: If query mentions relationships, ensure multiple tables
            query_lower = query.lower()
            if any(phrase in query_lower for phrase in ["with their", "with", "and their", "corresponding"]):
                # Force related tables based on query content
                if "workers" in query_lower and "production" in query_lower:
                    relevant_tables.update(["workers", "production_info"])
                if "packaging" in query_lower and "transaction" in query_lower:
                    relevant_tables.update(["packaging_info", "transtatus"])
                if "workers" in query_lower and "hygiene" in query_lower:
                    relevant_tables.update(["workers", "person_hyg"])
                if "production" in query_lower and "quality" in query_lower:
                    relevant_tables.update(["production_info", "production_test"])
            
            for entity in entities['columns']:
                for table, columns in self.table_to_columns.items():
                    for column in columns:
                        if entity.lower() in column.lower() or column.lower() in entity.lower():
                            relevant_tables.add(table)
                            relevant_columns.add(f"{table}.{column}")
            
            # Build local context with focused information
            local_context = {
                'search_type': 'local',
                'relevant_tables': list(relevant_tables),
                'relevant_columns': list(relevant_columns),
                'entity_matches': {
                    'tables': entities['tables'],
                    'columns': entities['columns'],
                    'keywords': entities['keywords']
                },
                'relationships': [],
                'focused_schema': {}
            }
            
            # Add relationships for relevant tables
            for table in relevant_tables:
                if table in self.fk_graph:
                    for col, ref_table, ref_col in self.fk_graph[table]:
                        if ref_table in relevant_tables:
                            local_context['relationships'].append({
                                'source': f"{table}.{col}",
                                'target': f"{ref_table}.{ref_col}",
                                'type': 'foreign_key'
                            })
            
            # Build focused schema
            for table in relevant_tables:
                if table in self.table_to_columns:
                    local_context['focused_schema'][table] = {
                        'columns': self.table_to_columns[table],
                        'row_count': 'unknown',  # Could be enhanced with actual counts
                        'relationships': len([r for r in local_context['relationships'] if table in r['source'] or table in r['target']])
                    }
            
            LOGGER.info(f"Local search completed: {len(relevant_tables)} tables, {len(relevant_columns)} columns found")
            return local_context
            
        except Exception as exc:
            LOGGER.error(f"Failed to perform local schema search: {exc}")
            return {'search_type': 'local', 'error': str(exc)}

    def _drift_schema_search(self, query: str) -> Dict[str, Any]:
        """ADVANCED FEATURE 2C: Dynamic schema reasoning using semantic drift detection"""
        try:
            LOGGER.info(f"Performing drift schema search for: {query}")
            
            # Analyze query for semantic patterns and potential schema evolution
            query_analysis = self._analyze_query_semantics(query)
            
            # Detect potential schema drift or missing connections
            drift_context = {
                'search_type': 'drift',
                'query_analysis': query_analysis,
                'potential_missing_tables': [],
                'suggested_connections': [],
                'semantic_gaps': [],
                'reasoning_path': []
            }
            
            # Analyze for missing table connections
            mentioned_tables = query_analysis.get('mentioned_tables', [])
            if len(mentioned_tables) > 1:
                # Check if tables are connected in the schema
                for i, table1 in enumerate(mentioned_tables):
                    for table2 in mentioned_tables[i+1:]:
                        if not self._tables_connected(table1, table2):
                            drift_context['potential_missing_tables'].append({
                                'table1': table1,
                                'table2': table2,
                                'reason': 'No direct relationship found',
                                'suggestion': f"Consider adding foreign key relationship between {table1} and {table2}"
                            })
            
            # Suggest schema improvements based on query patterns
            query_patterns = query_analysis.get('patterns', [])
            for pattern in query_patterns:
                if pattern == 'multi_table_join' and len(mentioned_tables) > 2:
                    drift_context['suggested_connections'].append({
                        'type': 'join_optimization',
                        'tables': mentioned_tables,
                        'suggestion': 'Consider creating intermediate table for complex multi-table relationships'
                    })
                elif pattern == 'temporal_analysis':
                    drift_context['suggested_connections'].append({
                        'type': 'temporal_indexing',
                        'tables': mentioned_tables,
                        'suggestion': 'Consider adding temporal indexes for better time-series query performance'
                    })
            
            # Generate reasoning path
            drift_context['reasoning_path'] = [
                f"Analyzed query: '{query}'",
                f"Identified {len(mentioned_tables)} tables: {mentioned_tables}",
                f"Detected {len(query_patterns)} patterns: {query_patterns}",
                f"Found {len(drift_context['potential_missing_tables'])} potential missing connections",
                f"Suggested {len(drift_context['suggested_connections'])} schema improvements"
            ]
            
            LOGGER.info(f"Drift search completed: {len(drift_context['potential_missing_tables'])} missing connections identified")
            return drift_context
            
        except Exception as exc:
            LOGGER.error(f"Failed to perform drift schema search: {exc}")
            return {'search_type': 'drift', 'error': str(exc)}

    def _extract_query_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query for local search"""
        try:
            query_lower = query.lower()
            
            entities = {
                'tables': [],
                'columns': [],
                'keywords': []
            }
            
            # Extract table mentions
            for table in self.table_to_columns.keys():
                if table.lower() in query_lower:
                    entities['tables'].append(table)
            
            # Extract column mentions
            for table, columns in self.table_to_columns.items():
                for column in columns:
                    if column.lower() in query_lower:
                        entities['columns'].append(f"{table}.{column}")
            
            # Extract keywords (non-schema terms)
            words = query_lower.split()
            schema_terms = set()
            for table in self.table_to_columns.keys():
                schema_terms.update(table.lower().split('_'))
            for columns in self.table_to_columns.values():
                for column in columns:
                    schema_terms.update(column.lower().split('_'))
            
            entities['keywords'] = [word for word in words if word not in schema_terms and len(word) > 2]
            
            return entities
            
        except Exception as exc:
            LOGGER.error(f"Failed to extract query entities: {exc}")
            return {'tables': [], 'columns': [], 'keywords': []}

    def _analyze_query_semantics(self, query: str) -> Dict[str, Any]:
        """Analyze query semantics for drift detection"""
        try:
            query_lower = query.lower()
            
            analysis = {
                'mentioned_tables': [],
                'patterns': [],
                'complexity': 'simple',
                'semantic_intent': 'unknown'
            }
            
            # Extract mentioned tables
            for table in self.table_to_columns.keys():
                if table.lower() in query_lower:
                    analysis['mentioned_tables'].append(table)
            
            # Detect patterns
            if 'join' in query_lower or 'with' in query_lower:
                analysis['patterns'].append('multi_table_join')
            if any(word in query_lower for word in ['trend', 'over time', 'historical', 'past']):
                analysis['patterns'].append('temporal_analysis')
            if any(word in query_lower for word in ['average', 'sum', 'count', 'max', 'min']):
                analysis['patterns'].append('aggregation')
            if any(word in query_lower for word in ['group', 'by', 'each']):
                analysis['patterns'].append('grouping')
            
            # Determine complexity
            if len(analysis['mentioned_tables']) > 2:
                analysis['complexity'] = 'complex'
            elif len(analysis['patterns']) > 2:
                analysis['complexity'] = 'medium'
            
            # Determine semantic intent
            if any(word in query_lower for word in ['show', 'display', 'list']):
                analysis['semantic_intent'] = 'retrieval'
            elif any(word in query_lower for word in ['analyze', 'compare', 'trend']):
                analysis['semantic_intent'] = 'analysis'
            elif any(word in query_lower for word in ['count', 'total', 'sum']):
                analysis['semantic_intent'] = 'aggregation'
            
            return analysis
            
        except Exception as exc:
            LOGGER.error(f"Failed to analyze query semantics: {exc}")
            return {'mentioned_tables': [], 'patterns': [], 'complexity': 'simple', 'semantic_intent': 'unknown'}

    def _tables_connected(self, table1: str, table2: str) -> bool:
        """Check if two tables are connected via foreign key relationship"""
        try:
            # Check direct FK relationships
            if table1 in self.fk_graph:
                for col, ref_table, ref_col in self.fk_graph[table1]:
                    if ref_table == table2:
                        return True
            
            if table2 in self.fk_graph:
                for col, ref_table, ref_col in self.fk_graph[table2]:
                    if ref_table == table1:
                        return True
            
            # Check via schema graph (for indirect connections)
            if self.schema_graph.has_node(("table", table1)) and self.schema_graph.has_node(("table", table2)):
                try:
                    path = nx.shortest_path(self.schema_graph, ("table", table1), ("table", table2))
                    return len(path) <= 3  # Allow up to 2 hops
                except nx.NetworkXNoPath:
                    return False
            
            return False
            
        except Exception as exc:
            LOGGER.error(f"Failed to check table connection: {exc}")
            return False

    def _generate_schema_reports(self) -> None:
        """ADVANCED FEATURE 3: Generate hierarchical schema documentation and community reports"""
        try:
            LOGGER.info("Generating comprehensive schema reports...")
            
            # Generate community-level reports
            community_reports = self._generate_community_reports()
            
            # Generate hierarchical documentation
            hierarchical_docs = self._generate_hierarchical_documentation()
            
            # Generate schema insights
            schema_insights = self._generate_schema_insights()
            
            # Generate relationship analysis
            relationship_analysis = self._generate_relationship_analysis()
            
            # Generate performance recommendations
            performance_recommendations = self._generate_performance_recommendations()
            
            self.community_reports = {
                'community_reports': community_reports,
                'hierarchical_documentation': hierarchical_docs,
                'schema_insights': schema_insights,
                'relationship_analysis': relationship_analysis,
                'performance_recommendations': performance_recommendations,
                'generation_timestamp': str(datetime.now()),
                'schema_statistics': self._calculate_schema_statistics()
            }
            
            LOGGER.info("Schema reports generation completed successfully")
            
        except Exception as exc:
            LOGGER.error(f"Failed to generate schema reports: {exc}")
            self.community_reports = {'error': str(exc)}

    def _generate_community_reports(self) -> Dict[str, Any]:
        """Generate detailed reports for each community"""
        try:
            reports = {}
            
            # Table community reports
            for community_name, tables in self.schema_communities.get('table_communities', {}).items():
                report = {
                    'community_type': 'table_community',
                    'community_name': community_name,
                    'tables': tables,
                    'summary': f"Community focused on {community_name.replace('_', ' ')} operations",
                    'key_relationships': [],
                    'common_patterns': [],
                    'complexity_score': len(tables),
                    'recommendations': []
                }
                
                # Analyze relationships within community
                for table in tables:
                    if table in self.fk_graph:
                        for col, ref_table, ref_col in self.fk_graph[table]:
                            if ref_table in tables:
                                report['key_relationships'].append({
                                    'source': f"{table}.{col}",
                                    'target': f"{ref_table}.{ref_col}",
                                    'type': 'internal_community_fk'
                                })
                
                # Analyze common patterns
                columns_by_type = {}
                for table in tables:
                    if table in self.table_to_columns:
                        for column in self.table_to_columns[table]:
                            column_lower = column.lower()
                            if 'id' in column_lower:
                                columns_by_type.setdefault('identifiers', []).append(f"{table}.{column}")
                            elif 'date' in column_lower or 'time' in column_lower:
                                columns_by_type.setdefault('temporal', []).append(f"{table}.{column}")
                            elif 'status' in column_lower or 'state' in column_lower:
                                columns_by_type.setdefault('status_fields', []).append(f"{table}.{column}")
                
                report['common_patterns'] = columns_by_type
                
                # Generate recommendations
                if len(tables) > 3:
                    report['recommendations'].append("Consider breaking into smaller sub-communities for better maintainability")
                if len(report['key_relationships']) == 0:
                    report['recommendations'].append("No internal relationships found - verify community grouping")
                
                reports[community_name] = report
            
            # Column community reports
            for community_name, columns in self.schema_communities.get('column_communities', {}).items():
                report = {
                    'community_type': 'column_community',
                    'community_name': community_name,
                    'columns': columns,
                    'summary': f"Column community for {community_name.replace('_', ' ')} fields",
                    'usage_patterns': {},
                    'data_types': [],
                    'recommendations': []
                }
                
                # Analyze usage patterns
                tables_using_community = set()
                for column in columns:
                    table_name = column.split('.')[0]
                    tables_using_community.add(table_name)
                
                report['usage_patterns'] = {
                    'tables_using_community': list(tables_using_community),
                    'cross_table_usage': len(tables_using_community) > 1
                }
                
                # Generate recommendations
                if len(tables_using_community) > 3:
                    report['recommendations'].append("High cross-table usage - consider standardization")
                
                reports[f"column_{community_name}"] = report
            
            return reports
            
        except Exception as exc:
            LOGGER.error(f"Failed to generate community reports: {exc}")
            return {}

    def _generate_hierarchical_documentation(self) -> Dict[str, Any]:
        """Generate hierarchical documentation of the schema"""
        try:
            documentation = {
                'schema_overview': {
                    'total_tables': len(self.table_to_columns),
                    'total_columns': sum(len(cols) for cols in self.table_to_columns.values()),
                    'total_relationships': sum(len(fks) for fks in self.fk_graph.values()),
                    'community_count': len(self.schema_communities.get('table_communities', {}))
                },
                'hierarchy_levels': {
                    'level_1_domains': self._get_level_1_domains(),
                    'level_2_functional_groups': self._get_level_2_functional_groups(),
                    'level_3_relationship_clusters': self._get_level_3_relationship_clusters(),
                    'level_4_semantic_patterns': self._get_level_4_semantic_patterns()
                },
                'navigation_guide': self._generate_navigation_guide(),
                'best_practices': self._generate_best_practices()
            }
            
            return documentation
            
        except Exception as exc:
            LOGGER.error(f"Failed to generate hierarchical documentation: {exc}")
            return {}

    def _generate_schema_insights(self) -> Dict[str, Any]:
        """Generate insights about schema structure and patterns"""
        try:
            insights = {
                'complexity_analysis': {
                    'most_complex_tables': self._find_most_complex_tables(),
                    'most_connected_tables': self._find_most_connected_tables(),
                    'isolated_tables': self._find_isolated_tables()
                },
                'naming_patterns': self._analyze_naming_patterns(),
                'data_model_patterns': self._analyze_data_model_patterns(),
                'potential_issues': self._identify_potential_issues(),
                'optimization_opportunities': self._identify_optimization_opportunities()
            }
            
            return insights
            
        except Exception as exc:
            LOGGER.error(f"Failed to generate schema insights: {exc}")
            return {}

    def _generate_relationship_analysis(self) -> Dict[str, Any]:
        """Generate detailed relationship analysis"""
        try:
            analysis = {
                'relationship_types': self._categorize_relationships(),
                'relationship_strength': self._calculate_relationship_strength(),
                'circular_dependencies': self._detect_circular_dependencies(),
                'orphaned_tables': self._find_orphaned_tables(),
                'relationship_recommendations': self._generate_relationship_recommendations()
            }
            
            return analysis
            
        except Exception as exc:
            LOGGER.error(f"Failed to generate relationship analysis: {exc}")
            return {}

    def _generate_performance_recommendations(self) -> Dict[str, Any]:
        """Generate performance optimization recommendations"""
        try:
            recommendations = {
                'indexing_recommendations': self._generate_indexing_recommendations(),
                'query_optimization': self._generate_query_optimization_tips(),
                'schema_optimization': self._generate_schema_optimization_tips(),
                'scalability_considerations': self._generate_scalability_considerations()
            }
            
            return recommendations
            
        except Exception as exc:
            LOGGER.error(f"Failed to generate performance recommendations: {exc}")
            return {}

    def _calculate_schema_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive schema statistics"""
        try:
            stats = {
                'basic_stats': {
                    'total_tables': len(self.table_to_columns),
                    'total_columns': sum(len(cols) for cols in self.table_to_columns.values()),
                    'avg_columns_per_table': sum(len(cols) for cols in self.table_to_columns.values()) / len(self.table_to_columns) if self.table_to_columns else 0,
                    'total_foreign_keys': sum(len(fks) for fks in self.fk_graph.values())
                },
                'community_stats': {
                    'total_communities': len(self.schema_communities.get('table_communities', {})),
                    'avg_tables_per_community': sum(len(tables) for tables in self.schema_communities.get('table_communities', {}).values()) / len(self.schema_communities.get('table_communities', {})) if self.schema_communities.get('table_communities') else 0,
                    'largest_community': max(len(tables) for tables in self.schema_communities.get('table_communities', {}).values()) if self.schema_communities.get('table_communities') else 0
                },
                'complexity_metrics': {
                    'schema_density': len(self.fk_graph) / len(self.table_to_columns) if self.table_to_columns else 0,
                    'avg_relationships_per_table': sum(len(fks) for fks in self.fk_graph.values()) / len(self.table_to_columns) if self.table_to_columns else 0
                }
            }
            
            return stats
            
        except Exception as exc:
            LOGGER.error(f"Failed to calculate schema statistics: {exc}")
            return {}

    # Helper methods for report generation
    def _get_level_1_domains(self) -> List[str]:
        """Get top-level domains"""
        return list(self.schema_communities.get('semantic_communities', {}).keys())

    def _get_level_2_functional_groups(self) -> List[str]:
        """Get functional group level"""
        return list(self.schema_communities.get('table_communities', {}).keys())

    def _get_level_3_relationship_clusters(self) -> List[str]:
        """Get relationship cluster level"""
        return list(self.schema_communities.get('fk_communities', {}).keys())

    def _get_level_4_semantic_patterns(self) -> List[str]:
        """Get semantic pattern level"""
        return list(self.schema_communities.get('column_communities', {}).keys())

    def _generate_navigation_guide(self) -> Dict[str, str]:
        """Generate navigation guide for the schema"""
        return {
            'for_beginners': "Start with semantic communities to understand domain structure",
            'for_analysts': "Use table communities for cross-domain analysis",
            'for_developers': "Focus on FK communities for relationship understanding",
            'for_optimization': "Review column communities for indexing strategies"
        }

    def _generate_best_practices(self) -> List[str]:
        """Generate best practices for schema usage"""
        return [
            "Use community-based querying for better performance",
            "Leverage relationship clusters for complex joins",
            "Consider column communities for data standardization",
            "Monitor schema evolution through drift detection",
            "Use hierarchical documentation for navigation"
        ]

    def _find_most_complex_tables(self) -> List[Dict[str, Any]]:
        """Find tables with highest complexity scores"""
        complexity_scores = []
        for table, columns in self.table_to_columns.items():
            score = len(columns) + len(self.fk_graph.get(table, []))
            complexity_scores.append({'table': table, 'complexity_score': score})
        
        return sorted(complexity_scores, key=lambda x: x['complexity_score'], reverse=True)[:5]

    def _find_most_connected_tables(self) -> List[Dict[str, Any]]:
        """Find tables with most relationships"""
        connection_scores = []
        for table, fk_list in self.fk_graph.items():
            connection_scores.append({'table': table, 'connection_count': len(fk_list)})
        
        return sorted(connection_scores, key=lambda x: x['connection_count'], reverse=True)[:5]

    def _find_isolated_tables(self) -> List[str]:
        """Find tables with no relationships"""
        isolated = []
        for table in self.table_to_columns.keys():
            if table not in self.fk_graph and not any(table in fks for fks in self.fk_graph.values()):
                isolated.append(table)
        return isolated

    def _analyze_naming_patterns(self) -> Dict[str, Any]:
        """Analyze naming patterns across the schema"""
        patterns = {
            'table_naming': {},
            'column_naming': {},
            'consistency_score': 0
        }
        
        # Analyze table naming
        for table in self.table_to_columns.keys():
            if '_' in table:
                pattern = 'snake_case'
            elif table[0].isupper():
                pattern = 'PascalCase'
            else:
                pattern = 'camelCase'
            patterns['table_naming'][pattern] = patterns['table_naming'].get(pattern, 0) + 1
        
        return patterns

    def _analyze_data_model_patterns(self) -> Dict[str, Any]:
        """Analyze data modeling patterns"""
        return {
            'normalization_level': 'high' if len(self.fk_graph) > len(self.table_to_columns) * 0.5 else 'medium',
            'relationship_patterns': 'star' if any(len(fks) > 3 for fks in self.fk_graph.values()) else 'linear',
            'naming_consistency': 'consistent' if len(set(len(cols) for cols in self.table_to_columns.values())) < 3 else 'inconsistent'
        }

    def _identify_potential_issues(self) -> List[str]:
        """Identify potential schema issues"""
        issues = []
        
        # Check for tables with no relationships
        isolated = self._find_isolated_tables()
        if isolated:
            issues.append(f"Isolated tables found: {isolated}")
        
        # Check for naming inconsistencies
        table_patterns = self._analyze_naming_patterns()['table_naming']
        if len(table_patterns) > 1:
            issues.append("Inconsistent table naming patterns detected")
        
        return issues

    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify optimization opportunities"""
        opportunities = []
        
        # Check for potential indexing opportunities
        if len(self.schema_communities.get('column_communities', {}).get('identifier_community', [])) > 5:
            opportunities.append("Consider composite indexes for identifier columns")
        
        # Check for relationship optimization
        if len(self.fk_graph) < len(self.table_to_columns) * 0.3:
            opportunities.append("Consider adding more relationships for better data integrity")
        
        return opportunities

    def _categorize_relationships(self) -> Dict[str, int]:
        """Categorize relationship types"""
        categories = {
            'one_to_one': 0,
            'one_to_many': 0,
            'many_to_many': 0
        }
        
        # Simple categorization based on FK patterns
        for table, fk_list in self.fk_graph.items():
            categories['one_to_many'] += len(fk_list)
        
        return categories

    def _calculate_relationship_strength(self) -> Dict[str, float]:
        """Calculate relationship strength metrics"""
        return {
            'avg_relationships_per_table': sum(len(fks) for fks in self.fk_graph.values()) / len(self.table_to_columns) if self.table_to_columns else 0,
            'schema_connectivity': len(self.fk_graph) / len(self.table_to_columns) if self.table_to_columns else 0
        }

    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the schema"""
        # Simple circular dependency detection
        cycles = []
        visited = set()
        rec_stack = set()
        
        def has_cycle(node, path):
            if node in rec_stack:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            if node in self.fk_graph:
                for _, ref_table, _ in self.fk_graph[node]:
                    if has_cycle(ref_table, path.copy()):
                        return True
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        for table in self.table_to_columns.keys():
            if table not in visited:
                has_cycle(table, [])
        
        return cycles

    def _find_orphaned_tables(self) -> List[str]:
        """Find tables that are referenced but don't exist"""
        orphaned = []
        all_tables = set(self.table_to_columns.keys())
        
        for table, fk_list in self.fk_graph.items():
            for _, ref_table, _ in fk_list:
                if ref_table not in all_tables:
                    orphaned.append(ref_table)
        
        return list(set(orphaned))

    def _generate_relationship_recommendations(self) -> List[str]:
        """Generate recommendations for relationship improvements"""
        recommendations = []
        
        isolated = self._find_isolated_tables()
        if isolated:
            recommendations.append(f"Consider adding relationships for isolated tables: {isolated}")
        
        cycles = self._detect_circular_dependencies()
        if cycles:
            recommendations.append("Circular dependencies detected - review schema design")
        
        return recommendations

    def _generate_indexing_recommendations(self) -> List[str]:
        """Generate indexing recommendations"""
        recommendations = []
        
        # Recommend indexes for frequently joined columns
        fk_columns = []
        for table, fk_list in self.fk_graph.items():
            for col, _, _ in fk_list:
                fk_columns.append(f"{table}.{col}")
        
        if fk_columns:
            recommendations.append(f"Consider indexes on foreign key columns: {fk_columns[:5]}")
        
        # Recommend composite indexes for identifier communities
        identifier_columns = self.schema_communities.get('column_communities', {}).get('identifier_community', [])
        if len(identifier_columns) > 3:
            recommendations.append("Consider composite indexes for identifier columns")
        
        return recommendations

    def _generate_query_optimization_tips(self) -> List[str]:
        """Generate query optimization tips"""
        return [
            "Use community-based filtering to reduce query scope",
            "Leverage foreign key relationships for efficient joins",
            "Consider materialized views for complex analytical queries",
            "Use appropriate indexing strategies based on query patterns",
            "Monitor query performance using schema relationship analysis"
        ]

    def _generate_schema_optimization_tips(self) -> List[str]:
        """Generate schema optimization tips"""
        return [
            "Normalize tables to reduce redundancy",
            "Consider denormalization for read-heavy workloads",
            "Use appropriate data types to minimize storage",
            "Implement proper foreign key constraints",
            "Regular schema review and refactoring"
        ]

    def _generate_scalability_considerations(self) -> List[str]:
        """Generate scalability considerations"""
        return [
            "Consider partitioning for large tables",
            "Implement read replicas for analytical workloads",
            "Use connection pooling for high-concurrency scenarios",
            "Monitor and optimize query performance regularly",
            "Consider NoSQL alternatives for unstructured data"
        ]

    def get_advanced_schema_context(self, query: str, search_strategy: str = "hybrid") -> Dict[str, Any]:
        """ADVANCED FEATURE INTEGRATION: Get comprehensive schema context using multiple search strategies"""
        try:
            LOGGER.info(f"Getting advanced schema context for query: {query} using strategy: {search_strategy}")
            
            context = {
                'query': query,
                'search_strategy': search_strategy,
                'timestamp': str(datetime.now()),
                'results': {}
            }
            
            if search_strategy == "global" or search_strategy == "hybrid":
                context['results']['global'] = self._global_schema_search(query)
            
            if search_strategy == "local" or search_strategy == "hybrid":
                context['results']['local'] = self._local_schema_search(query)
            
            if search_strategy == "drift" or search_strategy == "hybrid":
                context['results']['drift'] = self._drift_schema_search(query)
            
            # Combine results for hybrid strategy
            if search_strategy == "hybrid":
                context['results']['combined'] = self._combine_search_results(context['results'])
            
            # Add community insights
            context['community_insights'] = self._get_community_insights_for_query(query)
            
            # Add schema reports summary
            context['schema_summary'] = self._get_schema_summary_for_query(query)
            
            LOGGER.info(f"Advanced schema context generated successfully with {len(context['results'])} search results")
            return context
            
        except Exception as exc:
            LOGGER.error(f"Failed to get advanced schema context: {exc}")
            return {'error': str(exc), 'query': query, 'search_strategy': search_strategy}

    def _combine_search_results(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from multiple search strategies"""
        try:
            combined = {
                'strategy': 'hybrid_combined',
                'relevant_tables': set(),
                'relevant_columns': set(),
                'relationships': [],
                'recommendations': [],
                'insights': [],
                'confidence_score': 0.0
            }
            
            # Combine tables from all strategies
            for strategy, result in search_results.items():
                if isinstance(result, dict):
                    if 'relevant_tables' in result:
                        combined['relevant_tables'].update(result['relevant_tables'])
                    if 'relevant_columns' in result:
                        combined['relevant_columns'].update(result['relevant_columns'])
                    if 'relationships' in result:
                        combined['relationships'].extend(result['relationships'])
                    if 'recommendations' in result:
                        combined['recommendations'].extend(result['recommendations'])
            
            # Convert sets back to lists
            combined['relevant_tables'] = list(combined['relevant_tables'])
            combined['relevant_columns'] = list(combined['relevant_columns'])
            
            # Calculate confidence score based on agreement between strategies
            strategies_with_results = sum(1 for result in search_results.values() if isinstance(result, dict) and not result.get('error'))
            combined['confidence_score'] = strategies_with_results / len(search_results) if search_results else 0.0
            
            # Generate combined insights
            combined['insights'] = [
                f"Found {len(combined['relevant_tables'])} relevant tables across {strategies_with_results} search strategies",
                f"Identified {len(combined['relationships'])} relationships",
                f"Generated {len(combined['recommendations'])} recommendations",
                f"Overall confidence: {combined['confidence_score']:.2f}"
            ]
            
            return combined
            
        except Exception as exc:
            LOGGER.error(f"Failed to combine search results: {exc}")
            return {'error': str(exc)}

    def _get_community_insights_for_query(self, query: str) -> Dict[str, Any]:
        """Get community-specific insights for a query"""
        try:
            insights = {
                'relevant_communities': [],
                'community_recommendations': [],
                'cross_community_analysis': {}
            }
            
            query_lower = query.lower()
            
            # Find relevant communities
            for community_type, communities in self.schema_communities.items():
                for community_name, community_data in communities.items():
                    relevance_score = 0
                    
                    # Score based on table/column mentions
                    if isinstance(community_data, list):
                        for item in community_data:
                            if item.lower() in query_lower:
                                relevance_score += 1
                    
                    if relevance_score > 0:
                        insights['relevant_communities'].append({
                            'community_name': community_name,
                            'community_type': community_type,
                            'relevance_score': relevance_score,
                            'members': community_data
                        })
            
            # Sort by relevance
            insights['relevant_communities'].sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Generate recommendations
            if len(insights['relevant_communities']) > 1:
                insights['community_recommendations'].append("Multi-community query detected - consider using community-specific handlers")
            
            if any(c['relevance_score'] > 2 for c in insights['relevant_communities']):
                insights['community_recommendations'].append("High community relevance - focus on community-specific tables")
            
            return insights
            
        except Exception as exc:
            LOGGER.error(f"Failed to get community insights: {exc}")
            return {'error': str(exc)}

    def _get_schema_summary_for_query(self, query: str) -> Dict[str, Any]:
        """Get schema summary relevant to the query"""
        try:
            summary = {
                'total_schema_size': {
                    'tables': len(self.table_to_columns),
                    'columns': sum(len(cols) for cols in self.table_to_columns.values()),
                    'relationships': sum(len(fks) for fks in self.fk_graph.values())
                },
                'query_relevance': {
                    'mentioned_tables': [],
                    'mentioned_columns': [],
                    'complexity_estimate': 'simple'
                },
                'performance_indicators': {
                    'expected_complexity': 'low',
                    'recommended_approach': 'standard',
                    'optimization_tips': []
                }
            }
            
            query_lower = query.lower()
            
            # Find mentioned tables and columns
            for table in self.table_to_columns.keys():
                if table.lower() in query_lower:
                    summary['query_relevance']['mentioned_tables'].append(table)
            
            for table, columns in self.table_to_columns.items():
                for column in columns:
                    if column.lower() in query_lower:
                        summary['query_relevance']['mentioned_columns'].append(f"{table}.{column}")
            
            # Estimate complexity
            mentioned_count = len(summary['query_relevance']['mentioned_tables'])
            if mentioned_count > 3:
                summary['query_relevance']['complexity_estimate'] = 'complex'
                summary['performance_indicators']['expected_complexity'] = 'high'
                summary['performance_indicators']['recommended_approach'] = 'optimized'
                summary['performance_indicators']['optimization_tips'].append("Consider using community-based filtering")
            elif mentioned_count > 1:
                summary['query_relevance']['complexity_estimate'] = 'medium'
                summary['performance_indicators']['expected_complexity'] = 'medium'
                summary['performance_indicators']['optimization_tips'].append("Monitor join performance")
            
            return summary
            
        except Exception as exc:
            LOGGER.error(f"Failed to get schema summary: {exc}")
            return {'error': str(exc)}

    def _retrieve_schema_subgraph(
        self, user_query: str, max_hops: int = 3
    ) -> Dict[str, Any]:
        """Advanced Graph RAG: Enhanced subgraph retrieval with semantic scoring and intelligent path finding."""
        try:
            q = user_query.lower()
            
            # Enhanced seed node discovery with semantic scoring
            seeds: List[Tuple[Tuple[str, ...], float]] = []  # (node, relevance_score)
            
            for node in self.schema_graph.nodes:
                kind = node[0]
                score = 0.0
                
                if kind == "table":
                    name = node[1].lower()
                    # Exact match
                    if name in q:
                        score += 1.0
                    # Partial match (plural/singular)
                    elif name[:-1] in q or (name + "s") in q:
                        score += 0.8
                    # Semantic similarity for common patterns
                    elif any(pattern in name for pattern in ["employee", "project", "department"] if pattern in q):
                        score += 0.6
                    
                elif kind == "column":
                    _, table_name, col_name = node
                    col_lower = col_name.lower()
                    # Column name matches
                    if col_lower in q:
                        score += 0.9
                    # Common column patterns
                    elif any(pattern in col_lower for pattern in ["name", "id", "date", "status"] if pattern in q):
                        score += 0.5
                    # Table context boost
                    if table_name.lower() in q:
                        score += 0.3
                
                if score > 0.3:  # Threshold for relevance
                    seeds.append((node, score))
            
            # Sort by relevance and take top seeds
            seeds.sort(key=lambda x: x[1], reverse=True)
            seed_nodes = [node for node, score in seeds[:8]]
            
            if not seed_nodes:
                # Fallback: use top table names
                seed_nodes = [n for n in self.schema_graph.nodes if n[0] == "table"][:3]
            
            # Multi-hop BFS expansion with decay
            included = set(seed_nodes)
            frontier = list(seed_nodes)
            hop_scores = {node: 1.0 for node in seed_nodes}
            
            for hop in range(max_hops):
                next_frontier: List[Tuple[str, ...]] = []
                decay_factor = 0.7 ** hop  # Exponential decay
                
                for n in frontier:
                    current_score = hop_scores.get(n, 0.5)
                    
                    for m in self.schema_graph.neighbors(n):
                        if m not in included:
                            # Score based on edge type and semantic relevance
                            edge_data = self.schema_graph.get_edge_data(n, m, {})
                            edge_score = self._score_graph_edge(edge_data, q)
                            
                            propagated_score = current_score * decay_factor * edge_score
                            
                            if propagated_score > 0.1:  # Minimum threshold
                                included.add(m)
                                next_frontier.append(m)
                                hop_scores[m] = propagated_score
                
                frontier = next_frontier
            
            # Collect enhanced results with scores
            tables = []
            table_scores = {}
            for n in included:
                if n[0] == "table":
                    table_name = n[1]
                    tables.append(table_name)
                    table_scores[table_name] = hop_scores.get(n, 0.5)
            
            # Sort tables by relevance
            tables = sorted(tables, key=lambda t: table_scores.get(t, 0), reverse=True)[:10]
            
            # Enhanced column collection with relevance
            columns: Dict[str, List[str]] = {}
            for n in included:
                if n[0] == "column":
                    _, table_name, col_name = n
                    if table_name in tables:
                        columns.setdefault(table_name, []).append(col_name)
            
            # Enhanced FK edges with semantic scores
            fk_edges: List[Dict[str, Any]] = []
            for u, v, data in self.schema_graph.edges(data=True):
                if (u in included and v in included and 
                    data.get("kind") == "fk" and 
                    u[0] == "table" and v[0] == "table"):
                    
                    from_table, to_table = u[1], v[1]
                    if from_table in tables and to_table in tables:
                        via = data.get("via", ("", ""))
                        edge_score = self._score_graph_edge(data, q)
                        
                        fk_edges.append({
                            "from": from_table,
                            "to": to_table,
                            "columns": via,
                            "score": edge_score,
                            "join_hint": f"JOIN `{to_table}` ON `{from_table}`.`{via[0]}` = `{to_table}`.`{via[1]}`"
                        })
            
            # Sort FK edges by relevance
            fk_edges.sort(key=lambda x: x["score"], reverse=True)
            
            # Generate intelligent join paths
            join_paths = self._generate_join_paths(tables[:6], fk_edges, q)
            
            return {
                "tables": tables,
                "columns": columns,
                "fk_edges": fk_edges,
                "table_scores": table_scores,
                "join_paths": join_paths,
                "semantic_context": self._extract_semantic_context(q, tables)
            }
            
        except Exception as exc:
            LOGGER.debug("Enhanced subgraph retrieval failed: %s", exc)
            return {"tables": [], "columns": {}, "fk_edges": [], "table_scores": {}, "join_paths": []}

    def _score_graph_edge(self, edge_data: Dict[str, Any], query: str) -> float:
        """Score the relevance of a graph edge to the query."""
        score = 0.5  # Base score
        
        edge_kind = edge_data.get("kind", "")
        if edge_kind == "fk":
            score += 0.3  # Foreign keys are important for joins
        
        # Boost if edge involves tables mentioned in query
        via = edge_data.get("via", ("", ""))
        if any(col.lower() in query.lower() for col in via if col):
            score += 0.2
            
        return min(score, 1.0)

    def _generate_join_paths(self, tables: List[str], fk_edges: List[Dict], query: str) -> List[Dict[str, Any]]:
        """Generate intelligent join paths for complex queries."""
        if len(tables) < 2:
            return []
        
        join_paths = []
        
        # Generate direct join paths for high-scoring table pairs
        for i, table1 in enumerate(tables[:4]):
            for table2 in tables[i+1:4]:
                # Find direct connection
                direct_edge = None
                for edge in fk_edges:
                    if ((edge["from"] == table1 and edge["to"] == table2) or
                        (edge["from"] == table2 and edge["to"] == table1)):
                        direct_edge = edge
                        break
                
                if direct_edge:
                    join_paths.append({
                        "tables": [table1, table2],
                        "path_type": "direct",
                        "sql": direct_edge["join_hint"],
                        "score": direct_edge["score"],
                        "complexity": 1
                    })
        
        # Generate multi-hop paths for complex queries
        if len(tables) >= 3 and any(word in query.lower() for word in ["with", "and", "related", "across"]):
            # Simple 3-table path
            if len(tables) >= 3:
                path_tables = tables[:3]
                path_sql = self._build_multi_table_join(path_tables, fk_edges)
                if path_sql:
                    join_paths.append({
                        "tables": path_tables,
                        "path_type": "multi_hop",
                        "sql": path_sql,
                        "score": 0.7,
                        "complexity": len(path_tables) - 1
                    })
        
        # Sort by score and complexity
        join_paths.sort(key=lambda x: (x["score"], -x["complexity"]), reverse=True)
        return join_paths[:5]

    def _build_multi_table_join(self, tables: List[str], fk_edges: List[Dict]) -> str:
        """Build multi-table JOIN SQL from FK relationships."""
        if len(tables) < 2:
            return ""
        
        joins = []
        base_table = tables[0]
        
        for target_table in tables[1:]:
            # Find connection to base or already joined tables
            join_found = False
            for edge in fk_edges:
                if ((edge["from"] == base_table and edge["to"] == target_table) or
                    (edge["from"] == target_table and edge["to"] == base_table)):
                    joins.append(edge["join_hint"])
                    join_found = True
                    break
            
            if not join_found:
                # Fallback to simple ID-based join
                joins.append(f"JOIN `{target_table}` ON `{base_table}`.id = `{target_table}`.{base_table}_id")
        
        return " ".join(joins)

    def _extract_semantic_context(self, query: str, tables: List[str]) -> Dict[str, Any]:
        """Extract semantic context from query for enhanced SQL generation."""
        query_lower = query.lower()
        
        context = {
            "intent": "select",
            "aggregation": None,
            "grouping": None,
            "filtering": None,
            "ordering": None,
            "temporal": None
        }
        
        # Detect aggregation patterns
        if any(word in query_lower for word in ["count", "sum", "avg", "average", "total", "max", "min"]):
            context["aggregation"] = "aggregate"
        
        # Detect grouping patterns
        if any(phrase in query_lower for phrase in ["by", "per", "each", "group"]):
            context["grouping"] = "group_by"
        
        # Detect temporal patterns
        if any(word in query_lower for word in ["year", "month", "date", "time", "recent", "last", "since"]):
            context["temporal"] = "time_based"
        
        # Detect comparison patterns
        if any(word in query_lower for word in ["compare", "vs", "versus", "between", "against"]):
            context["intent"] = "comparison"
        
        return context

    def _pre_validate_schema_usage(self, sql: str, context: RetrievedContext) -> List[str]:
        """Pre-validate SQL against schema context to catch common errors early."""
        issues = []
        sql_lower = sql.lower()
        
        # Extract schema information from context
        schema_tables = set()
        schema_columns = {}
        fk_relationships = {}
        
        for text in context.texts:
            if "Table `" in text:
                # Extract table name
                import re
                table_match = re.search(r"Table `(\w+)`", text)
                if table_match:
                    table_name = table_match.group(1)
                    schema_tables.add(table_name)
                    
                    # Extract columns
                    col_matches = re.findall(r"(\w+)\([^)]+\)", text)
                    schema_columns[table_name] = col_matches
                    
                    # Extract FK relationships
                    fk_matches = re.findall(r"(\w+) -> (\w+)\((\w+)\)", text)
                    for from_col, to_table, to_col in fk_matches:
                        fk_relationships[f"{table_name}.{from_col}"] = f"{to_table}.{to_col}"
        
        # Check for common schema violations
        
        # 1. Wrong table names (employee vs workers)
        if "`employee`" in sql and "workers" in schema_tables:
            issues.append("Using wrong table name: `employee` should be `workers`")
        
        # 2. Wrong column references for workers table
        if "`workers`" in sql and "employee_id" in sql:
            if "workers" in schema_columns and "id" in schema_columns["workers"]:
                issues.append("Wrong column reference: workers table uses `id` not `employee_id`")
        
        # 3. Incorrect JOIN patterns
        if "workers" in sql_lower and "employee_projects" in sql_lower:
            if "w.employee_id" in sql or "workers.employee_id" in sql:
                issues.append("Incorrect JOIN: use `workers.id = employee_projects.employee_id`")
        
        # 4. Missing table references in schema
        for table in schema_tables:
            if f"`{table}`" not in sql and table in sql_lower:
                # Table mentioned but not properly quoted
                issues.append(f"Table `{table}` should be properly quoted with backticks")
        
        return issues

    def _compute_alignment_bonus(self, user_query: str, plan: Dict[str, Any], sql: str) -> int:
        """Compute a small alignment bonus for candidates whose SQL matches intent-critical cues.

        Bonuses are intentionally small and additive to existing validation scores.
        """
        try:
            s = sql.lower()
            q = user_query.lower()
            bonus = 0
            # Distribution / group by when question says by/per/each
            if any(k in q for k in [" by ", " per ", " each ", "group by"]) and " group by " in s:
                bonus += 1
            # Exactly-N projects
            m = re.search(r"\b(\d{1,3})\s+active\s+project", q) or re.search(r"\b(\d{1,3})\s+project", q)
            if m:
                n = int(m.group(1))
                if f"having count(distinct ep.project_id) = {n}" in s:
                    bonus += 2
                if any(w in q for w in ["active", "ongoing", "planned"]) and " p.status " in s:
                    bonus += 1
            # Time series comparison should include group by year
            if any(k in q for k in ["over the last", "by year", "per year", "trend"]):
                if " group by " in s and (" year" in s or "year(" in s):
                    bonus += 1
            # Avoid hard-coded ID filters for departments
            if re.search(r"\bdepartment_id\s*=\s*\d+", s):
                bonus -= 1
            return bonus
        except Exception:
            return 0

    def _maybe_build_compare_two_departments_over_years(self, user_query: str) -> Optional[str]:
        """Build time comparison for two departments over last N years or generic years.

        Extracts two department names (e.g., Engineering vs Sales) and produces a year, count_A, count_B table.
        """
        q = user_query.lower()
        if not ("compare" in q and "over" in q and "year" in q and "department" in q):
            # also accept explicit names without the word department
            if not ("compare" in q and "over" in q and "year" in q and any(n in q for n in ["engineering", "sales", "hr", "marketing", "finance"])):
                return None
        # Find two department tokens by name heuristics
        known = ["engineering", "sales", "hr", "marketing", "finance"]
        depts = [w for w in known if w in q]
        if len(depts) < 2:
            return None
        dept_a, dept_b = depts[0], depts[1]
        # Year window
        years_match = re.search(r"last\s+(\d{1,2})\s+years", q)
        year_filter = ""
        if years_match:
            try:
                n = int(years_match.group(1))
                if 1 <= n <= 20:
                    # Use MySQL YEAR(CURDATE()) for relative window
                    year_filter = f"WHERE p.year >= YEAR(CURDATE()) - {n - 1}\n"
            except Exception:
                pass
        if "performance" not in self.allowed_tables or "workers" not in self.allowed_tables or "departments" not in self.allowed_tables:
            return None
        limit_val = self.safe_exec.default_limit if hasattr(self.safe_exec, "default_limit") else 50
        sql = (
            "SELECT p.year,\n"
            f"  SUM(CASE WHEN d.name = '{dept_a.capitalize()}' THEN 1 ELSE 0 END) AS {dept_a}_count,\n"
            f"  SUM(CASE WHEN d.name = '{dept_b.capitalize()}' THEN 1 ELSE 0 END) AS {dept_b}_count\n"
            "FROM `performance` p JOIN `workers` w ON p.worker_id=w.id\n"
            "JOIN `departments` d ON w.department_id=d.id\n"
            f"{year_filter}"
            "GROUP BY p.year\n"
            "ORDER BY p.year\n"
            f"LIMIT {limit_val}"
        )
        return sql

    def _maybe_build_price_analysis(self, user_query: str) -> Optional[str]:
        """Deterministic builder for price-related queries."""
        q = user_query.lower()
        
        # Price queries - must use prices table
        if any(word in q for word in ['price', 'cost', 'ricotta', 'cream', 'oil', 'ingredient']):
            # Price trend analysis
            if any(word in q for word in ['trend', 'over time', 'change', 'analyze', 'analysis']):
                return (
                    "SELECT DATE_FORMAT(date, '%Y-%m') AS month, "
                    "AVG(ricotta) AS avg_ricotta_price, "
                    "AVG(cream) AS avg_cream_price, "
                    "AVG(oil) AS avg_oil_price "
                    "FROM prices "
                    "WHERE date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH) "
                    "GROUP BY DATE_FORMAT(date, '%Y-%m') "
                    "ORDER BY month "
                    "LIMIT 50"
                )
            
            # Current price comparison
            if any(word in q for word in ['compare', 'comparison', 'current', 'latest']):
                return (
                    "SELECT 'ricotta' AS ingredient, ricotta AS price "
                    "FROM prices "
                    "WHERE date = (SELECT MAX(date) FROM prices) "
                    "UNION ALL "
                    "SELECT 'cream' AS ingredient, cream AS price "
                    "FROM prices "
                    "WHERE date = (SELECT MAX(date) FROM prices) "
                    "UNION ALL "
                    "SELECT 'oil' AS ingredient, oil AS price "
                    "FROM prices "
                    "WHERE date = (SELECT MAX(date) FROM prices) "
                    "LIMIT 50"
                )
        
        return None
        
    def _maybe_build_smart_date_filtered_queries(self, user_query: str) -> Optional[str]:
        """Smart deterministic builder for date-filtered queries."""
        q = user_query.lower()
        
        # Production volumes/weights for today/this week/this month/last month
        if ('production volume' in q or 'production volumes' in q or 'production weight' in q or 'production weights' in q):
            if 'today' in q or 'to day' in q:
                # Since dates are integers, get the most recent data - use totalMaterUsage which has actual data
                return "SELECT SUM(totalMaterUsage) AS total_production_today FROM `production_info` WHERE `date` = (SELECT MAX(date) FROM production_info) LIMIT 50"
            elif 'this week' in q:
                # Get last 7 days worth of data (assuming integer dates) - use totalMaterUsage
                return "SELECT SUM(totalMaterUsage) AS total_production_this_week FROM `production_info` WHERE `date` >= (SELECT MAX(date) - 7 FROM production_info) LIMIT 50"
            elif 'this month' in q:
                # Get last 30 days worth of data (assuming integer dates) - use totalMaterUsage
                return "SELECT SUM(totalMaterUsage) AS total_production_this_month FROM `production_info` WHERE `date` >= (SELECT MAX(date) - 30 FROM production_info) LIMIT 50"
            elif ('last month' in q or 'last mounth' in q or 'in last month' in q or 'in last mounth' in q):
                # Since dates are stored as integers, get recent data instead - use totalMaterUsage
                return "SELECT SUM(totalMaterUsage) AS total_production_last_month FROM `production_info` WHERE `date` >= (SELECT MAX(date) - 30 FROM production_info) LIMIT 50"
        
        # Waste for today/this week/this month (including misspelling "mounth" and "packaging waste")
        if ('waste' in q or 'packaging waste' in q) and ('today' in q or 'this week' in q or 'this month' in q or 'this mounth' in q or '2 mounth' in q or '2 month' in q):
            if 'today' in q:
                return "SELECT SUM(value) AS total_waste_today FROM `pack_waste` WHERE DATE(`date`) = CURDATE() LIMIT 50"
            elif 'this week' in q:
                return "SELECT SUM(value) AS total_waste_this_week FROM `pack_waste` WHERE `date` >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) LIMIT 50"
            elif ('this month' in q or 'this mounth' in q or '2 month' in q or '2 mounth' in q):
                return "SELECT SUM(value) AS total_waste_this_month FROM `pack_waste` WHERE MONTH(`date`) = MONTH(CURDATE()) AND YEAR(`date`) = YEAR(CURDATE()) LIMIT 50"
        
        # Hygiene checks for today/recent
        if 'hygiene' in q and ('today' in q or 'recent' in q):
            if 'today' in q:
                return "SELECT personName, beard, nail, handLeg, robe FROM person_hyg WHERE DATE(date) = CURDATE() ORDER BY date DESC LIMIT 50"
            elif 'recent' in q:
                return "SELECT personName, beard, nail, handLeg, robe FROM person_hyg WHERE date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) ORDER BY date DESC LIMIT 50"
        
        # Worker detail queries - show actual worker information
        if 'workers' in q and any(word in q for word in ['details', 'technical', 'brief', 'list', 'show', 'information']):
            return "SELECT firstName, lastName, section FROM workers ORDER BY section, lastName LIMIT 50"
        
        # Hygiene analytical queries - show hygiene compliance data
        if ('hygien' in q or 'hygiene' in q) and any(word in q for word in ['analytical', 'review', 'analysis', 'compliance', 'rates']):
            return "SELECT personName, beard, nail, handLeg, robe, date FROM person_hyg ORDER BY date DESC LIMIT 50"
        
        # Production by bake types for charts (pie chart, bar chart, etc.)
        if ('production' in q and ('bake' in q or 'backe' in q) and ('chart' in q or 'pie' in q or 'bar' in q)) and ('last month' in q or 'last mounth' in q or 'mounth' in q):
            return "SELECT `bakeType`, SUM(`totalMaterUsage`) as production_volume FROM `production_info` WHERE `date` >= (SELECT MAX(`date`) - 30 FROM `production_info`) GROUP BY `bakeType` ORDER BY production_volume DESC LIMIT 50"
        
        return None

    def _maybe_build_smart_waste_analysis(self, user_query: str) -> Optional[str]:
        """Smart waste analysis based on schema patterns"""
        q = user_query.lower()
        
        # Waste by type patterns
        if any(pattern in q for pattern in ['waste by type', 'waste distribution', 'total waste', 'waste types']):
            return "SELECT `type`, SUM(`value`) as total_waste FROM `pack_waste` GROUP BY `type` ORDER BY total_waste DESC LIMIT 50"
        
        # Waste over time patterns
        if any(pattern in q for pattern in ['waste over time', 'waste trends', 'waste by month']):
            return "SELECT `date`, SUM(`value`) as total_waste FROM `pack_waste` GROUP BY `date` ORDER BY `date` DESC LIMIT 50"
        
        return None

    def _maybe_build_smart_production_analysis(self, user_query: str) -> Optional[str]:
        """Smart production analysis based on schema patterns"""
        q = user_query.lower()
        
        # Production by bake type
        if any(pattern in q for pattern in ['production by type', 'production volume', 'bake types', 'production by bake']):
            return "SELECT `bakeType`, SUM(`totalUsage`) as total_production FROM `production_info` GROUP BY `bakeType` ORDER BY total_production DESC LIMIT 50"
        
        # Production efficiency
        if any(pattern in q for pattern in ['production efficiency', 'efficiency analysis', 'production trends']):
            return "SELECT `bakeType`, AVG(`totalUsage`) as avg_production, AVG(`humidity`) as avg_humidity, AVG(`temp`) as avg_temp FROM `production_info` GROUP BY `bakeType` ORDER BY avg_production DESC LIMIT 50"
        
        return None

    def _maybe_build_smart_worker_analysis(self, user_query: str) -> Optional[str]:
        """Smart worker analysis based on schema patterns"""
        q = user_query.lower()
        
        # Workers by section
        if any(pattern in q for pattern in ['workers by section', 'employees by section', 'staff by section', 'section distribution']):
            return "SELECT `section`, COUNT(*) as worker_count FROM `workers` GROUP BY `section` ORDER BY worker_count DESC LIMIT 50"
        
        # List all workers
        if any(pattern in q for pattern in ['list workers', 'show workers', 'all workers', 'worker list']):
            return "SELECT `firstName`, `lastName`, `section` FROM `workers` ORDER BY `section`, `lastName` LIMIT 50"
        
        return None

    def _maybe_build_smart_hygiene_analysis(self, user_query: str) -> Optional[str]:
        """Smart hygiene analysis based on schema patterns"""
        q = user_query.lower()
        
        # Hygiene violations
        if any(pattern in q for pattern in ['hygiene violations', 'hygiene issues', 'hygiene problems', 'failed hygiene']):
            return "SELECT `personName`, COUNT(*) as violations FROM `person_hyg` WHERE `beard` = 'fail' OR `nail` = 'fail' OR `handLeg` = 'fail' OR `robe` = 'fail' GROUP BY `personName` ORDER BY violations DESC LIMIT 50"
        
        # Hygiene compliance rates (analytical pattern)
        if any(pattern in q for pattern in ['hygiene compliance', 'compliance rates', 'hygiene rates']):
            return "SELECT `personName`, COUNT(*) as total_checks, SUM(CASE WHEN `beard` = 'pass' AND `nail` = 'pass' AND `handLeg` = 'pass' AND `robe` = 'pass' THEN 1 ELSE 0 END) as passed_checks, ROUND((SUM(CASE WHEN `beard` = 'pass' AND `nail` = 'pass' AND `handLeg` = 'pass' AND `robe` = 'pass' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as compliance_rate FROM `person_hyg` GROUP BY `personName` ORDER BY compliance_rate DESC LIMIT 50"
        
        # Hygiene check results for today/recent
        if any(pattern in q for pattern in ['hygiene check results', 'hygiene results', 'check results']):
            return "SELECT `personName`, `beard`, `nail`, `handLeg`, `robe`, `apron`, `gloves`, `date` FROM `person_hyg` ORDER BY `date` DESC LIMIT 50"
        
        return None

    def _maybe_build_complex_multi_table_analysis(self, user_query: str) -> Optional[str]:
        """Handle complex queries requiring multiple tables or advanced reasoning"""
        q = user_query.lower()
        
        # Production volume correlation with quality
        if 'correlation' in q and 'production' in q and 'quality' in q:
            return "SELECT pi.bakeType, AVG(pi.totalUsage) as avg_production, COUNT(pt.bakeID) as test_count FROM `production_info` pi LEFT JOIN `production_test` pt ON pi.bakeID = pt.bakeID GROUP BY pi.bakeType ORDER BY avg_production DESC LIMIT 50"
        
        # Most expensive ingredients with price changes
        if 'expensive ingredients' in q and ('change' in q or 'time' in q):
            return "SELECT 'ricotta' as ingredient, MAX(ricotta) as max_price, MIN(ricotta) as min_price, (MAX(ricotta) - MIN(ricotta)) as price_change FROM `prices` UNION SELECT 'cream', MAX(cream), MIN(cream), (MAX(cream) - MIN(cream)) FROM `prices` UNION SELECT 'oil', MAX(oil), MIN(oil), (MAX(oil) - MIN(oil)) FROM `prices` ORDER BY max_price DESC LIMIT 50"
        
        # Production batches from specific time period - ENHANCED
        if ('production batches' in q or 'all production' in q) and ('last week' in q or 'recent' in q or 'from last' in q):
            return "SELECT `bakeID`, `date`, `bakeType`, `totalUsage`, `humidity`, `temp` FROM `production_info` ORDER BY `date` DESC LIMIT 50"
        
        # Packaging information for recent batches - ENHANCED
        if ('packaging information' in q and 'recent' in q) or ('packaging' in q and 'batches' in q) or ('packaging information' in q):
            return "SELECT `bakeType`, `TotalWeight`, `tranWeight`, `date`, `bakeID` FROM `packaging_info` ORDER BY `date` DESC LIMIT 50"
        
        # Top expensive ingredients - should return multiple ingredients
        if 'top' in q and 'expensive' in q and 'ingredients' in q:
            return "SELECT 'ricotta' as ingredient, MAX(ricotta) as price FROM `prices` UNION SELECT 'cream', MAX(cream) FROM `prices` UNION SELECT 'oil', MAX(oil) FROM `prices` UNION SELECT 'buttermilkPowder', MAX(buttermilkPowder) FROM `prices` ORDER BY price DESC LIMIT 5"
        
        # Waste generation trends with time series
        if 'waste generation trends' in q or ('waste' in q and 'trends' in q):
            return "SELECT DATE_FORMAT(`date`, '%Y-%m') as month, `type`, SUM(`value`) as total_waste FROM `pack_waste` WHERE `date` >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH) GROUP BY month, `type` ORDER BY month, total_waste DESC LIMIT 50"
        
        # Production efficiency trends over time
        if 'production efficiency trends' in q or ('efficiency' in q and 'over time' in q):
            return "SELECT DATE_FORMAT(`date`, '%Y-%m') as month, `bakeType`, AVG(`totalUsage`) as avg_production, AVG(`humidity`) as avg_humidity, AVG(`temp`) as avg_temp FROM `production_info` WHERE `date` >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH) GROUP BY month, `bakeType` ORDER BY month, avg_production DESC LIMIT 50"
        
        return None

    def _maybe_build_simple_aggregation(self, user_query: str) -> Optional[str]:
        """Build simple aggregation queries for SHORT_ANSWER mode.
        
        Handles queries like:
        - "What is the average humidity?"
        - "What is the total waste generated?"
        - "How many workers are there?"
        """
        q = user_query.lower()
        
        # Simple count queries
        if any(pattern in q for pattern in ['how many', 'count', 'number of']):
            if 'worker' in q:
                return "SELECT COUNT(*) FROM `workers` LIMIT 50"
            elif 'waste' in q:
                return "SELECT COUNT(*) FROM `pack_waste` LIMIT 50"
            elif 'production' in q:
                return "SELECT COUNT(*) FROM `production_info` LIMIT 50"
            elif 'packaging' in q:
                return "SELECT COUNT(*) FROM `packaging_info` LIMIT 50"
        
        # Average queries
        if any(pattern in q for pattern in ['average', 'avg', 'mean']):
            if 'humidity' in q:
                return "SELECT AVG(humidity) FROM `production_info` LIMIT 50"
            elif 'temperature' in q:
                return "SELECT AVG(temp) FROM `production_info` LIMIT 50"
            elif 'usage' in q and 'ricotta' in q:
                return "SELECT AVG(ricotta) FROM `production_info` LIMIT 50"
        
        # Sum queries
        if any(pattern in q for pattern in ['total', 'sum']):
            if 'waste' in q and 'total' in q:
                return "SELECT SUM(value) FROM `pack_waste` LIMIT 50"
            elif 'production' in q and 'volume' in q:
                return "SELECT SUM(totalUsage) FROM `production_info` LIMIT 50"
        
        # Max/Min queries
        if any(pattern in q for pattern in ['maximum', 'max', 'highest']):
            if 'weight' in q:
                return "SELECT MAX(tranProdWeight) FROM `production_info` LIMIT 50"
            elif 'humidity' in q:
                return "SELECT MAX(humidity) FROM `production_info` LIMIT 50"
        
        if any(pattern in q for pattern in ['minimum', 'min', 'lowest']):
            if 'weight' in q:
                return "SELECT MIN(tranProdWeight) FROM `production_info` LIMIT 50"
            elif 'humidity' in q:
                return "SELECT MIN(humidity) FROM `production_info` LIMIT 50"
        
        return None

    def _maybe_build_table_listing(self, user_query: str) -> Optional[str]:
        """Build simple table listing queries for TABLE mode.
        
        Handles queries like:
        - "List all unique bake types"
        - "Show me recent batches"
        - "List workers by section"
        """
        q = user_query.lower()
        
        # Unique/distinct value queries
        if any(pattern in q for pattern in ['unique', 'distinct', 'all unique']):
            if 'bake type' in q:
                return "SELECT DISTINCT bakeType FROM `packaging_info` ORDER BY bakeType LIMIT 50"
            elif 'waste type' in q:
                return "SELECT DISTINCT type FROM `pack_waste` ORDER BY type LIMIT 50"
        
        # Recent/latest queries
        if any(pattern in q for pattern in ['recent', 'latest', 'newest']):
            if 'batch' in q or 'production' in q:
                return "SELECT * FROM `production_info` ORDER BY date DESC LIMIT 50"
            elif 'packaging' in q:
                return "SELECT * FROM `packaging_info` ORDER BY date DESC LIMIT 50"
        
        # Grouping queries
        if 'by' in q:
            if 'worker' in q and 'section' in q:
                return "SELECT section, COUNT(*) as count FROM `workers` GROUP BY section ORDER BY count DESC LIMIT 50"
            elif 'bake type' in q:
                return "SELECT bakeType, COUNT(*) as count FROM `packaging_info` GROUP BY bakeType ORDER BY count DESC LIMIT 50"
            elif 'waste type' in q:
                return "SELECT type, COUNT(*) as count FROM `pack_waste` GROUP BY type ORDER BY count DESC LIMIT 50"
        
        return None

    def _is_deterministic_builder_sql(self, sql: str) -> bool:
        """Check if SQL was generated by a deterministic builder."""
        # Simple heuristic: deterministic builders tend to have specific patterns
        # and don't contain complex LLM-generated structures
        sql_lower = sql.lower()
        
        # Common deterministic patterns
        deterministic_patterns = [
            "select distinct",
            "select count(*)",
            "select avg(",
            "select sum(",
            "select max(",
            "select min(",
            "from `workers`",
            "from `packaging_info`",
            "from `pack_waste`",
            "from `production_info`",
            "from `prices`"
        ]
        
        # If it matches common deterministic patterns and is relatively simple
        if any(pattern in sql_lower for pattern in deterministic_patterns):
            # Additional check: not too complex (no complex joins, subqueries, etc.)
            complexity_indicators = [
                "join",
                "union",
                "subquery",
                "case when",
                "with ",
                "window",
                "over ("
            ]
            if not any(indicator in sql_lower for indicator in complexity_indicators):
                return True
                
        return False

    def _detect_mode(
        self, user_query: str, context: RetrievedContext, prefer_mode: Optional[str]
    ) -> str:
        if prefer_mode:
            return prefer_mode
            
        # Check cache first for performance (but skip for natural language queries to ensure accuracy)
        cache_key = user_query.lower().strip()
        # Skip cache for natural language patterns to ensure fresh detection
        natural_language_patterns = ['chart', 'plot', 'for today', 'for this week', 'for this month', 'show me']
        if not any(pattern in cache_key for pattern in natural_language_patterns) and cache_key in self._mode_cache:
            return self._mode_cache[cache_key]
            
        # Rule-based heuristics for faster and more accurate mode detection
        query_lower = user_query.lower()
        
        # ENHANCED MODE DETECTION with better pattern matching
        
        # VISUALIZATION patterns - check FIRST (highest priority for explicit requests)
        # Check for explicit chart requests BEFORE other patterns - ENHANCED
        explicit_chart_patterns = [
            'with a line chart', 'with line chart', 'line chart',
            'with a bar chart', 'with bar chart', 'bar chart', 
            'with a pie chart', 'with pie chart', 'pie chart',
            'with a histogram', 'with histogram', 'histogram',
            'show as bar chart', 'show as pie chart', 'show as line chart', 'show as histogram',
            'with a scatter plot', 'scatter plot'
        ]
        if any(pattern in query_lower for pattern in explicit_chart_patterns):
            detected_mode = "VISUALIZATION"
            self._mode_cache[cache_key] = detected_mode
            return detected_mode
        
        # General visualization patterns
        viz_patterns = [
            'pie chart', 'bar chart', 'line chart', 'scatter plot', 'histogram',
            'plot', 'chart', 'graph', 'visualize', 'visualization',
            'heatmap', 'dashboard'
        ]
        if any(pattern in query_lower for pattern in viz_patterns):
            detected_mode = "VISUALIZATION"
            self._mode_cache[cache_key] = detected_mode
            return detected_mode
        
        # SHORT_ANSWER patterns - check SECOND (scalar/single value queries)
        short_patterns = [
            # Count queries
            'how many', 'count of', 'total number', 'total count', 'number of',
            # Simple aggregation queries that expect single values
            'what is the total', 'what is the sum', 'what is the average', 
            'what is the max', 'what is the min', 'what is the highest', 'what is the lowest',
            # Direct scalar requests
            'total waste', 'total production', 'total volume', 'total weight',
            'average humidity', 'average temperature', 'average usage',
            'maximum production', 'minimum production', 'highest value', 'lowest value',
            # Natural language patterns for scalar requests
            'show me production volumes for', 'production volumes for today', 'production volumes for this',
            'show me total waste for', 'total waste for today', 'total waste for this',
            'show me average usage for', 'average usage for today', 'average usage for this'
        ]
        if any(pattern in query_lower for pattern in short_patterns):
            detected_mode = "SHORT_ANSWER"
            self._mode_cache[cache_key] = detected_mode
            return detected_mode
        
        # ANALYTICAL patterns - check THIRD (analysis and insights)
        analytical_patterns = [
            'analyze', 'analysis', 'insights', 'pattern', 'patterns',
            'efficiency', 'performance', 'optimization', 'improvement',
            'correlation', 'relationship', 'impact', 'effect', 'influence',
            'trend over time', 'change over time', 'evolution over',
            'compare', 'comparison', 'between different', 'compliance rates'
        ]
        if any(pattern in query_lower for pattern in analytical_patterns):
            detected_mode = "ANALYTICAL"
            self._mode_cache[cache_key] = detected_mode
            return detected_mode
        
        # SPECIAL CASES for specific query patterns
        
        # Production volume queries - should be SHORT_ANSWER when asking for totals/scalars
        if ('production volume' in query_lower or 'production volumes' in query_lower) and any(scalar_word in query_lower for scalar_word in ['this month', 'for this', 'total', 'sum', 'for today', 'today']):
            detected_mode = "SHORT_ANSWER"
            self._mode_cache[cache_key] = detected_mode
            return detected_mode
            
        # Price change queries should be ANALYTICAL
        if 'price' in query_lower and any(analytical_word in query_lower for analytical_word in ['changed', 'change', 'over time', 'trends']):
            detected_mode = "ANALYTICAL"
            self._mode_cache[cache_key] = detected_mode
            return detected_mode
        
        # TABLE patterns - check LAST (detailed listings and multi-row results)
        table_patterns = [
            # Direct listing requests
            'show me', 'list', 'display', 'get me', 'find', 'give me',
            # Multi-row requests
            'all ', 'every', 'which', 'what are', 'what are the',
            # Ranking and comparison (multi-row)
            'top ', 'most', 'least', 'best', 'worst', 
            # Grouping and categorization
            'by type', 'by section', 'by month', 'by person',
            # Recent and time-based listings
            'recent', 'latest', 'newest', 'last week', 'this week',
            # Specific data requests
            'batches', 'results', 'information', 'data', 'records',
            # List-style queries
            'unique', 'distinct', 'all unique', 'all distinct',
            # Specific domain patterns
            'bake types', 'packaging types', 'waste types'
        ]
        if any(pattern in query_lower for pattern in table_patterns):
            detected_mode = "TABLE"
            self._mode_cache[cache_key] = detected_mode
            return detected_mode
        
        # Enhanced fallback heuristics with better prioritization
        lower = user_query.lower()
        
        # 1. Explicit visuals first (highest priority)
        vis_keywords = ["plot", "chart", "graph", "trend", "visualize", "visualisation", "visualization", "pie"]
        if any(w in lower for w in vis_keywords):
            return "VISUALIZATION"
            
        # 2. Scalar questions (high priority)
        scalar_keywords = [
            "how many", "count", "total number", "avg", "average", "sum", "max", "min",
            "what is the total", "what is the average", "what is the maximum", "what is the minimum"
        ]
        if any(w in lower for w in scalar_keywords):
            return "SHORT_ANSWER"
            
        # 3. List/display requests (high priority for TABLE)
        list_keywords = [
            "list", "show", "display", "get", "find", "all", "unique", "distinct",
            "types", "categories", "batches", "recent", "latest"
        ]
        if any(w in lower for w in list_keywords):
            return "TABLE"
            
        # 4. Analytical comparisons/explanations (medium priority)
        analytical_keywords = ["analyze", "analysis", "insights", "why", "explain", "compare", "versus", "vs ", "trends"]
        if any(w in lower for w in analytical_keywords):
            return "ANALYTICAL"
            
        # 5. Distribution/grouping queries (default to TABLE)
        if any(k in lower for k in [" by ", " per ", " each ", "group by", "count by", "distribution"]):
            return "TABLE"
            
        # 6. Default fallback
        return "TABLE"

    def _generate_multi_table_sql(self, user_query: str) -> Optional[str]:
        """Generate SQL directly for known multi-table patterns."""
        query_lower = user_query.lower()
        
        # Workers with production data
        if "workers" in query_lower and "production" in query_lower:
            return """SELECT w.firstName, w.lastName, w.section, p.bakeType, p.totalUsage 
FROM workers w 
JOIN production_info p ON w.section = p.bakeType 
ORDER BY w.section, w.lastName 
LIMIT 50"""
        
        # Packaging with transaction details
        if "packaging" in query_lower and "transaction" in query_lower:
            return """SELECT pi.bakeType, pi.totalWeight, pi.tranWeight, t.tranStatus, t.tranFree 
FROM packaging_info pi 
JOIN transtatus t ON pi.tranNumber = t.tranNumber 
ORDER BY pi.bakeType 
LIMIT 50"""
        
        # Workers with hygiene violations
        if "workers" in query_lower and "hygiene" in query_lower:
            return """SELECT w.firstName, w.lastName, w.section, h.personName, h.beard, h.nail, h.handLeg 
FROM workers w 
JOIN person_hyg h ON w.name = h.personName 
WHERE h.beard = 'fail' OR h.nail = 'fail' OR h.handLeg = 'fail'
ORDER BY w.section, w.lastName 
LIMIT 50"""
        
        # Production with quality data
        if "production" in query_lower and "quality" in query_lower:
            return """SELECT p.bakeType, p.totalUsage, pt.totalUsage as testUsage, 
(p.totalUsage - pt.totalUsage) as variance
FROM production_info p 
JOIN production_test pt ON p.bakeType = pt.bakeType 
ORDER BY p.bakeType 
LIMIT 50"""
        
        # Packaging with waste analysis
        if "packaging" in query_lower and "waste" in query_lower:
            return """SELECT pi.bakeType, pi.totalWeight, pw.type, pw.value 
FROM packaging_info pi 
JOIN pack_waste pw ON pi.date = pw.date 
ORDER BY pi.bakeType, pw.value DESC 
LIMIT 50"""
        
        # Production with costs
        if "production" in query_lower and ("cost" in query_lower or "price" in query_lower):
            return """SELECT p.bakeType, p.totalUsage, pr.ricotta, pr.cream, pr.oil 
FROM production_info p 
JOIN prices pr ON p.date = pr.date 
ORDER BY p.bakeType 
LIMIT 50"""
        
        return None

    def _generate_sql(
        self, user_query: str, context: RetrievedContext, mode: str
    ) -> Optional[str]:
        # Enhanced prompt with context and mode-specific guidance
        ctx_snippets = "\n\n".join(context.texts[:3])  # Use relevant context
        
        # Check if this is a multi-table query and generate direct SQL
        multi_table_sql = self._generate_multi_table_sql(user_query)
        if multi_table_sql:
            LOGGER.info(f"Generated multi-table SQL directly: {multi_table_sql}")
            return multi_table_sql
        
        # Check if this is a multi-table query
        multi_table_hint = ""
        if any(phrase in user_query.lower() for phrase in ["with their", "with", "and their", "corresponding"]):
            multi_table_hint = """
MULTI-TABLE QUERY DETECTED! Use JOINs:
- "workers with production" → JOIN workers + production_info ON workers.section = production_info.bakeType
- "packaging with transactions" → JOIN packaging_info + transtatus ON packaging_info.tranNumber = transtatus.tranNumber
- "workers with hygiene" → JOIN workers + person_hyg ON workers.name = person_hyg.personName
"""
        
        prompt = f"""You are an expert SQL generator for a food production database (Farnan).

QUERY: {user_query}
MODE: {mode}
{multi_table_hint}

ACTUAL DATABASE SCHEMA:
- pack_waste: date, type, value (waste tracking by type and amount)
- production_info: bakeType, totalUsage, ricotta, cream, oil, humidity, temp (main production data)
- packaging_info: bakeType, TotalWeight, tranWeight (packaging specifications)
- person_hyg: personName, beard, nail, handLeg, robe (hygiene compliance checks)
- prices: ricotta, cream, oil, buttermilkPowder (ingredient pricing)
- workers: firstName, lastName, section (employee information)
- production_test: bakeType, totalUsage (production test data)
- repo_nc: cheeseType, delivery, returns, total, usage (cheese repository)

INTELLIGENT PATTERNS:
- "waste by type" → SELECT type, SUM(value) FROM pack_waste GROUP BY type
- "production by bake type" → SELECT bakeType, SUM(totalUsage) FROM production_info GROUP BY bakeType
- "workers by section" → SELECT section, COUNT(*) FROM workers GROUP BY section
- "hygiene violations" → SELECT personName, COUNT(*) FROM person_hyg WHERE beard='fail' OR nail='fail' GROUP BY personName
- "ingredient prices" → SELECT ricotta, cream, oil FROM prices ORDER BY date DESC

MODE RULES:
- SHORT_ANSWER: Single scalar (COUNT, SUM, AVG, MAX, MIN) for "how many", "total", "average"
- TABLE: Multiple rows for "show", "list", "display", "by type/section"
- ANALYTICAL: Trends/patterns for "analyze", "compare", "trends", "over time"
- VISUALIZATION: Chart data for "pie chart", "bar chart", "histogram", "plot"

SCHEMA CONTEXT:
{ctx_snippets}

ENHANCED TABLE SELECTION:
- Price/Cost/Expensive/Cheap queries → 'prices' table ONLY (ricotta, cream, oil columns)
- Production/Volume/Batch/Bake queries → 'production_info' table (totalUsage, bakeType columns)
- Waste/Disposal queries → 'pack_waste' table (type, value columns)
- Hygiene/Compliance/Violation queries → 'person_hyg' table (personName, beard, nail columns)
- Worker/Employee/Staff/Section queries → 'workers' table (firstName, section columns)
- Packaging/Package queries → 'packaging_info' table (bakeType, TotalWeight columns)
- Test/Quality queries → 'production_test' table (bakeType, totalUsage columns)

DOMAIN-SPECIFIC MAPPINGS:
- "production volumes" → production_info.totalUsage
- "packaging waste" → pack_waste table (NOT packaging_info)
- "hygiene check results" → person_hyg table
- "recent batches" → production_info table
- "packaging information" → packaging_info table

CRITICAL TABLE RULES:
- NEVER use 'production_info' for price/cost queries
- NEVER use 'packaging_info' for waste queries
- ALWAYS use 'person_hyg' for hygiene queries

EXAMPLES:
- 'packs data for this month' → SELECT * FROM `packs` WHERE MONTH(`date`) = MONTH(CURDATE()) LIMIT 50
- 'pack_waste data for this month' → SELECT * FROM `pack_waste` WHERE MONTH(`date`) = MONTH(CURDATE()) LIMIT 50
- 'Waste distribution by type' → SELECT `type`, COUNT(*) FROM `pack_waste` GROUP BY `type` LIMIT 50
- 'prices data for this month' → SELECT * FROM `prices` WHERE MONTH(`date`) = MONTH(CURDATE()) LIMIT 50
- 'Production volumes this month' → SELECT SUM(totalUsage) FROM `production_info` WHERE MONTH(date) = MONTH(CURDATE()) LIMIT 50
- 'repo_nc count' → SELECT COUNT(*) FROM `repo_nc` LIMIT 50
- 'users count' → SELECT COUNT(*) FROM `users` LIMIT 50
- 'workers count' → SELECT COUNT(*) FROM `workers` LIMIT 50
- 'How many workers?' → SELECT COUNT(*) FROM `workers` LIMIT 50

RULES:RULES:
- For WORKER/EMPLOYEE queries → use 'workers' table
- For PRODUCTION data → use 'production_info' table
- For PACKAGING data → use 'packaging_info' table  
- For WASTE data → use 'pack_waste' table (NOT packaging_info)
- For HYGIENE data → use 'person_hyg' table
- For PRICE data → use 'prices' table
- For TEST data → use 'production_test' table

EXAMPLES:
- "How many workers?" → SELECT COUNT(*) FROM `workers` LIMIT 50
- "Show production volumes" → SELECT SUM(totalUsage) FROM `production_info` LIMIT 50
- "Waste distribution by type" → SELECT `type`, COUNT(*) FROM `pack_waste` GROUP BY `type` LIMIT 50
- "Packaging types with pie chart" → SELECT `bakeType`, COUNT(*) FROM `packaging_info` GROUP BY `bakeType` LIMIT 50

RULES:
- Use ONLY the provided schema context
- Always add LIMIT 50
- Use backticks around table/column names
- For SHORT_ANSWER: return single value
- For VISUALIZATION: return 2+ columns (label, value)
- Output ONLY SQL in ```sql``` fences"""
        
        try:
            resp = self.llm.invoke(prompt)
            raw = resp.content if hasattr(resp, "content") else resp
            txt: str = _stringify_llm_content(raw)
            sql = _extract_sql_from_text(txt)
            
            # Apply fixes to the generated SQL
            if sql:
                sql = self._apply_sql_fixes(sql, user_query, context)
                LOGGER.info(f"Applied fixes to SQL: {sql}")
            
            return sql
        except Exception as exc:
            LOGGER.error("SQL generation failed: %s", exc)
            return None

    def _apply_sql_fixes(self, sql: str, user_query: str, context: RetrievedContext) -> str:
        """Apply all fixes to the generated SQL."""
        try:
            query_lower = user_query.lower()
            sql_lower = sql.lower()
            
            # Fix 1: Replace ALL production volume fields with TotalWeight for production queries
            production_terms = ['production volume', 'total usage', 'production data', 'average production', 'production for today', 'production volumes']
            if any(term in query_lower for term in production_terms):
                # Handle multiple field variations
                field_replacements = [
                    ('totalusage', 'TotalWeight'),
                    ('totalmaterusage', 'TotalWeight'), 
                    ('total_usage', 'TotalWeight'),
                    ('total_mater_usage', 'TotalWeight'),
                    ('totalUsage', 'TotalWeight'),
                    ('totalMaterUsage', 'TotalWeight'),
                    ('total_Usage', 'TotalWeight'),
                    ('total_Mater_Usage', 'TotalWeight')
                ]
                
                for old_field, new_field in field_replacements:
                    if old_field in sql_lower:
                        sql = sql.replace(old_field, new_field)
                        sql = sql.replace(old_field.upper(), new_field)
                        sql = sql.replace(old_field.capitalize(), new_field)
                        LOGGER.info(f"Fixed: Replaced {old_field} with {new_field}")
                
                # Replace table references
                sql = sql.replace('production_info', 'packaging_info')
                sql = sql.replace('`production_info`', '`packaging_info`')
                LOGGER.info("Fixed: Replaced production_info table with packaging_info")
            
            # Fix 2: Handle "today" queries - replace MAX(date) with proper date filtering
            if any(term in query_lower for term in ['today', 'for today', 'production for today']):
                # Replace MAX(date) patterns with proper date filtering
                if 'max(date)' in sql_lower or 'MAX(date)' in sql:
                    # Get the latest date from database and use it as "today"
                    persian_today = "14040623"  # Latest date in your database
                    sql = sql.replace('MAX(date)', f"'{persian_today}'")
                    sql = sql.replace('max(date)', f"'{persian_today}'")
                    LOGGER.info("Fixed: Replaced MAX(date) with specific date for 'today' queries")
            
            # Fix 3: Add date filtering for temporal queries
            temporal_keywords = ['recently', 'ago', 'over time', 'trends', 'last week', 'last month', '1 month ago']
            if any(keyword in query_lower for keyword in temporal_keywords):
                if 'WHERE' not in sql.upper() or not any(word in sql.upper() for word in ['DATE_SUB', 'CURDATE', 'INTERVAL', '1403', '1404']):
                    # Add Persian date filtering
                    persian_date_filter = "WHERE date >= '14030000' AND date <= '14050000'"
                    
                    if 'WHERE' in sql.upper():
                        # Add to existing WHERE clause
                        sql = sql.replace('WHERE', f"{persian_date_filter} AND")
                    else:
                        # Add new WHERE clause
                        if 'GROUP BY' in sql.upper():
                            sql = sql.replace('GROUP BY', f"{persian_date_filter} GROUP BY")
                        elif 'ORDER BY' in sql.upper():
                            sql = sql.replace('ORDER BY', f"{persian_date_filter} ORDER BY")
                        elif 'LIMIT' in sql.upper():
                            sql = sql.replace('LIMIT', f"{persian_date_filter} LIMIT")
                        else:
                            sql += f" {persian_date_filter}"
                    
                    LOGGER.info("Fixed: Added Persian date filtering for temporal queries")
            
                # Fix 4: Fix hygiene queries to properly filter for failures
                if any(term in query_lower for term in ['hygiene', 'failed', 'violations', 'compliance', 'violation']):
                    # If using wrong table, fix it
                    if 'workers' in sql_lower and 'person_hyg' not in sql_lower:
                        sql = sql.replace('workers', 'person_hyg')
                        sql = sql.replace('`workers`', '`person_hyg`')
                        LOGGER.info("Fixed: Changed workers table to person_hyg for hygiene queries")
                    
                    # If it's a simple count, make it a proper hygiene violations query
                    if 'count(*)' in sql_lower and 'person_hyg' in sql_lower:
                        sql = """SELECT personName, 
                                (CASE WHEN beard = 'fail' THEN 1 ELSE 0 END +
                                 CASE WHEN nail = 'fail' THEN 1 ELSE 0 END +
                                 CASE WHEN handLeg = 'fail' THEN 1 ELSE 0 END +
                                 CASE WHEN robe = 'fail' THEN 1 ELSE 0 END) as violations
                                FROM person_hyg 
                                WHERE beard = 'fail' OR nail = 'fail' OR handLeg = 'fail' OR robe = 'fail'
                                GROUP BY personName 
                                ORDER BY violations DESC 
                                LIMIT 1"""
                        LOGGER.info("Fixed: Converted simple count to proper hygiene violations query")
                    elif 'person_hyg' in sql_lower and 'fail' not in sql_lower:
                        # Add hygiene failure filtering
                        hygiene_filter = "(beard = 'fail' OR nail = 'fail' OR handLeg = 'fail' OR robe = 'fail')"
                        
                        if 'WHERE' in sql.upper():
                            sql = sql.replace('WHERE', f"WHERE {hygiene_filter} AND")
                        else:
                            if 'GROUP BY' in sql.upper():
                                sql = sql.replace('GROUP BY', f"WHERE {hygiene_filter} GROUP BY")
                            elif 'ORDER BY' in sql.upper():
                                sql = sql.replace('ORDER BY', f"WHERE {hygiene_filter} ORDER BY")
                            elif 'LIMIT' in sql.upper():
                                sql = sql.replace('LIMIT', f"WHERE {hygiene_filter} LIMIT")
                            else:
                                sql += f" WHERE {hygiene_filter}"
                        
                        LOGGER.info("Fixed: Added hygiene failure filtering")
            
            # Fix 5: Fix production queries that use SUM/AVG on various field names
            sum_avg_patterns = [
                ('sum(totalusage)', 'SUM(TotalWeight)'),
                ('avg(totalusage)', 'AVG(TotalWeight)'),
                ('sum(totalmaterusage)', 'SUM(TotalWeight)'),
                ('avg(totalmaterusage)', 'AVG(TotalWeight)'),
                ('sum(total_usage)', 'SUM(TotalWeight)'),
                ('avg(total_usage)', 'AVG(TotalWeight)'),
                ('sum(total_mater_usage)', 'SUM(TotalWeight)'),
                ('avg(total_mater_usage)', 'AVG(TotalWeight)')
            ]
            
            for old_pattern, new_pattern in sum_avg_patterns:
                if old_pattern in sql_lower:
                    sql = sql.replace(old_pattern, new_pattern)
                    sql = sql.replace(old_pattern.upper(), new_pattern)
                    sql = sql.replace(old_pattern.capitalize(), new_pattern)
                    LOGGER.info(f"Fixed: Changed {old_pattern} to {new_pattern}")
            
            return sql
            
        except Exception as e:
            LOGGER.error(f"Error applying SQL fixes: {e}")
            return sql

    def _preflight_validate_identifiers(
        self, sql: str
    ) -> Tuple[bool, List[str], List[str]]:
        invalid_tables: List[str] = []
        invalid_columns: List[str] = []
        try:
            # Capture table -> alias bindings from FROM/JOIN clauses
            alias_to_table: Dict[str, str] = {}
            table_pattern = re.compile(
                r"(?:FROM|JOIN)\s+`([^`]+)`(?:\s+([a-zA-Z_][\w]*))?", re.IGNORECASE
            )
            for tbl, alias in table_pattern.findall(sql):
                if tbl not in self.allowed_tables and tbl not in invalid_tables:
                    invalid_tables.append(tbl)
                # Map alias and bare table name to canonical table
                if alias:
                    alias_to_table[alias] = tbl
                alias_to_table[tbl] = tbl

            # Validate dotted identifiers belong to the referenced table
            dotted_pattern = re.compile(r"\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b")
            for alias, col in dotted_pattern.findall(sql):
                table_for_alias = alias_to_table.get(alias)
                if not table_for_alias:
                    # Alias does not map to a base table (likely a derived table); skip strict column check
                    continue
                # Ensure column exists in that specific table
                cols = set(self.table_to_columns.get(table_for_alias, []))
                if col not in cols and col not in invalid_columns:
                    invalid_columns.append(col)
        except Exception:
            pass
        ok = not invalid_tables and not invalid_columns
        return ok, invalid_tables, invalid_columns

    def _validate_fk_joins(self, sql: str) -> Tuple[bool, List[str]]:
        """Lightweight FK-graph validation for JOIN conditions.

        Parses simple ON clauses of the form a.col = b.col and verifies that
        either (a -> b) or (b -> a) exists in the introspected FK graph.
        """
        try:
            # Build alias -> table map
            alias_to_table: Dict[str, str] = {}
            for tbl, alias in re.findall(
                r"(?:FROM|JOIN)\s+`([^`]+)`(?:\s+([a-zA-Z_][\w]*))?",
                sql,
                flags=re.IGNORECASE,
            ):
                if alias:
                    alias_to_table[alias] = tbl
                alias_to_table[tbl] = tbl

            problems: List[str] = []
            for a_alias, a_col, b_alias, b_col in re.findall(
                r"ON\s+([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\s*=\s*([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)",
                sql,
                flags=re.IGNORECASE,
            ):
                a_table = alias_to_table.get(a_alias)
                b_table = alias_to_table.get(b_alias)
                if not a_table or not b_table:
                    # Likely join with a derived table; skip FK validation for this ON clause
                    continue
                # Allow self-joins (e.g., manager relationships) without requiring FK edge
                if a_table == b_table:
                    continue
                # Check fk edges in either direction
                a_edges = {(c, rt, rc) for c, rt, rc in self.fk_graph.get(a_table, [])}
                b_edges = {(c, rt, rc) for c, rt, rc in self.fk_graph.get(b_table, [])}
                ok = (a_col, b_table, b_col) in a_edges or (
                    b_col,
                    a_table,
                    a_col,
                ) in b_edges
                if not ok:
                    problems.append(
                        f"suspicious join: {a_table}.{a_col} = {b_table}.{b_col}"
                    )
            return len(problems) == 0, problems
        except Exception:
            return True, []

    def _maybe_build_per_group_topk(self, user_query: str) -> Optional[str]:
        """Deterministic builder for queries like 'top 5 employees by salary in each department'."""
        q = user_query.lower()
        m = re.search(r"top\s+(\d+)", q)
        if not m:
            return None
        k = int(m.group(1))
        if k <= 0 or k > 1000:
            return None
        if not ("department" in q or "each" in q or "per" in q or "by department" in q):
            return None
        # Require schema pieces
        if not ({"workers", "departments"} <= self.allowed_tables):
            return None
        # Note: workers table doesn't have salary, department_id, first_name, last_name columns
        # It has: id, firstName, lastName, name, section, creator, createDate
        # This builder won't work with current schema
            return None
        # Build windowed per-department top-k (no extra join back to base table)
        limit_val = (
            self.safe_exec.default_limit
            if hasattr(self.safe_exec, "default_limit")
            else 50
        )
        sql = (
            "SELECT ranked.first_name, ranked.last_name, d.name AS department_name, ranked.salary\n"
            "FROM (\n"
            "  SELECT w.id, w.name, w.department_id, w.salary,\n"
            "         ROW_NUMBER() OVER (PARTITION BY w.department_id ORDER BY w.salary DESC) AS salary_rank\n"
            "  FROM `workers` w\n"
            ") ranked\n"
            "JOIN `departments` d ON ranked.department_id = d.id\n"
            f"WHERE ranked.salary_rank <= {k}\n"
            "ORDER BY d.name, ranked.salary_rank\n"
            f"LIMIT {limit_val}"
        )
        return sql

    def _apply_column_mapping_fallback(self, sql: str, error_msg: str) -> str:
        """Apply column mapping fallback for common schema mismatches."""
        if not hasattr(self, 'vector_manager') or not self.vector_manager:
            return sql
            
        # Check for common column errors
        if "Unknown column" in error_msg or "doesn't exist" in error_msg:
            # Extract table and column from error message
            import re
            table_match = re.search(r"`(\w+)`\.`(\w+)`", error_msg)
            if table_match:
                table_name = table_match.group(1)
                invalid_column = table_match.group(2)
                
                # Try to get column mapping
                mapped_column = self.vector_manager.get_column_mapping(table_name, invalid_column)
                if mapped_column:
                    # Replace the invalid column with the mapped one
                    pattern = f"`{table_name}`\\.`{invalid_column}`"
                    replacement = f"`{table_name}`.`{mapped_column}`"
                    repaired_sql = re.sub(pattern, replacement, sql)
                    LOGGER.info(f"Mapped column {invalid_column} to {mapped_column} in table {table_name}")
                    return repaired_sql
                    
                # Try similarity matching
                suggested_column = self.vector_manager.suggest_column_mapping(table_name, invalid_column)
                if suggested_column:
                    pattern = f"`{table_name}`\\.`{invalid_column}`"
                    replacement = f"`{table_name}`.`{suggested_column}`"
                    repaired_sql = re.sub(pattern, replacement, sql)
                    LOGGER.info(f"Suggested column mapping: {invalid_column} -> {suggested_column} in table {table_name}")
                    return repaired_sql
        
        return sql

    def _maybe_build_trend_analysis(self, user_query: str) -> Optional[str]:
        """Deterministic builder for trend analysis queries."""
        q = user_query.lower()
        
        # Production trend analysis
        if any(word in q for word in ['production', 'volume', 'output']) and any(word in q for word in ['trend', 'over time', 'change']):
            return (
                "SELECT DATE_FORMAT(date, '%Y-%m') AS month, "
                "SUM(totalUsage) AS total_production "
                "FROM production_info "
                "WHERE date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH) "
                "GROUP BY DATE_FORMAT(date, '%Y-%m') "
                "ORDER BY month "
                "LIMIT 50"
            )
        
        # Waste trend analysis
        if any(word in q for word in ['waste', 'waste generation']) and any(word in q for word in ['trend', 'over time', 'change']):
            return (
                "SELECT DATE_FORMAT(date, '%Y-%m') AS month, "
                "SUM(value) AS total_waste "
                "FROM pack_waste "
                "WHERE date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH) "
                "GROUP BY DATE_FORMAT(date, '%Y-%m') "
                "ORDER BY month "
                "LIMIT 50"
            )
        
        # Quality trend analysis
        if any(word in q for word in ['quality', 'test', 'pass', 'fail']) and any(word in q for word in ['trend', 'over time']):
            return (
                "SELECT DATE_FORMAT(date, '%Y-%m') AS month, "
                "COUNT(*) AS total_tests, "
                "SUM(CASE WHEN result = 'pass' THEN 1 ELSE 0 END) AS passed_tests, "
                "ROUND(SUM(CASE WHEN result = 'pass' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pass_rate "
                "FROM production_test "
                "WHERE date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH) "
                "GROUP BY DATE_FORMAT(date, '%Y-%m') "
                "ORDER BY month "
                "LIMIT 50"
            )
        
        return None

    def _maybe_build_waste_efficiency_analysis(self, user_query: str) -> Optional[str]:
        """Deterministic builder for waste and efficiency analysis."""
        q = user_query.lower()
        
        # Waste efficiency analysis - ONLY for explicit waste efficiency queries
        # Must contain BOTH "waste" AND "efficiency" OR "waste" AND "analysis"
        if (('waste efficiency' in q or 'efficiency analysis' in q or 
             ('waste' in q and 'efficiency' in q) or
             ('waste' in q and 'analysis' in q)) and
            # Must NOT be simple listing queries
            not any(simple in q for simple in ['how many', 'count', 'number of', 'what is', 'list', 'show', 'display', 'get', 'find', 'all', 'unique', 'distinct', 'recent', 'latest']) and
            # Must NOT be about listing types or categories
            not any(list_word in q for list_word in ['types', 'categories', 'batches', 'recent', 'latest'])):
            return (
                "SELECT pw.type, "
                "SUM(pw.value) AS total_waste, "
                "COUNT(DISTINCT pw.date) AS waste_days, "
                "ROUND(SUM(pw.value) / COUNT(DISTINCT pw.date), 2) AS avg_waste_per_day "
                "FROM pack_waste pw "
                "WHERE pw.date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH) "
                "GROUP BY pw.type "
                "ORDER BY total_waste DESC "
                "LIMIT 50"
            )
        
        # Production efficiency analysis - only for explicit production efficiency queries
        if (('production efficiency' in q or 'efficiency analysis' in q or
             ('production' in q and 'efficiency' in q) or
             ('production' in q and 'analysis' in q)) and
            not any(simple in q for simple in ['how many', 'count', 'number of', 'what is'])):
            return (
                "SELECT w.name, "
                "SUM(p.totalUsage) AS total_production, "
                "COUNT(DISTINCT p.date) AS production_days, "
                "ROUND(SUM(p.totalUsage) / COUNT(DISTINCT p.date), 2) AS avg_production_per_day "
                "FROM production_info p "
                "JOIN workers w ON p.worker_id = w.id "
                "WHERE p.date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH) "
                "GROUP BY w.id, w.name "
                "ORDER BY total_production DESC "
                "LIMIT 50"
            )
        
        return None

    def _maybe_build_price_comparison(self, user_query: str) -> Optional[str]:
        """Deterministic builder for price comparison queries."""
        q = user_query.lower()
        
        # Most expensive ingredients
        if any(word in q for word in ['expensive', 'highest', 'most']) and any(word in q for word in ['ingredient', 'price', 'cost']):
            return (
                "SELECT ingredient_name, "
                "MAX(price) AS max_price, "
                "MIN(price) AS min_price, "
                "ROUND(AVG(price), 2) AS avg_price, "
                "COUNT(*) AS price_updates "
                "FROM prices "
                "WHERE date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH) "
                "GROUP BY ingredient_name "
                "ORDER BY max_price DESC "
                "LIMIT 50"
            )
        
        # Price trends over time
        if any(word in q for word in ['price', 'cost']) and any(word in q for word in ['trend', 'over time', 'change']):
            return (
                "SELECT DATE_FORMAT(date, '%Y-%m') AS month, "
                "ingredient_name, "
                "ROUND(AVG(price), 2) AS avg_price "
                "FROM prices "
                "WHERE date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH) "
                "GROUP BY DATE_FORMAT(date, '%Y-%m'), ingredient_name "
                "ORDER BY month, avg_price DESC "
                "LIMIT 50"
            )
        
        return None

    def _maybe_build_employee_profile(self, user_query: str) -> Optional[str]:
        """Deterministic aggregated profile for queries like 'analyze worker id 1363'.

        Produces a single-row summary to avoid row-per-skill duplication.
        """
        q = user_query.lower()
        m = re.search(r"(?:employee|worker)\s+id\s+(\d+)", q)
        if not m:
            return None
        worker_id = int(m.group(1))
        if worker_id <= 0:
            return None
        limit_val = (
            self.safe_exec.default_limit
            if hasattr(self.safe_exec, "default_limit")
            else 50
        )
        sql = (
            "SELECT\n"
            "  e.id, e.first_name, e.last_name,\n"
            "  d.name AS department_name,\n"
            "  COALESCE(pe.tasks_completed, 0) AS tasks_completed,\n"
            "  COALESCE(pe.projects_involved, 0) AS projects_involved,\n"
            "  COALESCE(pe.hours_worked, 0) AS hours_worked,\n"
            "  COALESCE(pe.client_feedback_score, 0) AS client_feedback_score,\n"
            "  COALESCE(pe.manager_rating, 0) AS manager_rating,\n"
            "  COALESCE(pe.overtime_hours, 0) AS overtime_hours,\n"
            "  COUNT(DISTINCT ep.project_id) AS projects_count,\n"
            "  GROUP_CONCAT(DISTINCT s.skill_name ORDER BY s.skill_name SEPARATOR ', ') AS skills\n"
            "FROM `employe` e\n"
            "LEFT JOIN `departments` d ON e.department_id = d.id\n"
            "LEFT JOIN `performance` pe ON e.id = pe.employee_id\n"
            "LEFT JOIN `employee_projects` ep ON e.id = ep.employee_id\n"
            "LEFT JOIN `skills` s ON e.id = s.employee_id\n"
            f"WHERE e.id = {worker_id}\n"
            "GROUP BY e.id, e.first_name, e.last_name, d.name\n"
            f"LIMIT {limit_val}"
        )
        return sql

    def _maybe_build_distribution_count(self, user_query: str) -> Optional[str]:
        """Deterministic builder for distribution counts (e.g., employee count by department).

        Triggers on phrases like 'count ... by department', 'employees per department', 'each department'.
        """
        q = user_query.lower()
        if not ("department" in q and any(k in q for k in [" by ", " per ", " each ", "group by"])):
            return None
        if not any(k in q for k in ["employee", "employees", "staff", "people", "worker", "workers"]):
            return None
        # Require schema pieces
        if not ("workers" in self.allowed_tables and "departments" in self.allowed_tables):
            return None
        # Note: workers table doesn't have department_id column
        # It has: id, firstName, lastName, name, section, creator, createDate
        # This builder won't work with current schema
            return None
        limit_val = (
            self.safe_exec.default_limit if hasattr(self.safe_exec, "default_limit") else 50
        )
        sql = (
            "SELECT d.name, COUNT(*)\n"
            "FROM `employe` e\n"
            "JOIN `departments` d ON e.department_id = d.id\n"
            "GROUP BY d.name\n"
            "ORDER BY COUNT(*) DESC\n"
            f"LIMIT {limit_val}"
        )
        return sql

    def _maybe_build_same_department_as_manager(self, user_query: str) -> Optional[str]:
        """Deterministic builder for 'employees whose manager is in the same department'.

        Requires `employe` to have `manager_id` and `department_id`. Uses a self-join.
        If columns are missing, returns None for LLM or other heuristics to handle.
        """
        q = user_query.lower()
        if not (
            ("same department" in q and "manager" in q)
            or ("manager" in q and "department" in q and any(k in q for k in ["same", "equal"]))
        ):
            return None
        if "workers" not in self.allowed_tables:
            return None
        # Note: workers table doesn't have department_id, manager_id columns
        # It has: id, firstName, lastName, name, section, creator, createDate
        # This builder won't work with current schema
            return None
        limit_val = (
            self.safe_exec.default_limit if hasattr(self.safe_exec, "default_limit") else 50
        )
        if any(k in q for k in ["how many", "count", "number"]):
            sql = (
                "SELECT COUNT(*)\n"
                "FROM `employe` e\n"
                "JOIN `employe` m ON e.manager_id = m.id\n"
                "WHERE e.department_id = m.department_id\n"
                f"LIMIT {limit_val}"
            )
        else:
            sql = (
                "SELECT e.first_name, e.last_name\n"
                "FROM `employe` e\n"
                "JOIN `employe` m ON e.manager_id = m.id\n"
                "WHERE e.department_id = m.department_id\n"
                f"LIMIT {limit_val}"
            )
        return sql

    def _maybe_build_employees_with_n_projects(self, user_query: str) -> Optional[str]:
        """Deterministic builder for queries like 'employees working in N (active) projects [by department | pie]'.

        Supports exact N. If 'active'/'ongoing'/'planned' present, filters projects by status.
        If 'by department'/'per department'/'each department' present, groups by department.
        If 'pie' present, returns label/value columns suitable for pie chart.
        """
        q = user_query.lower()
        # Must mention employee(s)/worker(s) and projects
        if not (("employee" in q or "worker" in q) and "project" in q):
            return None
        # Extract N
        m = re.search(r"\b(\d{1,3})\s+active\s+project", q)
        active_hint = False
        if m:
            n = int(m.group(1))
            active_hint = True
        else:
            m = re.search(r"\b(\d{1,3})\s+project", q)
            if not m:
                return None
            n = int(m.group(1))
        if n <= 0 or n > 1000:
            return None
        # Schema requirements
        required_tables = {"employee_projects", "workers"}
        if not required_tables <= self.allowed_tables:
            return None
        # Build subquery for employees with exactly N projects (optionally active)
        status_filter = ""
        if active_hint or any(s in q for s in ["active", "ongoing", "planned"]):
            # Require projects table
            if "projects" not in self.allowed_tables:
                return None
            status_filter = (
                " JOIN `projects` p ON ep.project_id = p.id\n"
                "  WHERE p.status IN ('Ongoing','Planned')\n"
            )
        sub = (
            "SELECT ep.employee_id\n"
            "FROM `employee_projects` ep\n"
            f"{status_filter}"
            "GROUP BY ep.employee_id\n"
            f"HAVING COUNT(DISTINCT ep.project_id) = {n}"
        )
        by_dept = any(k in q for k in [" by ", " per ", " each "]) and "department" in q
        limit_val = self.safe_exec.default_limit if hasattr(self.safe_exec, "default_limit") else 50
        if by_dept:
            # Need departments table
            if "departments" not in self.allowed_tables:
                return None
            is_pie = "pie" in q
            label_col = "label" if is_pie else "name"
            value_col = "value" if is_pie else "COUNT(*)"
            sql = (
                f"SELECT d.name AS {label_col}, COUNT(*) AS {value_col}\n"
                "FROM (\n" + sub + "\n) e2\n"
                "JOIN `employe` e ON e2.employee_id = e.id\n"
                "JOIN `departments` d ON e.department_id = d.id\n"
                "GROUP BY d.name\n"
                "ORDER BY COUNT(*) DESC\n"
                f"LIMIT {limit_val}"
            )
            return sql
        # Scalar count
        if any(w in q for w in ["how many", "count", "number"]):
            sql = (
                "SELECT COUNT(*)\n"
                "FROM (\n" + sub + "\n) t\n"
                f"LIMIT {limit_val}"
            )
            return sql
        # Otherwise, list employees
        sql = (
            "SELECT e.first_name, e.last_name\n"
            "FROM (\n" + sub + "\n) e2\n"
            "JOIN `employe` e ON e2.employee_id = e.id\n"
            f"LIMIT {limit_val}"
        )
        return sql

    # ---------------- Multi-step pipeline helpers -----------------
    def _plan_intent(self, user_query: str) -> Dict[str, Any]:
        text = user_query.lower()
        intent = "list"
        if any(k in text for k in ["how many", "how much", "count", "total"]):
            intent = "scalar"
        if any(k in text for k in ["distribution", "by "]):
            intent = "distribution"
        if any(k in text for k in ["top ", "most", "highest", "lowest"]):
            intent = "top/most"
        if any(k in text for k in ["compare", "versus", "vs "]):
            intent = "join-comparison"
        # Naive multi-intent split
        multi = []
        if " and " in text and (
            "how many" in text or "which" in text or "most" in text
        ):
            parts = [p.strip() for p in user_query.split(" and ") if p.strip()]
            if len(parts) > 1:
                multi = parts[:3]
        entities = list(self.allowed_tables)[:10]
        metrics: List[str] = []
        if "count" in text or "how many" in text or "total" in text:
            metrics.append("count")
        if "avg" in text or "average" in text:
            metrics.append("avg")
        if "sum" in text:
            metrics.append("sum")
        plan = {
            "intent": intent,
            "entities": entities,
            "metrics": metrics,
            "joins": [],
            "constraints": [],
            "multi_intent_split": multi,
        }
        return plan

    def _shape_ok(self, plan: Dict[str, Any], sql: str) -> Tuple[bool, str]:
        s = sql.strip().lower()
        has_group = " group by " in s
        is_scalar = (
            re.search(r"\bcount\s*\(\s*\*\s*\)", s) is not None
            or re.search(r"\b(sum|avg|min|max)\s*\(", s) is not None
        )
        has_limit = " limit " in s
        has_order = " order by " in s
        intent = plan.get("intent", "list")
        
        # More lenient validation - only check for obvious mismatches
        if intent == "scalar":
            # Scalar queries should return single values, but allow GROUP BY for complex aggregations
            if not is_scalar and not has_group:
                return False, "Scalar intent requires aggregate function"
        elif intent == "distribution":
            # Distribution queries should group data, but be lenient about exact structure
            if not has_group and not is_scalar:
                return False, "Distribution intent should group data or use aggregates"
        elif intent == "top/most":
            # Top/most queries should have ordering, but LIMIT is automatically added
            if not has_order:
                return False, "Top/most intent requires ORDER BY"
        
        return True, "ok"

    def _generate_candidates(
        self, user_query: str, context: RetrievedContext, mode: str, k: int = 3
    ) -> List[str]:
        candidates: List[str] = []
        # DISABLED: Deterministic builders using non-existent Farnan tables
        # These builders reference tables that don't exist in Farnan database:
        # - employe (should be workers)
        # - departments (doesn't exist)
        # - performance (doesn't exist)
        # - employee_projects (doesn't exist)
        # - skills (doesn't exist)
        
        # -1) Deterministic aggregated employee profile when explicitly asked - DISABLED
        # profile_sql = self._maybe_build_employee_profile(user_query)
        # if profile_sql:
        #     candidates.append(profile_sql)
        # 0) Deterministic heuristic candidate for per-group top-k - DISABLED
        # heuristic = self._maybe_build_per_group_topk(user_query)
        # if heuristic:
        #     candidates.append(heuristic)
        # 0.1) Deterministic distribution count builder - DISABLED
        # dist_sql = self._maybe_build_distribution_count(user_query)
        # if dist_sql:
        #     candidates.append(dist_sql)
        # 0.2) Deterministic same-department-as-manager builder - DISABLED
        # same_dept_sql = self._maybe_build_same_department_as_manager(user_query)
        # if same_dept_sql:
        #     candidates.append(same_dept_sql)
        # 0.3) Deterministic trend analysis builder
        trend_sql = self._maybe_build_trend_analysis(user_query)
        if trend_sql:
            candidates.append(trend_sql)
        # 0.4) Deterministic waste/efficiency analysis builder
        waste_sql = self._maybe_build_waste_efficiency_analysis(user_query)
        if waste_sql:
            candidates.append(waste_sql)
        # 0.5) Deterministic price analysis builder
        price_sql = self._maybe_build_price_analysis(user_query)
        if price_sql:
            candidates.append(price_sql)
        
        # 0.6) Smart date filtering builder (HIGHEST PRIORITY for natural language)
        date_filtered_sql = self._maybe_build_smart_date_filtered_queries(user_query)
        if date_filtered_sql:
            # Add to candidates instead of returning immediately to go through normal validation
            candidates.clear()
            candidates.append(date_filtered_sql)
            # Date filtering builder triggered
        # DISABLED: Price comparison builder uses wrong schema
        # The prices table has separate columns for each ingredient (ricotta, cream, oil, etc.)
        # NOT a generic ingredient_name and price column structure
        # 0.5) Deterministic price comparison builder - DISABLED
        # price_sql = self._maybe_build_price_comparison(user_query)
        # if price_sql:
        #     candidates.append(price_sql)
        # DISABLED: More builders using non-existent Farnan tables
        # 0.3) Deterministic employees with N projects builder - DISABLED
        # nproj_sql = self._maybe_build_employees_with_n_projects(user_query)
        # if nproj_sql:
        #     candidates.append(nproj_sql)
        # 0.6) Two-department comparison over years - DISABLED
        # cmp_sql = self._maybe_build_compare_two_departments_over_years(user_query)
        # if cmp_sql:
        #     candidates.append(cmp_sql)
        # 0.7) Deterministic simple aggregation builder for SHORT_ANSWER queries
        simple_agg_sql = self._maybe_build_simple_aggregation(user_query)
        if simple_agg_sql:
            candidates.append(simple_agg_sql)
        # 0.8) Deterministic table listing builder for TABLE queries
        table_list_sql = self._maybe_build_table_listing(user_query)
        if table_list_sql:
            candidates.append(table_list_sql)
        
        # 0.9) SMART DETERMINISTIC BUILDERS - Schema-aware patterns
        smart_waste_sql = self._maybe_build_smart_waste_analysis(user_query)
        if smart_waste_sql:
            candidates.append(smart_waste_sql)
            
        smart_production_sql = self._maybe_build_smart_production_analysis(user_query)
        if smart_production_sql:
            candidates.append(smart_production_sql)
            
        smart_worker_sql = self._maybe_build_smart_worker_analysis(user_query)
        if smart_worker_sql:
            candidates.append(smart_worker_sql)
            
        smart_hygiene_sql = self._maybe_build_smart_hygiene_analysis(user_query)
        if smart_hygiene_sql:
            candidates.append(smart_hygiene_sql)
            
        # 1.0) Complex multi-table analysis builder
        complex_analysis_sql = self._maybe_build_complex_multi_table_analysis(user_query)
        if complex_analysis_sql:
            candidates.append(complex_analysis_sql)
        
        # If we have deterministic builders, prioritize them and limit LLM candidates
        deterministic_count = len(candidates)
        if deterministic_count > 0:
            # Only generate 1 LLM candidate as fallback when we have deterministic builders
            llm_candidates = 1
        else:
            # Generate full k LLM candidates when no deterministic builders
            llm_candidates = k
            
        # 1..k) LLM candidates (limited when deterministic builders exist)
        for i in range(llm_candidates):
            # Use the same enhanced prompt approach as _generate_sql
            ctx_snippets = "\n\n".join(context.texts[:3])
            
            prompt = f"""You are an expert SQL generator for a food production database (Farnan).

QUERY: {user_query}
MODE: {mode}

ACTUAL DATABASE SCHEMA:
- pack_waste: date, type, value (waste tracking by type and amount)
- production_info: bakeType, totalUsage, ricotta, cream, oil, humidity, temp (main production data)
- packaging_info: bakeType, TotalWeight, tranWeight (packaging specifications)
- person_hyg: personName, beard, nail, handLeg, robe (hygiene compliance checks)
- prices: ricotta, cream, oil, buttermilkPowder (ingredient pricing)
- workers: firstName, lastName, section (employee information)
- production_test: bakeType, totalUsage (production test data)
- repo_nc: cheeseType, delivery, returns, total, usage (cheese repository)

INTELLIGENT PATTERNS:
- "waste by type" → SELECT type, SUM(value) FROM pack_waste GROUP BY type
- "production by bake type" → SELECT bakeType, SUM(totalUsage) FROM production_info GROUP BY bakeType
- "workers by section" → SELECT section, COUNT(*) FROM workers GROUP BY section
- "hygiene violations" → SELECT personName, COUNT(*) FROM person_hyg WHERE beard='fail' OR nail='fail' GROUP BY personName
- "ingredient prices" → SELECT ricotta, cream, oil FROM prices ORDER BY date DESC

MODE RULES:
- SHORT_ANSWER: Single scalar (COUNT, SUM, AVG, MAX, MIN) for "how many", "total", "average"
- TABLE: Multiple rows for "show", "list", "display", "by type/section"
- ANALYTICAL: Trends/patterns for "analyze", "compare", "trends", "over time"
- VISUALIZATION: Chart data for "pie chart", "bar chart", "histogram", "plot"

SCHEMA CONTEXT:
{ctx_snippets}

ENHANCED TABLE SELECTION:
- Price/Cost/Expensive/Cheap queries → 'prices' table ONLY (ricotta, cream, oil columns)
- Production/Volume/Batch/Bake queries → 'production_info' table (totalUsage, bakeType columns)
- Waste/Disposal queries → 'pack_waste' table (type, value columns)
- Hygiene/Compliance/Violation queries → 'person_hyg' table (personName, beard, nail columns)
- Worker/Employee/Staff/Section queries → 'workers' table (firstName, section columns)
- Packaging/Package queries → 'packaging_info' table (bakeType, TotalWeight columns)
- Test/Quality queries → 'production_test' table (bakeType, totalUsage columns)

DOMAIN-SPECIFIC MAPPINGS:
- "production volumes" → production_info.totalUsage
- "packaging waste" → pack_waste table (NOT packaging_info)
- "hygiene check results" → person_hyg table
- "recent batches" → production_info table
- "packaging information" → packaging_info table

CRITICAL TABLE RULES:
- NEVER use 'production_info' for price/cost queries
- NEVER use 'packaging_info' for waste queries
- ALWAYS use 'person_hyg' for hygiene queries

EXAMPLES:
- "How many workers?" → SELECT COUNT(*) FROM `workers` LIMIT 50
- "Show production volumes" → SELECT SUM(totalUsage) FROM `production_info` LIMIT 50
- "Waste distribution by type" → SELECT `type`, COUNT(*) FROM `pack_waste` GROUP BY `type` LIMIT 50
- "Packaging types with pie chart" → SELECT `bakeType`, COUNT(*) FROM `packaging_info` GROUP BY `bakeType` LIMIT 50

RULES:
- Use ONLY the provided schema context
- Always add LIMIT 50
- Use backticks around table/column names
- For SHORT_ANSWER: return single value
- For VISUALIZATION: return 2+ columns (label, value)
- Output ONLY SQL in ```sql``` fences"""
            
            try:
                resp = self.llm.invoke(prompt)
                raw = resp.content if hasattr(resp, "content") else resp
                txt: str = _stringify_llm_content(raw)
                sql = _extract_sql_from_text(txt)
                if sql:
                    candidates.append(sql)
            except Exception:
                continue
        # De-duplicate while preserving order
        seen: set[str] = set()
        unique_candidates: List[str] = []
        for c in candidates:
            norm = re.sub(r"\s+", " ", c.strip())
            if norm in seen:
                continue
            seen.add(norm)
            unique_candidates.append(c)
        return unique_candidates

    def _validate_candidate(self, plan: Dict[str, Any], sql: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"sql": sql, "valid": False, "errors": [], "score": 0}
        try:
            # Grammar/SELECT-only
            try:
                self.safe_exec.validate_select_only(sql)
                result["score"] += 1
            except Exception as e:
                result["errors"].append(f"select_only: {e}")
                return result
            # Identifiers (alias-aware)
            ok_ids, bad_tables, bad_cols = self._preflight_validate_identifiers(sql)
            if not ok_ids:
                result["errors"].append(
                    f"identifiers: invalid tables {bad_tables} / columns {bad_cols}"
                )
                return result
            result["score"] += 2
            # Shape
            ok_shape, msg = self._shape_ok(plan, sql)
            if not ok_shape:
                result["errors"].append(f"shape: {msg}")
                return result
            result["score"] += 1
            # FK-join sanity
            fk_ok, fk_problems = self._validate_fk_joins(sql)
            if not fk_ok:
                result["errors"].append(f"fk_joins: {fk_problems}")
                return result
            result["score"] += 1
            # Soft preference: joins aligned with schema graph edges
            try:
                alias_to_table: Dict[str, str] = {}
                for tbl, alias in re.findall(
                    r"(?:FROM|JOIN)\s+`([^`]+)`(?:\s+([a-zA-Z_][\w]*))?",
                    sql,
                    flags=re.IGNORECASE,
                ):
                    if alias:
                        alias_to_table[alias] = tbl
                    alias_to_table[tbl] = tbl
                edge_bonus = 0
                for a_alias, a_col, b_alias, b_col in re.findall(
                    r"ON\s+([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\s*=\s*([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)",
                    sql,
                    flags=re.IGNORECASE,
                ):
                    a_table = alias_to_table.get(a_alias)
                    b_table = alias_to_table.get(b_alias)
                    if (
                        a_table
                        and b_table
                        and self.schema_graph.has_edge(
                            ("table", a_table), ("table", b_table)
                        )
                    ):
                        edge_bonus += 1
                result["score"] += min(edge_bonus, 2)
            except Exception:
                pass
            # EXPLAIN
            try:
                _ = self.safe_exec.explain(sql)
                result["score"] += 1
            except Exception as e:
                result["errors"].append(f"explain: {e}")
                return result
            result["valid"] = True
            return result
        except Exception as e:
            result["errors"].append(str(e))
            return result

    def _format_table(self, df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "(no rows)"
        return df.to_markdown(index=False)

    def _short_answer_from_df(self, df: pd.DataFrame) -> Optional[str]:
        if df is None or df.empty:
            return None
        # Only emit short answer for a true scalar shape (1 row x 1 column)
        if df.shape == (1, 1):
            value = df.iat[0, 0]
            return str(value)
        # If there is a single row with a canonical scalar column name, allow
        if df.shape[0] == 1 and any(
            name.lower() in {"value", "count", "total", "sum", "avg", "average"}
            for name in df.columns
        ):
            # prefer the canonical column
            for name in df.columns:
                if name.lower() in {"value", "count", "total", "sum", "avg", "average"}:
                    return str(df[name].iloc[0])
        return None

    def _fallback_short_answer(self, df: pd.DataFrame) -> Optional[str]:
        """Conservative fallback for single-row, multi-column results.

        Use when we want a concise highlight even if it's not a pure scalar shape
        (e.g., COMBO aggregated profile). Picks a meaningful column if present.
        """
        try:
            if df is None or df.empty or len(df) != 1:
                return None
            preferred = ["count", "total", "sum", "avg", "average", "id"]
            lower_cols = [c.lower() for c in df.columns]
            for p in preferred:
                if p in lower_cols:
                    idx = lower_cols.index(p)
                    return str(df.iloc[0, idx])
            # fallback to the first cell
            return str(df.iloc[0, 0])
        except Exception:
            return None

    def _post_exec_shape_adjust(self, plan: Dict[str, Any], df: Optional[pd.DataFrame], mode: str) -> str:
        """Adjust mode after execution if the actual result shape doesn't fit the intent.

        - scalar intent must return 1x1; otherwise downgrade to TABLE
        - distribution intent should return at least 2 columns and multiple rows; otherwise TABLE
        """
        try:
            intent = plan.get("intent", "list")
            if intent == "scalar" or mode == "SHORT_ANSWER":
                if not (df is not None and df.shape == (1, 1)):
                    return "TABLE"
            if intent == "distribution":
                if not (df is not None and df.shape[1] >= 2 and len(df) >= 2):
                    return "TABLE"
        except Exception:
            return mode
        return mode

    def _maybe_visualize(
        self, df: pd.DataFrame, title: str, user_query: str = ""
    ) -> Optional[str]:
        if df is None or df.empty or len(df) == 0 or df.shape[1] < 2:
            return None
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set up the plotting style
            plt.style.use(
                "seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default"
            )
            sns.set_palette("husl")

            x_col = df.columns[0]
            y_candidates = [
                c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])
            ]
            if not y_candidates:
                return None
            # Multi-series: if there are multiple numeric columns, plot them as multiple lines/bars
            multi_series = len(y_candidates) > 1 and len(y_candidates) <= 6

            # Determine chart type based on data and query hints
            chart_type = self._determine_chart_type(
                df, x_col, y_candidates[0], user_query
            )

            fig, ax = plt.subplots(figsize=(10, 6))

            if chart_type == "line":
                if multi_series:
                    for yc in y_candidates:
                        ax.plot(
                            df[x_col],
                            df[yc],
                            marker="o",
                            linewidth=2,
                            markersize=6,
                            label=str(yc),
                        )
                    ax.legend()
                else:
                    ax.plot(
                        df[x_col],
                        df[y_candidates[0]],
                        marker="o",
                        linewidth=2,
                        markersize=6,
                    )
            elif chart_type == "scatter":
                ax.scatter(df[x_col], df[y_candidates[0]], alpha=0.6, s=50)
            elif chart_type == "pie":
                # For pie charts, use the first column as labels
                df_top = df.head(10)  # Limit to top 10 for readability
                ax.pie(
                    df_top[y_candidates[0]],
                    labels=df_top[x_col],
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax.axis("equal")
            elif chart_type == "histogram":
                # Prefer a well-known numeric column if present
                if "salary" in df.columns and pd.api.types.is_numeric_dtype(
                    df["salary"]
                ):
                    series = df["salary"].dropna()
                else:
                    series = df[y_candidates[0]].dropna()
                count = max(1, int(len(series)))
                # Ensure a positive number of bins; use a reasonable minimum
                bins = max(5, min(20, max(1, count // 2)))
                ax.hist(series, bins=bins, alpha=0.7, edgecolor="black")
            else:  # Default bar chart
                if multi_series:
                    # Grouped bars
                    import numpy as np

                    categories = df[x_col].astype(str).tolist()
                    x = np.arange(len(categories))
                    width = max(0.1, 0.8 / len(y_candidates))
                    for i, yc in enumerate(y_candidates):
                        ax.bar(
                            x + i * width,
                            df[yc],
                            width=width,
                            label=str(yc),
                            alpha=0.85,
                        )
                    ax.set_xticks(x + width * (len(y_candidates) - 1) / 2)
                    ax.set_xticklabels(categories, rotation=45)
                    ax.legend()
                else:
                    colors = plt.cm.Set3(range(len(df))) if df is not None and not df.empty else []
                    bars = ax.bar(
                        df[x_col].astype(str),
                        df[y_candidates[0]],
                        color=colors,
                        alpha=0.8,
                    )
                    ax.tick_params(axis="x", labelrotation=45)
                    # Add value labels on bars if not too many
                    if df is not None and not df.empty and len(df) <= 15:
                        for bar, value in zip(bars, df[y_candidates[0]]):
                            height = bar.get_height()
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height,
                                f"{value:.1f}"
                                if isinstance(value, float)
                                else str(value),
                                ha="center",
                                va="bottom",
                            )

            ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
            if chart_type != "pie":
                ax.set_xlabel(x_col, fontsize=12)
                ax.set_ylabel("value", fontsize=12)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            path = os.path.join(
                "outputs",
                "plots",
                f"viz_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}.png",
            )
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return path
        except Exception as exc:
            LOGGER.warning("Visualization failed: %s", exc)
            return None

    def _determine_chart_type(
        self, df: pd.DataFrame, x_col: str, y_col: str, user_query: str
    ) -> str:
        """Determine the best chart type based on data characteristics and user query."""
        query_lower = user_query.lower()

        # Check for explicit chart type requests
        if any(word in query_lower for word in ["pie", "donut"]):
            return "pie"
        if any(word in query_lower for word in ["scatter", "correlation"]):
            return "scatter"
        if any(word in query_lower for word in ["histogram", "distribution"]):
            return "histogram"
        if any(word in query_lower for word in ["line", "trend", "over time"]):
            return "line"

        # Automatic detection based on data
        if (
            pd.api.types.is_datetime64_any_dtype(df[x_col])
            or "date" in x_col.lower()
            or "time" in x_col.lower()
        ):
            return "line"

        # For small datasets with categories, consider pie chart
        if df is not None and not df.empty and len(df) <= 10 and not pd.api.types.is_numeric_dtype(df[x_col]):
            if any(
                word in query_lower
                for word in ["share", "percentage", "proportion", "breakdown"]
            ):
                return "pie"

        # Default to bar chart
        return "bar"

    def _analysis_narrative(
        self, user_query: str, df: pd.DataFrame, sql: str, context: RetrievedContext
    ) -> str:
        sys_prompt = mcp_handler.build_analytical_prompt()
        summary = {
            "question": user_query,
            "sql": sql,
            "columns": list(df.columns) if df is not None else [],
            "row_count": int(len(df)) if df is not None else 0,
            "sample_rows": df.head(5).to_dict(orient="records")
            if df is not None
            else [],
            "context": context.texts[:4],
        }
        prompt = (
            sys_prompt + "\n\n" + json.dumps(summary, ensure_ascii=False, default=str)
        )
        try:
            resp = self.llm.invoke(prompt)
            raw = resp.content if hasattr(resp, "content") else resp
            text_out: str = _stringify_llm_content(raw)
            return text_out
        except Exception as exc:
            LOGGER.warning("Analytical narration failed: %s", exc)
            return "Unable to generate analytical summary."

    def _maybe_forecast(self, df: pd.DataFrame, user_query: str) -> Optional[str]:
        """If the query implies forecasting, fit a simple model and save a forecast plot.

        Uses statsmodels (ARIMA) fallback; expects first column to be a datetime-like index or a year/month.
        """
        try:
            q = user_query.lower()
            if not any(
                k in q
                for k in ["forecast", "next year", "future", "predict", "projection"]
            ):
                return None
            import matplotlib.pyplot as plt
            from statsmodels.tsa.arima.model import ARIMA

            # Try to infer a time column and a single numeric target (sum if multiple)
            time_col = df.columns[0]
            numeric_cols = [
                c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])
            ]
            if not numeric_cols:
                return None
            y = df[numeric_cols[0]].astype(float)
            x = df[time_col]
            # Coerce to pandas PeriodIndex (year) or datetime
            if (
                pd.api.types.is_integer_dtype(x)
                or x.astype(str).str.match(r"^\d{4}$").all()
            ):
                # Treat as years
                idx = pd.PeriodIndex(x.astype(int), freq="Y").to_timestamp()
            else:
                idx = pd.to_datetime(x, errors="coerce")
            series = pd.Series(y.values, index=idx).dropna()
            if len(series) < 6:
                return None
            model = ARIMA(series, order=(1, 1, 1))
            fitted = model.fit()
            forecast = fitted.get_forecast(steps=1)
            _ = forecast.predicted_mean.iloc[-1]
            conf_int = forecast.conf_int().iloc[-1]
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            series.plot(ax=ax, label="history")
            forecast.predicted_mean.plot(ax=ax, style="r--", label="forecast")
            ax.fill_between(
                forecast.predicted_mean.index,
                conf_int[0],
                conf_int[1],
                color="pink",
                alpha=0.3,
            )
            ax.set_title("Forecast")
            ax.legend()
            plt.tight_layout()
            path = os.path.join(
                "outputs",
                "plots",
                f"forecast_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}.png",
            )
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return path
        except Exception as exc:
            LOGGER.warning("Forecasting failed: %s", exc)
            return None

    def _get_hybrid_rag_context(self, user_query: str) -> RetrievedContext:
        """HYBRID RAG INTEGRATION: Combine Vector RAG + Graph RAG for enhanced context retrieval.
        
        This method integrates our advanced Graph RAG features with the existing Vector RAG
        to provide comprehensive context for better query understanding and SQL generation.
        
        Args:
            user_query: The user's natural language query
            
        Returns:
            RetrievedContext: Combined context from both Vector and Graph RAG systems
        """
        try:
            LOGGER.info(f"Getting hybrid RAG context for query: {user_query}")
            
            # 1. Get Vector RAG context (existing functionality)
            vector_context = self.vector_manager.similarity_search(user_query, top_k=8)
            LOGGER.debug(f"Vector RAG found {len(vector_context.texts)} context items")
            
            # 2. Get Graph RAG context (new advanced features)
            graph_context = self.get_advanced_schema_context(user_query, search_strategy="hybrid")
            LOGGER.debug(f"Graph RAG retrieved context with {len(graph_context.get('results', {}))} search strategies")
            
            # 3. Combine contexts intelligently
            combined_texts = []
            combined_metadatas = []
            
            # Add Vector RAG context (existing functionality preserved)
            combined_texts.extend(vector_context.texts)
            combined_metadatas.extend(vector_context.metadatas)
            
            # Add Graph RAG context (new enhanced features)
            # Get schema summary from Graph RAG
            graph_context_summary = graph_context.get('schema_summary', [])
            if isinstance(graph_context_summary, list):
                combined_texts.extend(graph_context_summary)
                combined_metadatas.extend([{'source': 'graph_rag', 'strategy': 'hybrid'}] * len(graph_context_summary))
            elif isinstance(graph_context_summary, str) and graph_context_summary.strip():
                combined_texts.append(graph_context_summary)
                combined_metadatas.append({'source': 'graph_rag', 'strategy': 'hybrid'})
            
            # Also add combined results context if available
            combined_results = graph_context.get('results', {}).get('combined', {})
            if combined_results and combined_results.get('insights'):
                for insight in combined_results['insights'][:2]:  # Limit to top 2 insights
                    if isinstance(insight, str) and insight.strip():
                        combined_texts.append(f"Graph RAG insight: {insight}")
                        combined_metadatas.append({'source': 'graph_rag', 'type': 'combined_insight'})
            
            # Add schema insights from Graph RAG
            schema_insights = graph_context.get('schema_insights', [])
            if isinstance(schema_insights, list):
                for insight in schema_insights[:3]:  # Limit to top 3 insights
                    if isinstance(insight, str) and insight.strip():
                        combined_texts.append(f"Schema insight: {insight}")
                        combined_metadatas.append({'source': 'graph_rag', 'type': 'schema_insight'})
            
            # Add community insights
            community_insights = graph_context.get('community_insights', [])
            if isinstance(community_insights, list):
                for insight in community_insights[:2]:  # Limit to top 2 community insights
                    if isinstance(insight, str) and insight.strip():
                        combined_texts.append(f"Community insight: {insight}")
                        combined_metadatas.append({'source': 'graph_rag', 'type': 'community_insight'})
            
            # Create enhanced context
            enhanced_context = RetrievedContext(
                texts=combined_texts,
                metadatas=combined_metadatas
            )
            
            LOGGER.info(f"Hybrid RAG context created with {len(enhanced_context.texts)} total items "
                       f"({len(vector_context.texts)} from Vector RAG, "
                       f"{len(enhanced_context.texts) - len(vector_context.texts)} from Graph RAG)")
            
            return enhanced_context
            
        except Exception as exc:
            LOGGER.warning(f"Hybrid RAG context generation failed, falling back to Vector RAG only: {exc}")
            # Fallback to Vector RAG only if Graph RAG fails
            return self.vector_manager.similarity_search(user_query, top_k=8)

    def process(
        self,
        user_query: str,
        prefer_mode: Optional[str] = None,
        export: Optional[str] = None,
    ) -> NL2SQLOutput:
        start_time = time.time()
        # Soft override if the user includes phrases like "use combo mode" inline
        override_mode: Optional[str] = None
        try:
            lower_q = user_query.lower()
            if "use combo mode" in lower_q or "use combo mod" in lower_q:
                override_mode = "COMBO"
                user_query = re.sub(
                    r"use\s+combo\s+mod(e)?", "", user_query, flags=re.IGNORECASE
                ).strip()
            elif "use analytical mode" in lower_q:
                override_mode = "ANALYTICAL"
                user_query = re.sub(
                    r"use\s+analytical\s+mode", "", user_query, flags=re.IGNORECASE
                ).strip()
            elif "use table mode" in lower_q:
                override_mode = "TABLE"
                user_query = re.sub(
                    r"use\s+table\s+mode", "", user_query, flags=re.IGNORECASE
                ).strip()
            elif "use short answer" in lower_q:
                override_mode = "SHORT_ANSWER"
                user_query = re.sub(
                    r"use\s+short\s+answer", "", user_query, flags=re.IGNORECASE
                ).strip()
            elif "use visualization mode" in lower_q or "use chart" in lower_q:
                override_mode = "VISUALIZATION"
                user_query = re.sub(
                    r"use\s+visualization\s+mode|use\s+chart",
                    "",
                    user_query,
                    flags=re.IGNORECASE,
                ).strip()
        except Exception:
            pass
        
        # HYBRID RAG INTEGRATION: Combine Vector RAG + Graph RAG for enhanced context
        context = self._get_hybrid_rag_context(user_query)
        mode = self._detect_mode(user_query, context, override_mode or prefer_mode)
        # Step 1: Plan JSON
        plan = self._plan_intent(user_query)
        # Add user_query to plan for clarification logic
        plan["user_query"] = user_query
        
        # Check for underspecification before proceeding
        if self._is_underspecified(plan):
            return self._ask_clarification(plan)
        
        # Step 3: Candidate generation (k-best)
        candidates = self._generate_candidates(user_query, context, mode, k=3)
        validations: List[Dict[str, Any]] = []
        chosen_sql: Optional[str] = None
        # Step 4: Validate each candidate
        deterministic_count = len([c for c in candidates if self._is_deterministic_builder_sql(c)])
        for i, c in enumerate(candidates):
            # Pre-validate schema usage before full validation
            schema_issues = self._pre_validate_schema_usage(c, context)
            if schema_issues:
                validations.append({
                    "sql": c,
                    "valid": False,
                    "errors": schema_issues,
                    "score": 0,
                    "schema_pre_validation": "failed"
                })
                continue
                
            validation = self._validate_candidate(plan, c)
            
            # Give priority boost to deterministic builders (first candidates)
            if i < deterministic_count:
                validation["score"] = validation.get("score", 0) + 5  # Significant boost
                validation["deterministic"] = True
            else:
                validation["deterministic"] = False
                
            validations.append(validation)
        # Step 5: Alignment-based reranking + execution-guided selection
        # Add small alignment bonus based on intent cues
        for v in validations:
            v["score"] = v.get("score", 0) + self._compute_alignment_bonus(user_query, plan, v.get("sql", ""))
        ranked = sorted(validations, key=lambda v: v.get("score", 0), reverse=True)
        for v in ranked[:3]:
            if not v.get("valid"):
                continue
            try:
                test_df, test_sql = self.safe_exec.execute_select(
                    v["sql"], user_query, self
                )  # will cache
                # Prefer non-empty results
                chosen_sql = test_sql
                df = test_df
                executed_sql = test_sql
                break
            except Exception:
                continue
        # If none executed, fallback to best-scored SQL (will hit repair path later)
        if not chosen_sql and ranked:
            chosen_sql = ranked[0]["sql"]
        # If still nothing, single-shot
        if not chosen_sql:
            chosen_sql = self._generate_sql(user_query, context, mode)
        sql = chosen_sql
        df: Optional[pd.DataFrame] = locals().get("df") if "df" in locals() else None
        executed_sql = (
            locals().get("executed_sql") if "executed_sql" in locals() else None
        )
        error_msg: Optional[str] = None

        if sql:
            # Identifier preflight; if invalid, try targeted repair once
            ok_ids, bad_tables, bad_cols = self._preflight_validate_identifiers(sql)
            if not ok_ids:
                repair_payload = {
                    "question": user_query,
                    "failed_sql": sql,
                    "error": {
                        "invalid_tables": bad_tables,
                        "invalid_columns": bad_cols,
                    },
                    "allowed_tables": sorted(list(self.allowed_tables))[:100],
                    "allowed_columns": sorted(list(self.allowed_columns))[:300],
                }
                try:
                    resp_ids = self.llm.invoke(
                        mcp_handler.build_sql_repair_prompt()
                        + "\n\n"
                        + json.dumps(repair_payload, ensure_ascii=False)
                    )
                    raw_ids = (
                        resp_ids.content if hasattr(resp_ids, "content") else resp_ids
                    )
                    txt_ids: str = _stringify_llm_content(raw_ids)
                    repaired_ids = _extract_sql_from_text(txt_ids)
                    if repaired_ids:
                        sql = repaired_ids
                except Exception as _:
                    pass

            try:
                df, executed_sql = self.safe_exec.execute_select(sql, user_query, self)
            except Exception as exc:
                error_msg = str(exc)
                LOGGER.warning("Initial SQL failed, attempting repair: %s", exc)
                # single-shot repair attempt
                repair_prompt = mcp_handler.build_sql_repair_prompt()
                
                # Enhanced repair context with specific schema fixes and column mapping
                schema_hints = []
                error_str = str(exc).lower()
                repaired_sql = sql  # Start with original SQL
                
                # Try column mapping fallback first
                if "unknown column" in error_str or "doesn't exist" in error_str:
                    repaired_sql = self._apply_column_mapping_fallback(sql, str(exc))
                    if repaired_sql != sql:
                        LOGGER.info("Applied column mapping fallback")
                        try:
                            df, executed_sql = self.safe_exec.execute_select(repaired_sql, user_query, self)
                            sql = repaired_sql
                            error_msg = None  # Clear error since repair succeeded
                            # Skip LLM repair - fallback succeeded
                        except Exception:
                            pass  # Fallback didn't work, continue with LLM repair
                
                if "doesn't exist" in error_str:
                    schema_hints.append("⚠️ Table/column doesn't exist - check schema context for exact names")
                if "employee_id" in error_str and any("employe" in text for text in context.texts):
                    schema_hints.append("🔧 Use `employe.id` not `employe.employee_id` - check FK relationships")
                if "unknown column" in error_str:
                    schema_hints.append("🔧 Column name mismatch - use exact names from schema context")
                    # Add specific column mapping hints
                    schema_hints.append("🔧 Common mappings: pt.sample → sampleCount, package_type → type")
                
                payload = {
                    "question": user_query,
                    "failed_sql": sql,
                    "error": str(exc),
                    "context": context.texts[:6],
                    "repair_hints": schema_hints,
                    "critical_reminder": "Use EXACT table and column names from schema context. For employe table, primary key is `id` not `employee_id`"
                }
                try:
                    resp = self.llm.invoke(
                        repair_prompt + "\n\n" + json.dumps(payload, ensure_ascii=False)
                    )
                    raw = resp.content if hasattr(resp, "content") else resp
                    txt: str = _stringify_llm_content(raw)
                    repaired_sql = _extract_sql_from_text(txt)
                    if repaired_sql:
                        df, executed_sql = self.safe_exec.execute_select(repaired_sql, user_query, self)
                        sql = repaired_sql
                        error_msg = None  # Clear error since repair succeeded
                except Exception as exc2:
                    LOGGER.error("SQL repair failed: %s", exc2)
                    error_msg = f"Original error: {exc}. Repair failed: {exc2}"

        # Clarification gate: if no valid candidates and plan is underspecified, ask a question instead
        if (
            (not validations or sum(1 for v in validations if v.get("valid")) == 0)
            and (not plan.get("metrics") and plan.get("intent") not in {"list", "top/most"})
        ):
            analysis = (
                "Clarification needed: Which department or time period should we consider, and what metric (count, avg, sum)?"
            )
            return NL2SQLOutput(
                mode="ANALYTICAL",
                sql=None,
                table_markdown=None,
                short_answer=None,
                analysis=analysis,
                visualization_path=None,
                metadata={"row_count": 0, "execution_time_ms": (time.time() - start_time) * 1000, "confidence": 0.0},
            )

        # Post-execution shape adjustment and short-answer gating
        mode = self._post_exec_shape_adjust(plan, df, mode)
        table_md = self._format_table(df) if df is not None else None
        short_answer = self._short_answer_from_df(df) if df is not None else None
        if short_answer is None and df is not None and not df.empty and len(df) == 1 and mode in ("COMBO", "ANALYTICAL"):
            short_answer = self._fallback_short_answer(df)
        viz_path: Optional[str] = None
        forecast_path: Optional[str] = None
        analysis: Optional[str] = None

        if mode in ("VISUALIZATION", "COMBO") and df is not None:
            viz_path = self._maybe_visualize(
                df, title=user_query, user_query=user_query
            )
        if mode in ("ANALYTICAL", "COMBO"):
            analysis = self._analysis_narrative(
                user_query, df if df is not None else pd.DataFrame(), sql or "", context
            )
        # Optional forecasting layer
        if df is not None:
            fp = self._maybe_forecast(df, user_query)
            if fp:
                forecast_path = fp

        if export and df is not None:
            try:
                ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
                base = os.path.join("outputs", "exports", f"export_{ts}")
                if export.lower() == "csv":
                    df.to_csv(base + ".csv", index=False)
                elif export.lower() == "json":
                    df.to_json(base + ".json", orient="records")
            except Exception as exc:
                LOGGER.warning("Export failed: %s", exc)

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Add to history
        history_id = self.history.add_entry(
            user_query=user_query,
            sql_query=executed_sql or sql,
            mode=mode,
            execution_time_ms=execution_time_ms,
            row_count=len(df) if df is not None and not df.empty else 0,
            error=error_msg,
        )

        # Step 6: Confidence & artifacts
        num_valid = sum(1 for v in validations if v.get("valid"))
        total_candidates = len(validations)
        base_confidence = num_valid / total_candidates if total_candidates > 0 else 0.0
        
        # ENHANCED CONFIDENCE CALCULATION with multiple factors
        confidence = self._calculate_intelligent_confidence(
            user_query, sql or "", validations, base_confidence, deterministic_count > 0, df is not None
        )
        # Attach artifacts into analysis if not present
        if analysis is None:
            analysis_parts = []
            analysis_parts.append("Plan JSON:\n" + json.dumps(plan, indent=2))
            if candidates:
                for i, c in enumerate(candidates, 1):
                    analysis_parts.append(f"Candidate {i}:\n```sql\n{c}\n```")
            if validations:
                analysis_parts.append(
                    "Validation Results:\n" + json.dumps(validations, indent=2)
                )
            analysis_parts.append(f"Confidence: {confidence:.2f}")
            if forecast_path:
                analysis_parts.append(f"Forecast plot saved to: {forecast_path}")
            analysis = "\n\n".join(analysis_parts)

        return NL2SQLOutput(
            mode=mode,
            sql=executed_sql or sql,
            table_markdown=table_md,
            short_answer=short_answer,
            analysis=analysis,
            visualization_path=viz_path,
            metadata={
                "row_count": int(len(df)) if df is not None and not df.empty else 0,
                "execution_time_ms": execution_time_ms,
                "history_id": history_id,
                "confidence": confidence,
                "forecast_path": forecast_path,
            },
        )

    def _is_underspecified(self, plan: Dict[str, Any]) -> bool:
        """Check if the plan is underspecified and requires clarification."""
        # Check for vague staffing requests
        user_query = plan.get("user_query", "").lower()
        entities = plan.get("entities", [])
        
        # Staffing questions without specifics
        if "employee" in entities and any(word in user_query for word in ["need", "require", "more", "additional"]):
            if not any(kw in user_query for kw in ["department", "dept", "role", "skill", "project", "team"]):
                return True
        
        # Vague comparison requests
        if "compare" in user_query and "department" in user_query:
            if not any(kw in user_query for kw in ["vs", "versus", "and", "with", "engineering", "sales", "hr", "marketing", "finance"]):
                return True
        
        # Vague performance questions
        if any(word in user_query for word in ["performance", "doing", "how are we"]):
            if not any(kw in user_query for kw in ["department", "project", "employee", "year", "month", "metric"]):
                return True
        
        return False

    def _ask_clarification(self, plan: Dict[str, Any]) -> NL2SQLOutput:
        """Generate a clarification response based on the plan."""
        user_query = plan.get("user_query", "").lower()
        
        # Handle staffing questions specifically
        if "employee" in plan.get("entities", []) and any(word in user_query for word in ["need", "require", "more", "additional"]):
            clarification = "For staffing needs, please specify: which department, role, or skill set are you looking to expand?"
        # Handle comparison questions
        elif "compare" in user_query and "department" in user_query:
            clarification = "Please specify which departments you'd like to compare (e.g., 'Engineering vs Sales') and what metric (headcount, salary, performance)."
        # Handle vague performance questions
        elif any(word in user_query for word in ["performance", "doing", "how are we"]):
            clarification = "Please specify what aspect of performance you'd like to analyze (department, project, time period, or specific metrics)."
        else:
            clarification = "Please clarify your question with more specific details about entities, metrics, or time periods."
        
        return NL2SQLOutput(
            mode="CLARIFICATION",
            sql="",
            table_markdown="",
            short_answer="",
            analysis=clarification,
            visualization_path="",
            metadata={"clarification_required": True},
        )

    def _calculate_intelligent_confidence(
        self, user_query: str, sql: str, validations: List[Dict], base_confidence: float, 
        is_deterministic: bool, has_data: bool
    ) -> float:
        """Calculate intelligent confidence with multiple factors"""
        confidence = base_confidence
        
        # Schema match bonus - check if using correct tables
        correct_tables = ['pack_waste', 'production_info', 'workers', 'person_hyg', 'prices', 'packaging_info']
        if any(table in sql.lower() for table in correct_tables):
            confidence += 0.3
        
        # Pattern match bonus - check for proper SQL patterns
        good_patterns = ['group by', 'sum(', 'count(', 'avg(', 'order by']
        if any(pattern in sql.lower() for pattern in good_patterns):
            confidence += 0.2
        
        # Deterministic builder bonus
        if is_deterministic:
            confidence += 0.4
        
        # Data returned bonus with smart handling
        if has_data:
            confidence += 0.1
        else:
            # Only penalize if we expect data (not for edge cases)
            if not any(edge_case in user_query.lower() for edge_case in ['today', 'this week', 'recent']):
                confidence -= 0.2  # Reduced penalty for edge cases
        
        # Execution success check
        execution_success = any(v.get("valid", False) for v in validations)
        if execution_success:
            confidence += 0.2
        else:
            confidence -= 0.6
        
        # Schema mismatch penalty
        wrong_patterns = ['production_info.*price', 'pack_waste.*packaging']
        if any(pattern in sql.lower() for pattern in wrong_patterns):
            confidence -= 0.4
        
        # Special confidence boosts for specific query types
        query_lower = user_query.lower()
        
        # Price queries using prices table get extra confidence
        if 'price' in query_lower and 'prices' in sql.lower():
            confidence += 0.2
        
        # Hygiene queries using person_hyg table get extra confidence
        if 'hygiene' in query_lower and 'person_hyg' in sql.lower():
            confidence += 0.2
        
        # Complex analytical queries with proper structure get bonus
        if any(word in query_lower for word in ['analyze', 'correlation', 'trends']) and 'group by' in sql.lower():
            confidence += 0.15
        
        # Price trend queries with proper time filtering get extra confidence
        if 'price trends' in query_lower and 'date_sub' in sql.lower() and 'prices' in sql.lower():
            confidence += 0.3
        
        # Queries with proper date filtering get confidence boost
        if any(time_word in query_lower for time_word in ['last week', 'this month', 'recent']) and 'date' in sql.lower():
            confidence += 0.2
        
        return max(0.0, min(1.0, confidence))

    def process_query(self, user_query: str, prefer_mode: Optional[str] = None, export: Optional[str] = None) -> Dict[str, Any]:
        """Legacy method for backward compatibility. Calls process() and converts result to dict."""
        result = self.process(user_query, prefer_mode=prefer_mode, export=export)
        
        return {
            'mode': result.mode,
            'sql': result.sql,
            'table_markdown': result.table_markdown,
            'short_answer': result.short_answer,
            'analysis': result.analysis,
            'visualization_path': result.visualization_path,
            'metadata': result.metadata
        }
