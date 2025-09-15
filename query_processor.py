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

    def __init__(self, cache_dir: str = "outputs/cache", ttl_seconds: int = 3600):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        ensure_dirs()

    def _get_cache_key(self, sql: str) -> str:
        """Generate a cache key from SQL query."""
        normalized = re.sub(r"\s+", " ", sql.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def get(self, sql: str) -> Optional[Tuple[pd.DataFrame, str]]:
        """Get cached result if exists and not expired."""
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
    def execute_select(self, sql: str) -> Tuple[pd.DataFrame, str]:
        self.validate_select_only(sql)
        hinted = self._inject_exec_timeout_hint(sql)
        safe_sql = self._clamp_or_inject_limit(hinted)

        # Check cache first
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
        self._load_schema_identifiers()
        # Placeholder for a simple FK graph (table -> list of (col, ref_table, ref_col))
        self.fk_graph: Dict[str, List[Tuple[str, str, str]]] = {}
        self._load_fk_graph()
        # Graph RAG: schema graph (tables, columns, fk paths)
        self.schema_graph: nx.Graph = nx.Graph()
        self._build_schema_graph()

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
            self.schema_graph = g
        except Exception as exc:
            LOGGER.debug("Failed to build schema graph: %s", exc)

    def _retrieve_schema_subgraph(
        self, user_query: str, max_hops: int = 2
    ) -> Dict[str, Any]:
        """Heuristic subgraph retrieval: map keywords to tables/columns, expand by FK hops."""
        try:
            q = user_query.lower()
            # seed nodes by simple name matches
            seeds: List[Tuple[str, ...]] = []
            for node in self.schema_graph.nodes:
                kind = node[0]
                if kind == "table":
                    name = node[1]
                    if name.lower() in q:
                        seeds.append(node)
                elif kind == "column":
                    _, t, c = node
                    if c.lower() in q or t.lower() in q:
                        seeds.append(node)
            if not seeds:
                # fallback: use top 2 table names
                seeds = [n for n in self.schema_graph.nodes if n[0] == "table"][:2]
            # BFS expansion
            included = set(seeds)
            frontier = list(seeds)
            for _ in range(max_hops):
                next_frontier: List[Tuple[str, ...]] = []
                for n in frontier:
                    for m in self.schema_graph.neighbors(n):
                        if m not in included:
                            included.add(m)
                            next_frontier.append(m)
                frontier = next_frontier
            # Collect tables, columns, and fk edges
            tables = sorted({n[1] for n in included if n[0] == "table"})
            columns: Dict[str, List[str]] = {}
            for n in included:
                if n[0] == "column":
                    _, t, c = n
                    columns.setdefault(t, []).append(c)
            fk_edges: List[Tuple[str, str, Tuple[str, str]]] = []
            for u, v, data in self.schema_graph.edges(data=True):
                if u in included and v in included and data.get("kind") == "fk":
                    fk_edges.append((u[1], v[1], tuple(data.get("via") or ("", ""))))
            return {"tables": tables, "columns": columns, "fk_edges": fk_edges}
        except Exception:
            return {"tables": [], "columns": {}, "fk_edges": []}

    def _detect_mode(
        self, user_query: str, context: RetrievedContext, prefer_mode: Optional[str]
    ) -> str:
        if prefer_mode:
            return prefer_mode
        sys_prompt, few_shots = mcp_handler.build_intent_prompt()
        input_block = {
            "question": user_query,
            "context": context.texts[:6],
        }
        prompt = (
            sys_prompt
            + "\n\nExamples:\n"
            + "\n\n".join(few_shots)
            + "\n\nUser:\n"
            + json.dumps(input_block, ensure_ascii=False)
        )
        try:
            resp = self.llm.invoke(prompt)
            raw = resp.content if hasattr(resp, "content") else resp
            txt: str = _stringify_llm_content(raw)
            data = _parse_json_block(txt)
            if data and "mode" in data:
                return data["mode"].strip().upper()
        except Exception as exc:
            LOGGER.warning("Mode detection failed: %s", exc)
        # heuristics fallback
        lower = user_query.lower()
        if any(w in lower for w in ["plot", "chart", "graph", "trend"]):
            return "VISUALIZATION"
        if any(
            w in lower
            for w in [
                "how many",
                "count",
                "total number",
                "avg",
                "average",
                "sum",
                "max",
                "min",
            ]
        ):
            return "SHORT_ANSWER"
        if any(
            w in lower
            for w in ["why", "insight", "explain", "compare", "versus", "vs "]
        ):
            return "ANALYTICAL"
        return "TABLE"

    def _generate_sql(
        self, user_query: str, context: RetrievedContext, mode: str
    ) -> Optional[str]:
        sys_prompt = mcp_handler.build_sql_prompt()
        ctx_snippets = "\n\n".join(context.texts[:8])
        payload = {
            "question": user_query,
            "mode": mode,
            "context": ctx_snippets,
            "allowed_tables": sorted(list(self.allowed_tables))[:100],
            "allowed_columns": sorted(list(self.allowed_columns))[:300],
        }
        prompt = sys_prompt + "\n\n" + json.dumps(payload, ensure_ascii=False)
        try:
            resp = self.llm.invoke(prompt)
            raw = resp.content if hasattr(resp, "content") else resp
            txt: str = _stringify_llm_content(raw)
            sql = _extract_sql_from_text(txt)
            return sql
        except Exception as exc:
            LOGGER.error("SQL generation failed: %s", exc)
            return None

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
        if not ({"employe", "departments"} <= self.allowed_tables):
            return None
        required_cols = {"salary", "department_id", "first_name", "last_name"}
        if not required_cols <= self.allowed_columns:
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
            "  SELECT e.id, e.first_name, e.last_name, e.department_id, e.salary,\n"
            "         ROW_NUMBER() OVER (PARTITION BY e.department_id ORDER BY e.salary DESC) AS salary_rank\n"
            "  FROM `employe` e\n"
            ") ranked\n"
            "JOIN `departments` d ON ranked.department_id = d.id\n"
            f"WHERE ranked.salary_rank <= {k}\n"
            "ORDER BY d.name, ranked.salary_rank\n"
            f"LIMIT {limit_val}"
        )
        return sql

    def _maybe_build_employee_profile(self, user_query: str) -> Optional[str]:
        """Deterministic aggregated profile for queries like 'analyze employee id 1363'.

        Produces a single-row summary to avoid row-per-skill duplication.
        """
        q = user_query.lower()
        m = re.search(r"employee\s+id\s+(\d+)", q)
        if not m:
            return None
        employee_id = int(m.group(1))
        if employee_id <= 0:
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
            f"WHERE e.id = {employee_id}\n"
            "GROUP BY e.id, e.first_name, e.last_name, d.name\n"
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
        if intent == "scalar":
            if has_group:
                return False, "Scalar intent should not have GROUP BY"
            if not is_scalar:
                return False, "Scalar intent requires aggregate"
        if intent == "distribution":
            if not has_group:
                return False, "Distribution requires GROUP BY"
        if intent == "top/most":
            if not (has_order and has_limit):
                return False, "Top/most requires ORDER BY and LIMIT"
        return True, "ok"

    def _generate_candidates(
        self, user_query: str, context: RetrievedContext, mode: str, k: int = 3
    ) -> List[str]:
        candidates: List[str] = []
        # -1) Deterministic aggregated employee profile when explicitly asked
        profile_sql = self._maybe_build_employee_profile(user_query)
        if profile_sql:
            candidates.append(profile_sql)
        # 0) Deterministic heuristic candidate for per-group top-k
        heuristic = self._maybe_build_per_group_topk(user_query)
        if heuristic:
            candidates.append(heuristic)
        # 1..k) LLM candidates
        for i in range(k):
            payload_hint = f"Candidate:{i + 1}"
            sys_prompt = mcp_handler.build_sql_prompt()
            ctx_snippets = "\n\n".join(context.texts[:8])
            # Graph RAG context
            graph_ctx = self._retrieve_schema_subgraph(user_query)
            payload = {
                "question": user_query,
                "mode": mode,
                "context": ctx_snippets,
                "allowed_tables": sorted(list(self.allowed_tables))[:100],
                "allowed_columns": sorted(list(self.allowed_columns))[:300],
                "graph": graph_ctx,
                "hint": payload_hint,
            }
            prompt = sys_prompt + "\n\n" + json.dumps(payload, ensure_ascii=False)
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

    def _short_answer_from_df(self, df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "No result"
        # prioritize a column named value or count
        for name in df.columns:
            if name.lower() in {"value", "count", "total", "sum", "avg", "average"}:
                return str(df[name].iloc[0])
        return str(df.iloc[0, 0])

    def _maybe_visualize(
        self, df: pd.DataFrame, title: str, user_query: str = ""
    ) -> Optional[str]:
        if df is None or df.empty or df.shape[1] < 2:
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
                    colors = plt.cm.Set3(range(len(df)))
                    bars = ax.bar(
                        df[x_col].astype(str),
                        df[y_candidates[0]],
                        color=colors,
                        alpha=0.8,
                    )
                    ax.tick_params(axis="x", labelrotation=45)
                    # Add value labels on bars if not too many
                    if len(df) <= 15:
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
        if len(df) <= 10 and not pd.api.types.is_numeric_dtype(df[x_col]):
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
        context = self.vector_manager.similarity_search(user_query, top_k=8)
        mode = self._detect_mode(user_query, context, override_mode or prefer_mode)
        # Step 1: Plan JSON
        plan = self._plan_intent(user_query)
        # Step 3: Candidate generation (k-best)
        candidates = self._generate_candidates(user_query, context, mode, k=3)
        validations: List[Dict[str, Any]] = []
        chosen_sql: Optional[str] = None
        # Step 4: Validate each candidate
        for c in candidates:
            validations.append(self._validate_candidate(plan, c))
        # Step 5: Light reranking + execution-guided selection
        # Rank by score desc, then try execution for top-N
        ranked = sorted(validations, key=lambda v: v.get("score", 0), reverse=True)
        for v in ranked[:3]:
            if not v.get("valid"):
                continue
            try:
                test_df, test_sql = self.safe_exec.execute_select(
                    v["sql"]
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
                df, executed_sql = self.safe_exec.execute_select(sql)
            except Exception as exc:
                error_msg = str(exc)
                LOGGER.warning("Initial SQL failed, attempting repair: %s", exc)
                # single-shot repair attempt
                repair_prompt = mcp_handler.build_sql_repair_prompt()
                payload = {
                    "question": user_query,
                    "failed_sql": sql,
                    "error": str(exc),
                    "context": context.texts[:6],
                }
                try:
                    resp = self.llm.invoke(
                        repair_prompt + "\n\n" + json.dumps(payload, ensure_ascii=False)
                    )
                    raw = resp.content if hasattr(resp, "content") else resp
                    txt: str = _stringify_llm_content(raw)
                    repaired_sql = _extract_sql_from_text(txt)
                    if repaired_sql:
                        df, executed_sql = self.safe_exec.execute_select(repaired_sql)
                        sql = repaired_sql
                        error_msg = None  # Clear error since repair succeeded
                except Exception as exc2:
                    LOGGER.error("SQL repair failed: %s", exc2)
                    error_msg = f"Original error: {exc}. Repair failed: {exc2}"

        table_md = self._format_table(df) if df is not None else None
        short_answer = self._short_answer_from_df(df) if df is not None else None
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
            row_count=len(df) if df is not None else 0,
            error=error_msg,
        )

        # Step 6: Confidence & artifacts
        num_valid = sum(1 for v in validations if v.get("valid"))
        confidence = 0.33 * num_valid
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
                "row_count": int(len(df)) if df is not None else 0,
                "execution_time_ms": execution_time_ms,
                "history_id": history_id,
                "confidence": confidence,
                "forecast_path": forecast_path,
            },
        )
