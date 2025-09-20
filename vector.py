import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


LOGGER = logging.getLogger(__name__)


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


@dataclass
class RetrievedContext:
    texts: List[str]
    metadatas: List[Dict[str, Any]]


class QueryExpander:
    """Lightweight synonym expansion to improve retrieval without changing SQL generation semantics."""

    def __init__(self, synonyms_path: str) -> None:
        self.synonyms_path = synonyms_path
        self.synonyms: Dict[str, List[str]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.isfile(self.synonyms_path):
            try:
                with open(self.synonyms_path, "r", encoding="utf-8") as f:
                    self.synonyms = json.load(f)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to load synonyms file %s: %s", self.synonyms_path, exc
                )
                self.synonyms = {}

    def expand(self, query_text: str) -> str:
        """Append synonym hints to the query string to enrich retrieval context."""
        if not self.synonyms:
            return query_text
        hints: List[str] = []
        lowered = query_text.lower()
        for canonical, syns in self.synonyms.items():
            if canonical.lower() in lowered:
                hints.extend(syns)
        if hints:
            return f"{query_text}\n\nAlso consider related terms: {', '.join(sorted(set(hints)))}"
        return query_text


class VectorStoreManager:
    """Manages embedding, persistence, and retrieval for schema, samples, and past queries."""

    def __init__(
        self,
        persist_directory: str = ".vector_store",
        collection_name: str = "mysql_nl2sql",
        embedding_model: str = "mxbai-embed-large",
        synonyms_path: str = os.path.join("outputs", "synonyms.json"),
    ) -> None:
        ensure_dir(persist_directory)
        ensure_dir("outputs")
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self.query_expander = QueryExpander(synonyms_path)
        
        # Column mapping fallback for schema mismatches (corrected table names)
        self.column_mappings = {
            'packs': {
                'dateDate': 'date',
            },
            'pack_waste': {
                'dateDate': 'date',
            },
            'person_hyg': {
                'per_hy': 'per_hy_id',
            },
            'prices': {
                'dateDate': 'date',
            },
            'production_info': {
                'bakeID': 'bakeID',
            },
            'production_test': {
                'bakeID': 'bakeID',
            },
            'users': {
                'userId': 'userId',
            },
        }

    @property
    def vector_store(self) -> Chroma:
        return self._vector_store
    
    def get_column_mapping(self, table_name: str, column_name: str) -> Optional[str]:
        """Get column mapping fallback for schema mismatches."""
        if table_name in self.column_mappings:
            return self.column_mappings[table_name].get(column_name)
        return None
    
    def suggest_column_mapping(self, table_name: str, invalid_column: str) -> Optional[str]:
        """Suggest column mapping using Levenshtein distance for similar column names."""
        if table_name not in self.column_mappings:
            return None
        
        # Simple similarity matching
        mappings = self.column_mappings[table_name]
        for mapped_col, actual_col in mappings.items():
            if invalid_column.lower() in mapped_col.lower() or mapped_col.lower() in invalid_column.lower():
                return actual_col
        return None

    def _schema_snippet(self, engine: Engine, table_name: str) -> str:
        inspector = inspect(engine)
        cols = inspector.get_columns(table_name)
        pk = inspector.get_pk_constraint(table_name)
        pk_cols = pk.get("constrained_columns") or []
        
        # Enhanced column descriptions with business context
        col_desc = []
        for c in cols:
            col_name = c['name']
            col_type = str(c.get('type'))
            col_info = f"{col_name}({col_type})"
            
            # Add business context annotations
            if 'id' in col_name.lower() and col_name.lower() != 'id':
                col_info += " [identifier]"
            elif any(word in col_name.lower() for word in ['name', 'title', 'description']):
                col_info += " [text]"
            elif any(word in col_name.lower() for word in ['date', 'time', 'created', 'updated']):
                col_info += " [datetime]"
            elif any(word in col_name.lower() for word in ['price', 'cost', 'amount', 'total', 'sum']):
                col_info += " [monetary]"
            elif any(word in col_name.lower() for word in ['count', 'quantity', 'number', 'qty']):
                col_info += " [numeric]"
            elif any(word in col_name.lower() for word in ['status', 'type', 'category', 'level']):
                col_info += " [categorical]"
            
            if not c.get('nullable', True):
                col_info += " NOT NULL"
            if c.get('default') is not None:
                col_info += f" DEFAULT {c['default']}"
            col_desc.append(col_info)
        
        snippet = (
            f"Table `{table_name}` columns: {', '.join(col_desc)}. "
            f"Primary key: {', '.join(pk_cols) if pk_cols else 'None'}."
        )
        
        # Enhanced foreign keys with business relationship semantics
        fks = inspector.get_foreign_keys(table_name)
        if fks:
            fk_desc = []
            for fk in fks:
                constrained = ','.join(fk.get('constrained_columns', []))
                referred_table = fk.get('referred_table')
                referred_cols = ','.join(fk.get('referred_columns', []))
                
                # Add business relationship context
                relationship = f"{constrained} -> {referred_table}({referred_cols})"
                
                # Business relationship mapping for Farnan database (corrected table names)
                if table_name == 'production_info' and referred_table == 'workers':
                    relationship += f" [production performed by worker]"
                elif table_name == 'production_test' and referred_table == 'workers':
                    relationship += f" [test conducted by worker]"
                elif table_name == 'person_hyg' and referred_table == 'workers':
                    relationship += f" [hygiene check performed by worker]"
                elif table_name == 'packaging_info' and referred_table == 'workers':
                    relationship += f" [packaging handled by worker]"
                elif table_name == 'pack_waste' and referred_table == 'workers':
                    relationship += f" [waste generated by worker]"
                elif table_name == 'prices' and referred_table == 'workers':
                    relationship += f" [price updated by worker]"
                elif 'id' in constrained.lower():
                    relationship += f" [many-to-one: {table_name} belongs to {referred_table}]"
                
                fk_desc.append(relationship)
            snippet += f" Foreign keys: {'; '.join(fk_desc)}."
        
        # Indexes for performance hints
        try:
            indexes = inspector.get_indexes(table_name)
            if indexes:
                idx_desc = []
                for idx in indexes:
                    idx_name = idx.get('name', 'unnamed')
                    idx_cols = ','.join(idx.get('column_names', []))
                    unique = " UNIQUE" if idx.get('unique') else ""
                    idx_desc.append(f"{idx_name}({idx_cols}){unique}")
                snippet += f" Indexes: {'; '.join(idx_desc)}."
        except Exception:
            # Some databases don't support index inspection
            pass
        
        # Unique constraints
        try:
            unique_constraints = inspector.get_unique_constraints(table_name)
            if unique_constraints:
                uc_desc = []
                for uc in unique_constraints:
                    uc_cols = ','.join(uc.get('column_names', []))
                    uc_desc.append(f"UNIQUE({uc_cols})")
                snippet += f" Unique constraints: {'; '.join(uc_desc)}."
        except Exception:
            pass
            
        return snippet

    def _sample_rows_snippet(
        self, engine: Engine, table_name: str, sample_rows: int
    ) -> Tuple[str, Optional[pd.DataFrame]]:
        try:
            with engine.connect() as conn:
                df = pd.read_sql(
                    text(f"SELECT * FROM `{table_name}` LIMIT :lim"),
                    conn,
                    params={"lim": sample_rows},
                )
            if df.empty:
                return f"No sample rows available for `{table_name}`.", df
            
            # Enhanced analysis with data patterns
            head = df.head(min(sample_rows, 5))
            sample_text = head.to_csv(index=False)
            
            # Analyze data patterns and distributions
            patterns = []
            for col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                    
                # Data type analysis
                if pd.api.types.is_numeric_dtype(col_data):
                    patterns.append(f"{col}: numeric range [{col_data.min():.2f} to {col_data.max():.2f}]")
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    patterns.append(f"{col}: dates from {col_data.min()} to {col_data.max()}")
                elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                    unique_count = col_data.nunique()
                    total_count = len(col_data)
                    if unique_count == total_count:
                        patterns.append(f"{col}: unique values (likely identifier)")
                    elif unique_count <= 10:
                        sample_values = list(col_data.unique()[:5])
                        patterns.append(f"{col}: categorical [{unique_count} values: {sample_values}...]")
                    else:
                        patterns.append(f"{col}: text data [{unique_count}/{total_count} unique]")
            
            pattern_text = "\nData patterns: " + "; ".join(patterns) if patterns else ""
            
            return (
                f"Sample rows for `{table_name}` (CSV):\n{sample_text}{pattern_text}",
                df
            )
        except Exception as exc:
            LOGGER.warning("Failed to sample rows for %s: %s", table_name, exc)
            return f"Failed to sample rows for `{table_name}` due to error.", None

    def upsert_schema_and_samples(
        self, engine: Engine, sample_rows_per_table: int = 5, include_views: bool = True
    ) -> int:
        """Indexes schema metadata and sample rows into the vector store. Returns number of items added."""
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if include_views:
            try:
                tables.extend(
                    [v for v in inspector.get_view_names() if v not in tables]
                )
            except Exception:
                pass

        added = 0
        for table_name in tables:
            try:
                # Delete previous entries for the table to avoid duplication.
                try:
                    self._vector_store.delete(where={"table": table_name})
                except Exception:
                    pass

                schema_text = self._schema_snippet(engine, table_name)
                self._vector_store.add_texts(
                    texts=[schema_text],
                    metadatas=[{"type": "schema", "table": table_name}],
                )
                added += 1

                sample_text, _ = self._sample_rows_snippet(
                    engine, table_name, sample_rows_per_table
                )
                self._vector_store.add_texts(
                    texts=[sample_text],
                    metadatas=[{"type": "sample", "table": table_name}],
                )
                added += 1
            except Exception as exc:
                LOGGER.error("Failed to index table %s: %s", table_name, exc)
        try:
            # Ensure persistence on disk
            self._vector_store.persist()
        except Exception:
            pass
        return added

    def add_past_query(
        self, sql: str, result_summary: Optional[str] = None, note: Optional[str] = None
    ) -> None:
        text_block = f"SQL: {sql}"
        if result_summary:
            text_block += f"\nResult: {result_summary}"
        if note:
            text_block += f"\nNote: {note}"
        try:
            self._vector_store.add_texts(
                texts=[text_block],
                metadatas=[{"type": "past_query"}],
            )
            self._vector_store.persist()
        except Exception as exc:
            LOGGER.warning("Failed to add past query to vector store: %s", exc)

    def similarity_search(self, query_text: str, top_k: int = 8) -> RetrievedContext:
        enriched = self.query_expander.expand(query_text)
        try:
            docs = self._vector_store.similarity_search(enriched, k=top_k)
        except Exception as exc:
            LOGGER.warning("Vector search failed, returning empty context: %s", exc)
            return RetrievedContext(texts=[], metadatas=[])
        texts = [d.page_content for d in docs]
        metas = [d.metadata or {} for d in docs]
        return RetrievedContext(texts=texts, metadatas=metas)
