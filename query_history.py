import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

LOGGER = logging.getLogger(__name__)


@dataclass
class QueryHistoryEntry:
    """Represents a single query history entry."""
    id: str
    timestamp: str
    user_query: str
    sql_query: Optional[str]
    mode: str
    execution_time_ms: Optional[float]
    row_count: Optional[int]
    error: Optional[str]
    is_favorite: bool = False
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class QueryHistoryManager:
    """Manages query history with search, favorites, and statistics."""
    
    def __init__(self, history_file: str = "outputs/query_history.json", max_entries: int = 1000):
        self.history_file = history_file
        self.max_entries = max_entries
        self.history: List[QueryHistoryEntry] = []
        self._ensure_outputs_dir()
        self._load_history()
    
    def _ensure_outputs_dir(self) -> None:
        os.makedirs("outputs", exist_ok=True)
    
    def _load_history(self) -> None:
        """Load query history from file."""
        if not os.path.exists(self.history_file):
            return
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.history = [QueryHistoryEntry(**entry) for entry in data]
        except Exception as e:
            LOGGER.warning(f"Failed to load query history: {e}")
            self.history = []
    
    def _save_history(self) -> None:
        """Save query history to file."""
        try:
            # Keep only the most recent entries
            if len(self.history) > self.max_entries:
                self.history = self.history[-self.max_entries:]
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(entry) for entry in self.history], f, indent=2, ensure_ascii=False)
        except Exception as e:
            LOGGER.warning(f"Failed to save query history: {e}")
    
    def add_entry(
        self,
        user_query: str,
        sql_query: Optional[str],
        mode: str,
        execution_time_ms: Optional[float] = None,
        row_count: Optional[int] = None,
        error: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Add a new query to history and return its ID."""
        entry_id = f"q_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        timestamp = datetime.utcnow().isoformat()
        
        entry = QueryHistoryEntry(
            id=entry_id,
            timestamp=timestamp,
            user_query=user_query,
            sql_query=sql_query,
            mode=mode,
            execution_time_ms=execution_time_ms,
            row_count=row_count,
            error=error,
            tags=tags or []
        )
        
        self.history.append(entry)
        self._save_history()
        return entry_id
    
    def get_entry(self, entry_id: str) -> Optional[QueryHistoryEntry]:
        """Get a specific entry by ID."""
        for entry in self.history:
            if entry.id == entry_id:
                return entry
        return None
    
    def search_history(
        self,
        query: str = "",
        mode: Optional[str] = None,
        has_error: Optional[bool] = None,
        favorites_only: bool = False,
        limit: int = 50
    ) -> List[QueryHistoryEntry]:
        """Search query history with various filters."""
        results = []
        query_lower = query.lower()
        
        for entry in reversed(self.history):  # Most recent first
            # Apply filters
            if mode and entry.mode != mode:
                continue
            if has_error is not None and (entry.error is not None) != has_error:
                continue
            if favorites_only and not entry.is_favorite:
                continue
            
            # Text search in user query and SQL
            if query:
                if (query_lower not in entry.user_query.lower() and
                    (not entry.sql_query or query_lower not in entry.sql_query.lower()) and
                    not any(query_lower in tag.lower() for tag in entry.tags)):
                    continue
            
            results.append(entry)
            if len(results) >= limit:
                break
        
        return results
    
    def toggle_favorite(self, entry_id: str) -> bool:
        """Toggle favorite status of an entry. Returns new favorite status."""
        entry = self.get_entry(entry_id)
        if entry:
            entry.is_favorite = not entry.is_favorite
            self._save_history()
            return entry.is_favorite
        return False
    
    def add_tag(self, entry_id: str, tag: str) -> bool:
        """Add a tag to an entry."""
        entry = self.get_entry(entry_id)
        if entry and tag not in entry.tags:
            entry.tags.append(tag)
            self._save_history()
            return True
        return False
    
    def remove_tag(self, entry_id: str, tag: str) -> bool:
        """Remove a tag from an entry."""
        entry = self.get_entry(entry_id)
        if entry and tag in entry.tags:
            entry.tags.remove(tag)
            self._save_history()
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        total_queries = len(self.history)
        successful_queries = sum(1 for e in self.history if e.error is None)
        failed_queries = total_queries - successful_queries
        
        # Mode distribution
        mode_counts = {}
        for entry in self.history:
            mode_counts[entry.mode] = mode_counts.get(entry.mode, 0) + 1
        
        # Average execution time (for successful queries)
        exec_times = [e.execution_time_ms for e in self.history if e.execution_time_ms is not None]
        avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0
        
        # Recent activity (last 7 days)
        from datetime import timedelta
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        recent_queries = sum(1 for e in self.history 
                           if datetime.fromisoformat(e.timestamp) > recent_cutoff)
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
            "mode_distribution": mode_counts,
            "average_execution_time_ms": round(avg_exec_time, 2),
            "recent_queries_7_days": recent_queries,
            "favorites_count": sum(1 for e in self.history if e.is_favorite)
        }
    
    def get_popular_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most common query patterns."""
        # Simple approach: group by similar user queries
        query_groups = {}
        for entry in self.history:
            # Normalize the query for grouping
            normalized = re.sub(r'\d+', 'N', entry.user_query.lower())
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            if normalized not in query_groups:
                query_groups[normalized] = []
            query_groups[normalized].append(entry)
        
        # Sort by frequency and return top patterns
        popular = sorted(query_groups.items(), key=lambda x: len(x[1]), reverse=True)[:limit]
        
        result = []
        for pattern, entries in popular:
            result.append({
                "pattern": pattern,
                "count": len(entries),
                "example": entries[0].user_query,
                "success_rate": sum(1 for e in entries if e.error is None) / len(entries)
            })
        
        return result
    
    def clear_history(self) -> None:
        """Clear all history."""
        self.history = []
        self._save_history()
    
    def export_history(self, format: str = "json") -> str:
        """Export history to a file. Returns the file path."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        if format == "json":
            filename = f"query_history_export_{timestamp}.json"
            filepath = os.path.join("outputs", "exports", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([asdict(entry) for entry in self.history], f, indent=2, ensure_ascii=False)
        
        elif format == "csv":
            import pandas as pd
            filename = f"query_history_export_{timestamp}.csv"
            filepath = os.path.join("outputs", "exports", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            df = pd.DataFrame([asdict(entry) for entry in self.history])
            df.to_csv(filepath, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return filepath
