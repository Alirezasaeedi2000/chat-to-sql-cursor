#!/usr/bin/env python3
"""
Enhanced Database Manager - Improved connection handling and schema optimization
for better performance with different database types.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

LOGGER = logging.getLogger(__name__)

@dataclass
class DatabaseInfo:
    """Information about a database connection"""
    db_type: str
    version: str
    tables_count: int
    total_rows: int
    connection_string: str
    features: List[str]

class EnhancedDatabaseManager:
    """Enhanced database manager with optimized connection handling"""
    
    def __init__(self):
        self.connection_cache = {}
        self.db_info_cache = {}
        self.optimization_configs = {
            'mysql': {
                'pool_size': 10,
                'max_overflow': 20,
                'pool_recycle': 1800,
                'pool_pre_ping': True,
                'connect_args': {
                    'connect_timeout': 10,
                    'charset': 'utf8mb4',
                    'autocommit': False
                }
            },
            'postgresql': {
                'pool_size': 8,
                'max_overflow': 15,
                'pool_recycle': 1800,
                'pool_pre_ping': True,
                'connect_args': {
                    'connect_timeout': 10,
                    'application_name': 'nl2sql_assistant'
                }
            },
            'sqlite': {
                'pool_size': 5,
                'max_overflow': 10,
                'pool_recycle': 3600,
                'pool_pre_ping': False,
                'connect_args': {
                    'check_same_thread': False,
                    'timeout': 30
                }
            },
            'oracle': {
                'pool_size': 6,
                'max_overflow': 12,
                'pool_recycle': 1800,
                'pool_pre_ping': True,
                'connect_args': {
                    'connect_timeout': 10
                }
            },
            'sqlserver': {
                'pool_size': 8,
                'max_overflow': 16,
                'pool_recycle': 1800,
                'pool_pre_ping': True,
                'connect_args': {
                    'connect_timeout': 10,
                    'timeout': 30
                }
            }
        }
    
    def detect_database_type(self, connection_string: str) -> str:
        """Detect database type from connection string"""
        conn_lower = connection_string.lower()
        
        if 'mysql' in conn_lower or 'pymysql' in conn_lower:
            return 'mysql'
        elif 'postgresql' in conn_lower or 'postgres' in conn_lower:
            return 'postgresql'
        elif 'sqlite' in conn_lower:
            return 'sqlite'
        elif 'oracle' in conn_lower:
            return 'oracle'
        elif 'sqlserver' in conn_lower or 'mssql' in conn_lower:
            return 'sqlserver'
        else:
            return 'mysql'  # Default fallback
    
    def create_optimized_engine(self, connection_string: str, cache_key: Optional[str] = None) -> Engine:
        """Create optimized engine with caching"""
        if cache_key and cache_key in self.connection_cache:
            LOGGER.info(f"Using cached connection for {cache_key}")
            return self.connection_cache[cache_key]
        
        db_type = self.detect_database_type(connection_string)
        config = self.optimization_configs.get(db_type, self.optimization_configs['mysql'])
        
        try:
            engine = create_engine(
                connection_string,
                **config,
                echo=False  # Set to True for SQL debugging
            )
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            LOGGER.info(f"Successfully connected to {db_type} database")
            
            # Cache the connection
            if cache_key:
                self.connection_cache[cache_key] = engine
            
            return engine
            
        except Exception as e:
            LOGGER.error(f"Failed to create engine for {db_type}: {e}")
            raise
    
    def get_database_info(self, engine: Engine) -> DatabaseInfo:
        """Get comprehensive database information"""
        try:
            inspector = inspect(engine)
            db_type = self.detect_database_type(str(engine.url))
            
            # Get database version
            version = self._get_database_version(engine, db_type)
            
            # Get table information
            tables = inspector.get_table_names()
            tables_count = len(tables)
            
            # Estimate total rows
            total_rows = self._estimate_total_rows(engine, tables)
            
            # Get database features
            features = self._get_database_features(engine, db_type)
            
            return DatabaseInfo(
                db_type=db_type,
                version=version,
                tables_count=tables_count,
                total_rows=total_rows,
                connection_string=str(engine.url),
                features=features
            )
            
        except Exception as e:
            LOGGER.error(f"Failed to get database info: {e}")
            raise
    
    def _get_database_version(self, engine: Engine, db_type: str) -> str:
        """Get database version"""
        version_queries = {
            'mysql': "SELECT VERSION()",
            'postgresql': "SELECT version()",
            'sqlite': "SELECT sqlite_version()",
            'oracle': "SELECT * FROM v$version WHERE rownum = 1",
            'sqlserver': "SELECT @@VERSION"
        }
        
        query = version_queries.get(db_type, "SELECT 1")
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                version = str(result.fetchone()[0])
                return version
        except Exception:
            return "Unknown"
    
    def _estimate_total_rows(self, engine: Engine, tables: List[str]) -> int:
        """Estimate total rows across all tables"""
        total_rows = 0
        
        for table in tables[:10]:  # Limit to first 10 tables for performance
            try:
                with engine.connect() as conn:
                    # Try to get row count efficiently
                    result = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`"))
                    count = result.fetchone()[0]
                    total_rows += count
            except Exception:
                # If COUNT fails, estimate based on sample
                try:
                    with engine.connect() as conn:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM `{table}` LIMIT 1000"))
                        # Rough estimate
                        total_rows += 1000
                except Exception:
                    continue
        
        return total_rows
    
    def _get_database_features(self, engine: Engine, db_type: str) -> List[str]:
        """Get database-specific features"""
        features = []
        
        try:
            with engine.connect() as conn:
                if db_type == 'mysql':
                    # Check for MySQL features
                    result = conn.execute(text("SHOW VARIABLES LIKE 'version_comment'"))
                    if result.fetchone():
                        features.append("MySQL")
                    
                    # Check for JSON support
                    try:
                        conn.execute(text("SELECT JSON_OBJECT('test', 1)"))
                        features.append("JSON")
                    except:
                        pass
                
                elif db_type == 'postgresql':
                    # Check for PostgreSQL features
                    features.append("PostgreSQL")
                    
                    # Check for JSON support
                    try:
                        conn.execute(text("SELECT '{}'::json"))
                        features.append("JSON")
                    except:
                        pass
                
                elif db_type == 'sqlite':
                    features.append("SQLite")
                    
                    # Check for JSON support
                    try:
                        conn.execute(text("SELECT json('{}')"))
                        features.append("JSON")
                    except:
                        pass
        
        except Exception as e:
            LOGGER.debug(f"Could not determine features: {e}")
        
        return features
    
    def optimize_for_database(self, engine: Engine) -> Dict[str, Any]:
        """Get database-specific optimization recommendations"""
        db_info = self.get_database_info(engine)
        recommendations = []
        
        if db_info.db_type == 'mysql':
            recommendations.extend([
                "Use backticks around identifiers",
                "Consider using InnoDB engine for better performance",
                "Enable query cache if available"
            ])
        
        elif db_info.db_type == 'postgresql':
            recommendations.extend([
                "Use double quotes for identifiers",
                "Consider using EXPLAIN ANALYZE for query optimization",
                "Use appropriate indexes for better performance"
            ])
        
        elif db_info.db_type == 'sqlite':
            recommendations.extend([
                "Use PRAGMA statements for optimization",
                "Consider WAL mode for better concurrency",
                "Use appropriate indexes for better performance"
            ])
        
        return {
            "database_info": db_info,
            "recommendations": recommendations,
            "optimization_settings": self.optimization_configs.get(db_info.db_type, {})
        }
    
    def test_connection_performance(self, engine: Engine) -> Dict[str, float]:
        """Test connection performance metrics"""
        metrics = {}
        
        # Test basic query performance
        start_time = time.time()
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            metrics['basic_query_time'] = time.time() - start_time
        except Exception as e:
            LOGGER.error(f"Basic query test failed: {e}")
            metrics['basic_query_time'] = -1
        
        # Test schema inspection performance
        start_time = time.time()
        try:
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            metrics['schema_inspection_time'] = time.time() - start_time
            metrics['tables_found'] = len(tables)
        except Exception as e:
            LOGGER.error(f"Schema inspection test failed: {e}")
            metrics['schema_inspection_time'] = -1
            metrics['tables_found'] = 0
        
        return metrics
    
    def clear_cache(self):
        """Clear connection cache"""
        self.connection_cache.clear()
        self.db_info_cache.clear()
        LOGGER.info("Database connection cache cleared")

# Enhanced vector store manager for better schema indexing
class EnhancedVectorStoreManager:
    """Enhanced vector store manager with database-specific optimizations"""
    
    def __init__(self, persist_directory: str = ".vector_store", collection_name: str = "nl2sql"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.db_manager = EnhancedDatabaseManager()
    
    def create_database_specific_vector_store(self, connection_string: str) -> str:
        """Create database-specific vector store directory"""
        db_type = self.db_manager.detect_database_type(connection_string)
        
        # Create unique directory for this database
        db_hash = str(hash(connection_string))[:8]
        db_specific_dir = f"{self.persist_directory}_{db_type}_{db_hash}"
        
        os.makedirs(db_specific_dir, exist_ok=True)
        
        LOGGER.info(f"Created database-specific vector store: {db_specific_dir}")
        return db_specific_dir
    
    def get_optimized_sample_size(self, engine: Engine, table_name: str) -> int:
        """Get optimized sample size based on table characteristics"""
        try:
            with engine.connect() as conn:
                # Get table row count
                result = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`"))
                row_count = result.fetchone()[0]
                
                # Optimize sample size based on table size
                if row_count < 100:
                    return row_count  # Use all rows for small tables
                elif row_count < 1000:
                    return min(50, row_count)  # 50 rows for medium tables
                else:
                    return min(100, row_count)  # 100 rows for large tables
                    
        except Exception:
            return 5  # Default fallback
    
    def index_database_with_optimization(self, connection_string: str) -> Dict[str, Any]:
        """Index database with optimization for better performance"""
        engine = self.db_manager.create_optimized_engine(connection_string)
        
        # Get database optimization info
        optimization_info = self.db_manager.optimize_for_database(engine)
        
        # Create database-specific vector store
        vector_dir = self.create_database_specific_vector_store(connection_string)
        
        # Import and use the existing VectorStoreManager
        from vector import VectorStoreManager
        vector_manager = VectorStoreManager(
            persist_directory=vector_dir,
            collection_name=f"{self.collection_name}_{optimization_info['database_info'].db_type}"
        )
        
        # Index with optimized settings
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        indexed_items = 0
        for table in tables:
            try:
                # Get optimized sample size for each table
                sample_size = self.get_optimized_sample_size(engine, table)
                
                # Index table schema and samples
                table_items = vector_manager.upsert_schema_and_samples(
                    engine, 
                    sample_rows_per_table=sample_size
                )
                indexed_items += table_items
                
                LOGGER.info(f"Indexed table {table} with {sample_size} sample rows")
                
            except Exception as e:
                LOGGER.error(f"Failed to index table {table}: {e}")
        
        return {
            "indexed_items": indexed_items,
            "tables_indexed": len(tables),
            "vector_store_path": vector_dir,
            "database_info": optimization_info['database_info'],
            "recommendations": optimization_info['recommendations']
        }

def setup_enhanced_database_connection(connection_string: str) -> Tuple[Engine, Dict[str, Any]]:
    """Setup enhanced database connection with optimization"""
    db_manager = EnhancedDatabaseManager()
    
    # Create optimized engine
    engine = db_manager.create_optimized_engine(connection_string)
    
    # Get database information
    db_info = db_manager.get_database_info(engine)
    
    # Test performance
    performance_metrics = db_manager.test_connection_performance(engine)
    
    # Get optimization recommendations
    optimization_info = db_manager.optimize_for_database(engine)
    
    return engine, {
        "database_info": db_info,
        "performance_metrics": performance_metrics,
        "optimization_info": optimization_info
    }

if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with different database types
    test_connections = [
        "mysql+pymysql://root:@localhost:3306/farnan",
        "sqlite:///demo_test.db",
        # "postgresql://postgres:password@localhost:5432/test_db"
    ]
    
    for conn_string in test_connections:
        try:
            print(f"\n🔍 Testing connection: {conn_string}")
            engine, info = setup_enhanced_database_connection(conn_string)
            
            print(f"✅ Connected to {info['database_info'].db_type}")
            print(f"   Version: {info['database_info'].version}")
            print(f"   Tables: {info['database_info'].tables_count}")
            print(f"   Features: {', '.join(info['database_info'].features)}")
            print(f"   Performance: {info['performance_metrics']}")
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
