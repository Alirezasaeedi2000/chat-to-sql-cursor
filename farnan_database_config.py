#!/usr/bin/env python3
"""
Optimized Configuration for Farnan Database
Production-ready settings for food production/quality control system
"""

import os
from datetime import datetime
from typing import Dict, Any, List

# Database Configuration
FARNAN_DATABASE_CONFIG = {
    "connection_string": "mysql+pymysql://root:@localhost:3306/Farnan?charset=utf8mb4&autocommit=true",
    "pool_settings": {
        "pool_size": 20,
        "max_overflow": 30,
        "pool_pre_ping": True,
        "pool_recycle": 3600,
        "pool_timeout": 30,
        "echo": False
    },
    "query_timeout": 30,
    "connection_timeout": 10
}

# Schema Metadata for NLP-to-SQL
FARNAN_SCHEMA_METADATA = {
    "database_name": "Farnan",
    "domain": "Food Production & Quality Control",
    "description": "Production management system for food manufacturing with quality control, hygiene tracking, and waste management",
    
    "tables": {
        "production_info": {
            "description": "Main production records with batch information and quality metrics",
            "business_meaning": "Core production data tracking batches, dates, and production metrics",
            "key_columns": ["date", "batch", "product_type"],
            "semantic_tags": ["production", "batch", "quality", "manufacturing"]
        },
        "person_hyg": {
            "description": "Personnel hygiene compliance tracking for food safety",
            "business_meaning": "Staff hygiene monitoring to ensure food safety standards",
            "key_columns": ["personName", "date", "beard", "nail", "handLeg", "robe"],
            "semantic_tags": ["hygiene", "personnel", "safety", "compliance", "staff"]
        },
        "packaging_info": {
            "description": "Packaging details including weights, counts, and batch information",
            "business_meaning": "Packaging process tracking with weights and quality metrics",
            "key_columns": ["Packkey", "date", "bakeDate", "TotalWeight", "tranWeight"],
            "semantic_tags": ["packaging", "weight", "batch", "packaging_process"]
        },
        "pack_waste": {
            "description": "Packaging waste tracking and analysis",
            "business_meaning": "Waste management and reduction tracking for packaging materials",
            "key_columns": ["date", "type", "value"],
            "semantic_tags": ["waste", "packaging", "environment", "reduction"]
        },
        "production_test": {
            "description": "Quality testing results for production batches",
            "business_meaning": "Quality control testing data for production validation",
            "key_columns": ["date", "test_type", "result"],
            "semantic_tags": ["testing", "quality", "validation", "compliance"]
        },
        "packs": {
            "description": "Packaging specifications and configurations",
            "business_meaning": "Packaging material specifications and usage",
            "key_columns": ["date", "type", "carton", "packet"],
            "semantic_tags": ["packaging", "specifications", "materials", "configuration"]
        },
        "prices": {
            "description": "Ingredient and material pricing information",
            "business_meaning": "Cost tracking for ingredients and raw materials",
            "key_columns": ["date", "NC", "ricotta", "buttermilkPowder", "cream"],
            "semantic_tags": ["pricing", "ingredients", "cost", "materials"]
        },
        "workers": {
            "description": "Employee information and workforce management",
            "business_meaning": "Staff directory and workforce tracking",
            "key_columns": ["worker_id", "name", "department", "position"],
            "semantic_tags": ["employees", "staff", "workforce", "personnel"]
        },
        "users": {
            "description": "System user accounts and access control",
            "business_meaning": "User authentication and system access management",
            "key_columns": ["user_id", "username", "role", "permissions"],
            "semantic_tags": ["users", "authentication", "access", "security"]
        }
    },
    
    "business_relationships": {
        "production_flow": [
            "production_info → packaging_info",
            "packaging_info → packs",
            "production_info → production_test"
        ],
        "quality_control": [
            "person_hyg → production_info",
            "production_test → production_info",
            "pack_waste → packaging_info"
        ],
        "cost_management": [
            "prices → production_info",
            "pack_waste → prices"
        ]
    },
    
    "common_queries": {
        "production_analytics": [
            "Show me production volumes by date",
            "What are the quality test results for recent batches?",
            "Which products have the highest production rates?"
        ],
        "hygiene_compliance": [
            "Who failed hygiene checks today?",
            "Show hygiene compliance rates by person",
            "What are the most common hygiene violations?"
        ],
        "waste_management": [
            "How much packaging waste was generated this week?",
            "Which packaging types generate the most waste?",
            "Show waste trends over time"
        ],
        "cost_analysis": [
            "What are the current ingredient prices?",
            "Show cost breakdown by ingredient",
            "Which ingredients have the highest costs?"
        ]
    }
}

# Enhanced Synonyms for Better NLP Understanding
FARNAN_SYNONYMS = {
    "production": ["manufacturing", "batch", "output", "production run"],
    "hygiene": ["cleanliness", "sanitation", "safety", "compliance"],
    "packaging": ["wrapping", "packing", "containers", "materials"],
    "waste": ["scrap", "discard", "trash", "rejection"],
    "quality": ["standard", "grade", "condition", "specification"],
    "batch": ["lot", "group", "run", "production batch"],
    "weight": ["mass", "load", "heaviness"],
    "date": ["time", "day", "period", "when"],
    "person": ["employee", "worker", "staff", "personnel"],
    "test": ["check", "examination", "validation", "inspection"]
}

def get_optimized_connection_string() -> str:
    """Get optimized connection string for Farnan database"""
    return FARNAN_DATABASE_CONFIG["connection_string"]

def get_enhanced_schema_context() -> str:
    """Generate enhanced schema context for NLP-to-SQL"""
    context = f"""
    DATABASE: {FARNAN_SCHEMA_METADATA['database_name']}
    DOMAIN: {FARNAN_SCHEMA_METADATA['domain']}
    DESCRIPTION: {FARNAN_SCHEMA_METADATA['description']}
    
    TABLES AND THEIR BUSINESS MEANING:
    """
    
    for table_name, info in FARNAN_SCHEMA_METADATA['tables'].items():
        context += f"""
    - {table_name.upper()}: {info['description']}
      Business Purpose: {info['business_meaning']}
      Key Columns: {', '.join(info['key_columns'])}
      Tags: {', '.join(info['semantic_tags'])}
        """
    
    context += f"""
    
    BUSINESS RELATIONSHIPS:
    """
    for relationship_type, relationships in FARNAN_SCHEMA_METADATA['business_relationships'].items():
        context += f"\n    {relationship_type.upper()}:"
        for rel in relationships:
            context += f"\n      - {rel}"
    
    return context

def get_domain_specific_prompts() -> Dict[str, str]:
    """Get domain-specific prompts for better SQL generation"""
    return {
        "production_analysis": """
        You are analyzing a food production database. Focus on:
        - Production volumes, dates, and batch tracking
        - Quality metrics and compliance
        - Efficiency and performance indicators
        - Time-series analysis for production trends
        """,
        
        "hygiene_compliance": """
        You are analyzing hygiene compliance data. Focus on:
        - Staff hygiene check results
        - Compliance rates and violations
        - Safety standards and requirements
        - Personnel tracking and accountability
        """,
        
        "waste_management": """
        You are analyzing waste management data. Focus on:
        - Waste volumes and types
        - Reduction opportunities
        - Environmental impact metrics
        - Cost implications of waste
        """,
        
        "cost_analysis": """
        You are analyzing cost and pricing data. Focus on:
        - Ingredient and material costs
        - Cost trends over time
        - Budget planning and forecasting
        - Cost optimization opportunities
        """
    }

if __name__ == "__main__":
    print("🏭 FARNAN DATABASE CONFIGURATION")
    print("=" * 50)
    print(f"Connection String: {get_optimized_connection_string()}")
    print(f"\nSchema Context:\n{get_enhanced_schema_context()}")
    print(f"\nDomain Prompts Available: {list(get_domain_specific_prompts().keys())}")
