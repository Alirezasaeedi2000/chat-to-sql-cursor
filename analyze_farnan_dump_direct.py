#!/usr/bin/env python3
"""
Direct Farnan SQL Dump Analysis Script
Analyzes the MySQL dump directly without conversion to SQLite.
"""

import re
import json
import os
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Any, Tuple, Set

class DirectFarnanAnalyzer:
    def __init__(self, dump_file: str):
        self.dump_file = dump_file
        self.schema_info = {}
        self.data_patterns = {}
        self.relationships = {}
        self.insights = {}
        
    def analyze(self) -> Dict[str, Any]:
        """Main analysis function"""
        print("🔍 Starting direct Farnan SQL dump analysis...")
        
        try:
            # Step 1: Read and parse SQL dump
            print("📖 Reading SQL dump...")
            self._read_dump()
            
            # Step 2: Analyze schema
            print("🏗️ Analyzing schema structure...")
            self._analyze_schema_direct()
            
            # Step 3: Analyze data patterns
            print("📊 Analyzing data patterns...")
            self._analyze_data_patterns_direct()
            
            # Step 4: Discover relationships
            print("🔗 Discovering table relationships...")
            self._discover_relationships()
            
            # Step 5: Generate insights
            print("💡 Generating insights...")
            self._generate_insights()
            
            return self.insights
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise
    
    def _read_dump(self):
        """Read the SQL dump file"""
        with open(self.dump_file, 'r', encoding='utf-8') as f:
            self.sql_content = f.read()
        
        print(f"📄 Dump file size: {len(self.sql_content):,} characters")
        
        # Extract table creation statements
        self.table_creates = re.findall(
            r'CREATE TABLE[^;]+;', 
            self.sql_content, 
            re.IGNORECASE | re.DOTALL
        )
        
        # Extract INSERT statements
        self.insert_statements = re.findall(
            r'INSERT INTO[^;]+;', 
            self.sql_content, 
            re.IGNORECASE | re.DOTALL
        )
        
        print(f"📋 Found {len(self.table_creates)} table definitions")
        print(f"📝 Found {len(self.insert_statements)} insert statements")
    
    def _analyze_schema_direct(self):
        """Analyze schema directly from CREATE TABLE statements"""
        self.schema_info = {
            'tables': {},
            'total_tables': len(self.table_creates),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        for create_sql in self.table_creates:
            table_info = self._parse_create_table(create_sql)
            if table_info:
                table_name = table_info['name']
                self.schema_info['tables'][table_name] = table_info
                print(f"📋 {table_name}: {len(table_info['columns'])} columns")
    
    def _parse_create_table(self, create_sql: str) -> Dict[str, Any]:
        """Parse CREATE TABLE statement"""
        try:
            # Extract table name
            table_match = re.search(r'CREATE TABLE[^`]*`([^`]+)`', create_sql, re.IGNORECASE)
            if not table_match:
                return None
            
            table_name = table_match.group(1)
            
            # Extract column definitions
            columns = {}
            
            # Find column definitions between parentheses
            column_section = re.search(r'\(([^)]+)\)', create_sql, re.DOTALL)
            if not column_section:
                return None
            
            column_text = column_section.group(1)
            
            # Parse each column
            column_lines = [line.strip() for line in column_text.split('\n') if line.strip()]
            
            for line in column_lines:
                # Skip constraints, indexes, etc.
                if any(keyword in line.upper() for keyword in ['PRIMARY KEY', 'FOREIGN KEY', 'INDEX', 'KEY', 'CONSTRAINT']):
                    continue
                
                # Parse column definition
                col_match = re.match(r'`([^`]+)`\s+([^,\s]+)([^,]*)', line)
                if col_match:
                    col_name = col_match.group(1)
                    col_type = col_match.group(2)
                    col_attrs = col_match.group(3)
                    
                    # Parse attributes
                    not_null = 'NOT NULL' in col_attrs.upper()
                    auto_increment = 'AUTO_INCREMENT' in col_attrs.upper()
                    primary_key = 'PRIMARY KEY' in col_attrs.upper()
                    
                    # Extract default value
                    default_match = re.search(r"DEFAULT\s+'([^']*)'|DEFAULT\s+(\w+)", col_attrs, re.IGNORECASE)
                    default_val = None
                    if default_match:
                        default_val = default_match.group(1) or default_match.group(2)
                    
                    columns[col_name] = {
                        'type': col_type,
                        'not_null': not_null,
                        'auto_increment': auto_increment,
                        'primary_key': primary_key,
                        'default': default_val
                    }
            
            return {
                'name': table_name,
                'columns': columns,
                'column_count': len(columns),
                'create_sql': create_sql
            }
            
        except Exception as e:
            print(f"⚠️ Error parsing table: {e}")
            return None
    
    def _analyze_data_patterns_direct(self):
        """Analyze data patterns from INSERT statements"""
        self.data_patterns = {}
        
        # Group INSERT statements by table
        table_inserts = defaultdict(list)
        
        for insert_sql in self.insert_statements:
            table_match = re.search(r'INSERT INTO[^`]*`([^`]+)`', insert_sql, re.IGNORECASE)
            if table_match:
                table_name = table_match.group(1)
                table_inserts[table_name].append(insert_sql)
        
        # Analyze each table's data
        for table_name, inserts in table_inserts.items():
            if table_name not in self.schema_info['tables']:
                continue
                
            patterns = {
                'insert_count': len(inserts),
                'sample_data': [],
                'value_patterns': defaultdict(set)
            }
            
            # Extract sample data from first few INSERT statements
            for i, insert_sql in enumerate(inserts[:5]):  # First 5 inserts
                values = self._extract_values_from_insert(insert_sql)
                if values:
                    patterns['sample_data'].append(values)
            
            # Analyze value patterns
            for insert_sql in inserts[:20]:  # Analyze first 20 inserts
                values = self._extract_values_from_insert(insert_sql)
                if values:
                    for j, value in enumerate(values):
                        if j < len(self.schema_info['tables'][table_name]['columns']):
                            col_name = list(self.schema_info['tables'][table_name]['columns'].keys())[j]
                            patterns['value_patterns'][col_name].add(str(value))
            
            # Convert sets to lists and get counts
            for col_name, values in patterns['value_patterns'].items():
                patterns['value_patterns'][col_name] = {
                    'unique_values': list(values),
                    'unique_count': len(values),
                    'sample_values': list(values)[:10]  # First 10 unique values
                }
            
            self.data_patterns[table_name] = patterns
            
            print(f"📊 {table_name}: {patterns['insert_count']} insert statements, {len(patterns['sample_data'])} samples")
    
    def _extract_values_from_insert(self, insert_sql: str) -> List[str]:
        """Extract values from INSERT statement"""
        try:
            # Find VALUES clause
            values_match = re.search(r'VALUES\s*\(([^)]+)\)', insert_sql, re.IGNORECASE | re.DOTALL)
            if not values_match:
                return []
            
            values_text = values_match.group(1)
            
            # Simple value extraction (handles basic cases)
            values = []
            current_value = ""
            in_quotes = False
            quote_char = None
            
            i = 0
            while i < len(values_text):
                char = values_text[i]
                
                if not in_quotes:
                    if char in ["'", '"']:
                        in_quotes = True
                        quote_char = char
                        current_value += char
                    elif char == ',':
                        values.append(current_value.strip())
                        current_value = ""
                    else:
                        current_value += char
                else:
                    current_value += char
                    if char == quote_char:
                        # Check for escaped quote
                        if i + 1 < len(values_text) and values_text[i + 1] == quote_char:
                            i += 1
                            current_value += char
                        else:
                            in_quotes = False
                            quote_char = None
                
                i += 1
            
            # Add last value
            if current_value.strip():
                values.append(current_value.strip())
            
            return values
            
        except Exception as e:
            print(f"⚠️ Error extracting values: {e}")
            return []
    
    def _discover_relationships(self):
        """Discover relationships between tables"""
        self.relationships = {}
        
        # Collect all column names across tables
        all_columns = defaultdict(list)
        for table_name, table_info in self.schema_info['tables'].items():
            for col_name in table_info['columns']:
                all_columns[col_name].append(table_name)
        
        # Find potential foreign keys
        for col_name, tables in all_columns.items():
            if len(tables) > 1:
                # Column appears in multiple tables - potential foreign key
                for table in tables:
                    if table not in self.relationships:
                        self.relationships[table] = {}
                    self.relationships[table][col_name] = {
                        'type': 'potential_foreign_key',
                        'related_tables': [t for t in tables if t != table]
                    }
        
        # Look for ID columns
        for table_name, table_info in self.schema_info['tables'].items():
            for col_name, col_info in table_info['columns'].items():
                if col_info.get('primary_key') or col_info.get('auto_increment'):
                    if table_name not in self.relationships:
                        self.relationships[table_name] = {}
                    self.relationships[table_name][col_name] = {
                        'type': 'primary_key',
                        'auto_increment': col_info.get('auto_increment', False)
                    }
    
    def _generate_insights(self):
        """Generate insights for NLP-to-SQL improvement"""
        self.insights = {
            'schema_analysis': self.schema_info,
            'data_patterns': self.data_patterns,
            'relationships': self.relationships,
            'recommendations': self._generate_recommendations(),
            'vector_store_improvements': self._generate_vector_store_improvements(),
            'prompt_improvements': self._generate_prompt_improvements()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Table analysis
        total_tables = len(self.schema_info['tables'])
        recommendations.append(f"Total tables in database: {total_tables}")
        
        # Column analysis
        all_columns = set()
        for table_info in self.schema_info['tables'].values():
            all_columns.update(table_info['columns'].keys())
        
        recommendations.append(f"Total unique columns across all tables: {len(all_columns)}")
        
        # Data pattern recommendations
        for table_name, patterns in self.data_patterns.items():
            if patterns['insert_count'] > 0:
                recommendations.append(f"{table_name} has {patterns['insert_count']} insert statements - contains data")
                
                for col_name, value_info in patterns['value_patterns'].items():
                    if value_info['unique_count'] < 20:
                        recommendations.append(f"{table_name}.{col_name} has {value_info['unique_count']} unique values - good for categorical analysis")
        
        return recommendations
    
    def _generate_vector_store_improvements(self) -> Dict[str, Any]:
        """Generate improvements for vector store"""
        improvements = {
            'enhanced_schema_context': {},
            'sample_data': {},
            'column_mappings': {},
            'business_context': {}
        }
        
        # Enhanced schema context
        for table_name, table_info in self.schema_info['tables'].items():
            improvements['enhanced_schema_context'][table_name] = {
                'columns': list(table_info['columns'].keys()),
                'column_count': table_info['column_count'],
                'description': self._generate_table_description(table_name, table_info),
                'column_details': table_info['columns']
            }
        
        # Sample data for better context
        for table_name, patterns in self.data_patterns.items():
            if patterns['sample_data']:
                improvements['sample_data'][table_name] = patterns['sample_data'][:3]  # Top 3 samples
        
        # Column mappings for common mistakes
        improvements['column_mappings'] = self._generate_column_mappings()
        
        return improvements
    
    def _generate_table_description(self, table_name: str, table_info: Dict) -> str:
        """Generate business description for table"""
        descriptions = {
            'workers': 'Employee and personnel information',
            'production_info': 'Main production records and batch data',
            'person_hyg': 'Personnel hygiene compliance tracking',
            'packaging_info': 'Packaging specifications and batch details',
            'pack_waste': 'Packaging waste tracking and analysis',
            'production_test': 'Quality testing and control results',
            'prices': 'Ingredient and material pricing data',
            'packs': 'Packaging configurations and specifications',
            'repo_nc': 'Non-conformance reports and quality issues',
            'transtatus': 'Transaction status and tracking',
            'users': 'System users and access control'
        }
        
        return descriptions.get(table_name, f"Data table with {table_info['column_count']} columns")
    
    def _generate_column_mappings(self) -> Dict[str, Dict[str, str]]:
        """Generate column mappings for common mistakes"""
        mappings = {}
        
        for table_name, table_info in self.schema_info['tables'].items():
            table_mappings = {}
            columns = list(table_info['columns'].keys())
            
            # Common naming variations
            for col in columns:
                col_lower = col.lower()
                if 'id' in col_lower and col != 'id':
                    table_mappings[col.replace('_id', '')] = col
                if 'type' in col_lower:
                    table_mappings[col.replace('_type', '') + 'Type'] = col
                if 'date' in col_lower:
                    table_mappings[col.replace('_date', '') + 'Date'] = col
                # Add common mistakes we've seen
                if col == 'type':
                    table_mappings['wasteType'] = col  # Common mistake
                if col == 'value':
                    table_mappings['amount'] = col  # Common mistake
                    table_mappings['waste_amount'] = col
        
            if table_mappings:
                mappings[table_name] = table_mappings
        
        return mappings
    
    def _generate_prompt_improvements(self) -> Dict[str, Any]:
        """Generate prompt improvements"""
        return {
            'table_selection_rules': self._generate_table_selection_rules(),
            'domain_examples': self._generate_domain_examples(),
            'validation_rules': self._generate_validation_rules()
        }
    
    def _generate_table_selection_rules(self) -> List[str]:
        """Generate table selection rules"""
        rules = []
        
        for table_name, table_info in self.schema_info['tables'].items():
            columns = list(table_info['columns'].keys())
            
            if 'worker' in table_name.lower():
                rules.append(f"WORKER/EMPLOYEE queries → use '{table_name}' table")
            elif 'production' in table_name.lower():
                rules.append(f"PRODUCTION data → use '{table_name}' table")
            elif 'packaging' in table_name.lower():
                rules.append(f"PACKAGING data → use '{table_name}' table")
            elif 'waste' in table_name.lower():
                rules.append(f"WASTE data → use '{table_name}' table")
            elif 'hyg' in table_name.lower():
                rules.append(f"HYGIENE data → use '{table_name}' table")
            elif 'price' in table_name.lower():
                rules.append(f"PRICE data → use '{table_name}' table")
            elif 'test' in table_name.lower():
                rules.append(f"TEST data → use '{table_name}' table")
            elif 'pack' in table_name.lower():
                rules.append(f"PACK data → use '{table_name}' table")
            elif 'tran' in table_name.lower():
                rules.append(f"TRANSACTION data → use '{table_name}' table")
            elif 'user' in table_name.lower():
                rules.append(f"USER data → use '{table_name}' table")
            elif 'repo' in table_name.lower():
                rules.append(f"REPORT data → use '{table_name}' table")
        
        return rules
    
    def _generate_domain_examples(self) -> List[str]:
        """Generate domain-specific examples"""
        examples = []
        
        # Generate examples based on actual table structure
        for table_name, table_info in self.schema_info['tables'].items():
            columns = list(table_info['columns'].keys())
            
            if 'date' in columns:
                examples.append(f"'{table_name} data for this month' → SELECT * FROM `{table_name}` WHERE MONTH(`date`) = MONTH(CURDATE()) LIMIT 50")
            if 'type' in columns:
                examples.append(f"'{table_name} types' → SELECT `type`, COUNT(*) FROM `{table_name}` GROUP BY `type` LIMIT 50")
            if 'id' in columns:
                examples.append(f"'{table_name} count' → SELECT COUNT(*) FROM `{table_name}` LIMIT 50")
            
            # Specific examples for known tables
            if table_name == 'workers':
                examples.append("'How many workers?' → SELECT COUNT(*) FROM `workers` LIMIT 50")
            elif table_name == 'pack_waste':
                examples.append("'Waste distribution by type' → SELECT `type`, COUNT(*) FROM `pack_waste` GROUP BY `type` LIMIT 50")
            elif table_name == 'production_info':
                examples.append("'Production volumes this month' → SELECT SUM(totalUsage) FROM `production_info` WHERE MONTH(date) = MONTH(CURDATE()) LIMIT 50")
        
        return examples[:15]  # Limit to top 15 examples
    
    def _generate_validation_rules(self) -> List[str]:
        """Generate validation rules"""
        rules = [
            "Use backticks around all table and column names",
            "Always add LIMIT 50 to prevent large result sets",
            "Use proper MySQL syntax for date functions",
            "Validate table and column names against actual schema"
        ]
        
        # Add table-specific rules
        for table_name, table_info in self.schema_info['tables'].items():
            columns = list(table_info['columns'].keys())
            if 'type' in columns and table_name == 'pack_waste':
                rules.append(f"Use 'type' column in {table_name}, not 'wasteType'")
            if 'value' in columns and table_name == 'pack_waste':
                rules.append(f"Use 'value' column in {table_name}, not 'amount'")
        
        return rules

def main():
    """Main function"""
    dump_file = "farnan.sql"
    
    if not os.path.exists(dump_file):
        print(f"❌ Dump file not found: {dump_file}")
        return
    
    analyzer = DirectFarnanAnalyzer(dump_file)
    
    try:
        insights = analyzer.analyze()
        
        # Save insights to file
        output_file = "farnan_analysis_insights_direct.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n🎉 Analysis complete!")
        print(f"📊 Results saved to: {output_file}")
        print(f"\n📋 Summary:")
        print(f"   Tables: {insights['schema_analysis']['total_tables']}")
        print(f"   Recommendations: {len(insights['recommendations'])}")
        print(f"   Vector Store Improvements: {len(insights['vector_store_improvements']['enhanced_schema_context'])} tables")
        
        # Print key insights
        print(f"\n🔍 Key Insights:")
        for table_name, table_info in insights['schema_analysis']['tables'].items():
            print(f"   📋 {table_name}: {table_info['column_count']} columns")
            if table_name in insights['data_patterns']:
                data_info = insights['data_patterns'][table_name]
                print(f"      📊 {data_info['insert_count']} insert statements")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
