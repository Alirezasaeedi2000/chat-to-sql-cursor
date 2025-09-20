#!/usr/bin/env python3
"""
Visualization Handler for Hybrid NLP-to-SQL System
Handles chart/graph generation requests with SQL data extraction and visualization creation
"""

import logging
import time
import os
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from .base_handler import BaseHandler
from query_classifier import QueryClassification

LOGGER = logging.getLogger(__name__)


class VisualizationHandler(BaseHandler):
    """Handler for visualization/chart generation queries"""
    
    def __init__(self, engine, vector_manager, llm, config: Optional[Dict[str, Any]] = None):
        super().__init__(engine, vector_manager, llm, config)
        self.max_execution_time = self.config.get('max_execution_time', 5.0)
        self.supported_chart_types = self.config.get('supported_chart_types', ['bar', 'pie', 'line', 'scatter'])
        self.default_chart_size = self.config.get('default_chart_size', (10, 6))
        self.save_charts = self.config.get('save_charts', True)
        self.chart_output_dir = self.config.get('chart_output_dir', 'outputs/plots')
        
        # Ensure output directory exists
        os.makedirs(self.chart_output_dir, exist_ok=True)
    
    def can_handle(self, query: str, classification: QueryClassification) -> bool:
        """Check if this handler can process the query"""
        return (classification.type == 'visualization' and 
                classification.confidence >= 0.5)
    
    def process(self, query: str, classification: QueryClassification) -> Dict[str, Any]:
        """Process visualization query"""
        start_time = time.time()
        
        try:
            LOGGER.info(f"Processing visualization query: {query[:50]}...")
            
            # Step 1: Extract visualization specifications
            viz_spec = self._extract_visualization_spec(query, classification)
            
            # Step 2: Generate SQL for chart data
            sql = self._generate_chart_data_sql(query, viz_spec)
            
            # Step 3: Execute SQL and get data
            data = self.execute_sql(sql)
            
            if data.empty:
                LOGGER.warning("No data returned for visualization")
                return self._handle_empty_data(query, viz_spec)
            
            # Step 4: Create visualization
            chart_path = self._create_chart(data, viz_spec, query)
            
            # Step 5: Format result
            result = self._format_visualization_result(data, chart_path, viz_spec, sql)
            
            execution_time = time.time() - start_time
            self.update_stats(execution_time, True)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.update_stats(execution_time, False)
            return self.handle_error(query, classification, e)
    
    def _extract_visualization_spec(self, query: str, classification: QueryClassification) -> Dict[str, Any]:
        """Extract visualization specifications from query"""
        try:
            # Get schema context
            schema_context = self.get_schema_context()
            
            # Build prompt for visualization specification extraction
            prompt = f"""Extract visualization specifications from this query.

QUERY: "{query}"

SCHEMA CONTEXT:
{schema_context}

TASK: Analyze the query and determine what type of chart/visualization is requested and what data should be visualized.

AVAILABLE CHART TYPES: bar, pie, line, scatter, histogram

EXAMPLES:
- "Show production volumes as a bar chart" → chart_type: bar, data: production volumes
- "Create a pie chart of waste types" → chart_type: pie, data: waste types distribution
- "Display hygiene compliance rates with a line chart" → chart_type: line, data: compliance rates over time

Return JSON with:
{{
    "chart_type": "bar|pie|line|scatter|histogram",
    "data_source": "description of what data to visualize",
    "x_axis": "column name or description for x-axis",
    "y_axis": "column name or description for y-axis", 
    "title": "chart title",
    "x_label": "x-axis label",
    "y_label": "y-axis label",
    "group_by": "column to group data by (if applicable)",
    "aggregation": "sum|count|avg|max|min (if applicable)"
}}"""

            response = self.llm.invoke(prompt)
            viz_spec = self._parse_visualization_spec(response)
            
            # Validate and enhance specification
            viz_spec = self._validate_visualization_spec(viz_spec, query)
            
            LOGGER.info(f"Extracted visualization spec: {viz_spec['chart_type']} chart")
            return viz_spec
            
        except Exception as e:
            LOGGER.error(f"Failed to extract visualization spec: {e}")
            # Fallback specification
            return self._get_fallback_visualization_spec(query)
    
    def _parse_visualization_spec(self, response) -> Dict[str, Any]:
        """Parse visualization specification from LLM response"""
        try:
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            if '```json' in content:
                start = content.find('```json') + 7
                end = content.find('```', start)
                json_str = content[start:end].strip()
            elif '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
            else:
                raise ValueError("No JSON found in response")
            
            import json
            return json.loads(json_str)
            
        except Exception as e:
            LOGGER.error(f"Failed to parse visualization spec: {e}")
            raise
    
    def _validate_visualization_spec(self, viz_spec: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Validate and enhance visualization specification"""
        # Set defaults
        defaults = {
            'chart_type': 'bar',
            'title': f'Chart for: {query[:50]}',
            'x_label': 'X-Axis',
            'y_label': 'Y-Axis',
            'aggregation': 'count'
        }
        
        # Apply defaults for missing values
        for key, default_value in defaults.items():
            if key not in viz_spec or not viz_spec[key]:
                viz_spec[key] = default_value
        
        # Validate chart type
        if viz_spec['chart_type'] not in self.supported_chart_types:
            LOGGER.warning(f"Unsupported chart type: {viz_spec['chart_type']}, defaulting to bar")
            viz_spec['chart_type'] = 'bar'
        
        return viz_spec
    
    def _get_fallback_visualization_spec(self, query: str) -> Dict[str, Any]:
        """Get fallback visualization specification"""
        query_lower = query.lower()
        
        # Simple heuristics for fallback
        if 'pie' in query_lower:
            chart_type = 'pie'
        elif 'line' in query_lower:
            chart_type = 'line'
        elif 'scatter' in query_lower:
            chart_type = 'scatter'
        else:
            chart_type = 'bar'
        
        return {
            'chart_type': chart_type,
            'data_source': 'data from query',
            'title': f'Chart for: {query[:50]}',
            'x_label': 'X-Axis',
            'y_label': 'Y-Axis',
            'aggregation': 'count'
        }
    
    def _generate_chart_data_sql(self, query: str, viz_spec: Dict[str, Any]) -> str:
        """Generate SQL to get data for visualization"""
        try:
            schema_context = self.get_schema_context()
            
            prompt = f"""Generate SQL to get data for this visualization.

ORIGINAL QUERY: "{query}"

VISUALIZATION SPEC:
- Chart Type: {viz_spec['chart_type']}
- Data Source: {viz_spec['data_source']}
- X-Axis: {viz_spec.get('x_axis', 'N/A')}
- Y-Axis: {viz_spec.get('y_axis', 'N/A')}
- Group By: {viz_spec.get('group_by', 'N/A')}
- Aggregation: {viz_spec.get('aggregation', 'count')}

SCHEMA CONTEXT:
{schema_context}

TASK: Generate SQL that returns data suitable for a {viz_spec['chart_type']} chart.

RULES:
- Always include LIMIT 50
- Use backticks around table/column names
- For pie charts: return category and count
- For bar charts: return category and value
- For line charts: return time/date and value
- For scatter plots: return x and y values

EXAMPLES:
- Pie chart of waste types: SELECT `type`, COUNT(*) as count FROM `pack_waste` GROUP BY `type` LIMIT 50
- Bar chart of production by type: SELECT `bakeType`, SUM(`totalUsage`) as total FROM `production_info` GROUP BY `bakeType` LIMIT 50

Generate SQL for the visualization above. Return ONLY the SQL statement wrapped in ```sql``` fences."""

            response = self.llm.invoke(prompt)
            sql = self._extract_sql_from_response(response)
            
            if not sql:
                raise ValueError("No SQL generated for visualization")
            
            return sql
            
        except Exception as e:
            LOGGER.error(f"Failed to generate chart data SQL: {e}")
            # Fallback SQL
            return self._get_fallback_sql(viz_spec)
    
    def _get_fallback_sql(self, viz_spec: Dict[str, Any]) -> str:
        """Get fallback SQL for visualization"""
        chart_type = viz_spec['chart_type']
        
        if chart_type == 'pie':
            return "SELECT `type`, COUNT(*) as count FROM `pack_waste` GROUP BY `type` LIMIT 50"
        elif chart_type == 'bar':
            return "SELECT `bakeType`, SUM(`totalUsage`) as total FROM `production_info` GROUP BY `bakeType` LIMIT 50"
        elif chart_type == 'line':
            return "SELECT `date`, SUM(`totalUsage`) as total FROM `production_info` GROUP BY `date` ORDER BY `date` LIMIT 50"
        else:
            return "SELECT COUNT(*) as count FROM `workers` LIMIT 50"
    
    def _create_chart(self, data: pd.DataFrame, viz_spec: Dict[str, Any], query: str) -> str:
        """Create visualization chart"""
        try:
            # Set up the plot
            plt.figure(figsize=self.default_chart_size)
            
            chart_type = viz_spec['chart_type']
            
            # Create chart based on type
            if chart_type == 'pie':
                self._create_pie_chart(data, viz_spec)
            elif chart_type == 'bar':
                self._create_bar_chart(data, viz_spec)
            elif chart_type == 'line':
                self._create_line_chart(data, viz_spec)
            elif chart_type == 'scatter':
                self._create_scatter_chart(data, viz_spec)
            else:
                # Default to bar chart
                self._create_bar_chart(data, viz_spec)
            
            # Set title and labels
            plt.title(viz_spec['title'], fontsize=14, fontweight='bold')
            plt.xlabel(viz_spec['x_label'])
            plt.ylabel(viz_spec['y_label'])
            
            # Add grid for better readability
            plt.grid(True, alpha=0.3)
            
            # Generate filename and save
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"viz_{timestamp}_{hash(query) % 1000000}.png"
            chart_path = os.path.join(self.chart_output_dir, filename)
            
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            LOGGER.info(f"Chart saved to: {chart_path}")
            return chart_path
            
        except Exception as e:
            LOGGER.error(f"Failed to create chart: {e}")
            raise
    
    def _create_pie_chart(self, data: pd.DataFrame, viz_spec: Dict[str, Any]):
        """Create pie chart"""
        if len(data.columns) >= 2:
            labels = data.iloc[:, 0].astype(str)
            values = data.iloc[:, 1].astype(float)
            
            plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        else:
            raise ValueError("Pie chart requires at least 2 columns")
    
    def _create_bar_chart(self, data: pd.DataFrame, viz_spec: Dict[str, Any]):
        """Create bar chart"""
        if len(data.columns) >= 2:
            x_data = data.iloc[:, 0].astype(str)
            y_data = data.iloc[:, 1].astype(float)
            
            plt.bar(x_data, y_data)
            plt.xticks(rotation=45)
        else:
            raise ValueError("Bar chart requires at least 2 columns")
    
    def _create_line_chart(self, data: pd.DataFrame, viz_spec: Dict[str, Any]):
        """Create line chart"""
        if len(data.columns) >= 2:
            x_data = data.iloc[:, 0]
            y_data = data.iloc[:, 1].astype(float)
            
            plt.plot(x_data, y_data, marker='o', linewidth=2)
            plt.xticks(rotation=45)
        else:
            raise ValueError("Line chart requires at least 2 columns")
    
    def _create_scatter_chart(self, data: pd.DataFrame, viz_spec: Dict[str, Any]):
        """Create scatter plot"""
        if len(data.columns) >= 2:
            x_data = data.iloc[:, 0].astype(float)
            y_data = data.iloc[:, 1].astype(float)
            
            plt.scatter(x_data, y_data, alpha=0.6)
        else:
            raise ValueError("Scatter plot requires at least 2 columns")
    
    def _extract_sql_from_response(self, response) -> Optional[str]:
        """Extract SQL from LLM response"""
        try:
            content = response.content if hasattr(response, 'content') else str(response)
            
            if '```sql' in content:
                start = content.find('```sql') + 6
                end = content.find('```', start)
                sql = content[start:end].strip()
            elif '```' in content:
                start = content.find('```') + 3
                end = content.find('```', start)
                sql = content[start:end].strip()
            else:
                import re
                select_match = re.search(r'SELECT.*?(?=\n\n|\n$|$)', content, re.IGNORECASE | re.DOTALL)
                if select_match:
                    sql = select_match.group(0).strip()
                else:
                    return None
            
            return sql if sql else None
            
        except Exception as e:
            LOGGER.error(f"Failed to extract SQL from response: {e}")
            return None
    
    def _handle_empty_data(self, query: str, viz_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Handle case when no data is returned"""
        return {
            'mode': 'VISUALIZATION',
            'visualization_path': None,
            'table_markdown': '(no data available)',
            'sql': None,
            'metadata': {
                'handler_used': 'visualization',
                'classification': {'type': 'visualization'},
                'confidence': 0.0,
                'errors': ['No data available for visualization'],
                'chart_type': viz_spec.get('chart_type', 'unknown')
            }
        }
    
    def _format_visualization_result(self, data: pd.DataFrame, chart_path: str, 
                                   viz_spec: Dict[str, Any], sql: str) -> Dict[str, Any]:
        """Format visualization result"""
        return {
            'mode': 'VISUALIZATION',
            'visualization_path': chart_path,
            'table_markdown': data.to_markdown(index=False),
            'sql': sql,
            'metadata': {
                'handler_used': 'visualization',
                'classification': {'type': 'visualization'},
                'confidence': 0.8,
                'chart_type': viz_spec['chart_type'],
                'chart_title': viz_spec['title'],
                'row_count': len(data),
                'errors': None
            }
        }


# Example usage and testing
if __name__ == "__main__":
    print("Visualization Handler Test")
    print("=" * 40)
    print("Handler implementation complete")
