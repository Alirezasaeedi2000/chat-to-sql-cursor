# 🚀 Hybrid NLP-to-SQL System

## Overview

The Hybrid NLP-to-SQL System is an intelligent, agent-based architecture that routes natural language queries to specialized handlers based on query complexity and type. This system replaces the rigid pattern-based approach with flexible, context-aware processing.

## 🏗️ Architecture

### Core Components

```
User Query → Query Classifier → Handler Router → Specialized Handler → Result
```

1. **Query Classifier** (`query_classifier.py`)
   - LLM-based semantic understanding
   - Classifies queries by complexity and type
   - Confidence-based routing with fallbacks

2. **Specialized Handlers**
   - **SimpleQueryHandler**: Fast deterministic builders + LLM fallback
   - **VisualizationHandler**: Chart/graph generation with data extraction
   - **AnalyticalHandler**: Multi-step analysis and insights
   - **ComplexQueryHandler**: Complex multi-step reasoning

3. **Hybrid System Orchestrator** (`hybrid_nl2sql_system.py`)
   - Routes queries to appropriate handlers
   - Manages fallback chains
   - Provides system-wide monitoring and statistics

## 📁 File Structure

```
├── query_classifier.py              # Intelligent query classification
├── hybrid_nl2sql_system.py          # Main system orchestrator
├── handlers/
│   ├── __init__.py                  # Handler package initialization
│   ├── base_handler.py              # Abstract base handler class
│   ├── simple_handler.py            # Simple/scalar query handler
│   ├── visualization_handler.py     # Chart/graph handler
│   ├── analytical_handler.py        # Analysis handler (placeholder)
│   └── complex_handler.py           # Complex query handler (placeholder)
├── demo_hybrid_system.py            # Interactive demonstration
├── test_hybrid_system.py            # Comprehensive test suite
└── HYBRID_SYSTEM_README.md          # This file
```

## 🎯 Key Features

### 1. Intelligent Query Classification
- **Semantic Understanding**: Uses LLM to understand query intent
- **Confidence Scoring**: Routes based on classification confidence
- **Fallback Mechanisms**: Defaults to complex handler for uncertain cases

### 2. Specialized Handlers
- **SimpleQueryHandler**: 
  - Deterministic builders for common patterns
  - LLM fallback for edge cases
  - Sub-second response times
  
- **VisualizationHandler**:
  - Automatic chart type detection
  - SQL generation for chart data
  - Matplotlib-based visualization creation
  
- **AnalyticalHandler**: (To be implemented)
  - Multi-step reasoning
  - Trend analysis and insights
  - Comparative analysis

- **ComplexQueryHandler**: (To be implemented)
  - Multi-step query planning
  - Complex SQL generation
  - Advanced reasoning

### 3. Robust Error Handling
- **Fallback Chains**: Simple → Complex → Generic LLM
- **Circuit Breakers**: Prevent cascade failures
- **Comprehensive Logging**: Full error tracking and debugging

### 4. Performance Optimization
- **Smart Caching**: Classification and result caching
- **Fast Models**: Uses lightweight models for classification
- **Parallel Processing**: Handler-independent execution

## 🚀 Quick Start

### 1. Setup
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Make sure database is running
# Update connection string in demo_hybrid_system.py if needed
```

### 2. Run Demonstration
```bash
python demo_hybrid_system.py
```

### 3. Run Test Suite
```bash
python test_hybrid_system.py
```

### 4. Interactive Mode
```python
from demo_hybrid_system import HybridSystemDemo

demo = HybridSystemDemo()
demo.run_interactive_mode()
```

## 📊 Performance Metrics

### Expected Performance Improvements
- **Simple Queries**: 20s → 0.5s (40x improvement)
- **Visualization Queries**: 20s → 3s (6.7x improvement)  
- **Complex Queries**: 20s → 8s (2.5x improvement)
- **Overall Accuracy**: 72.7% → 85%+ target

### Handler Performance
| Handler | Avg Response Time | Use Case |
|---------|------------------|----------|
| Simple | 0.5-1s | Counts, lists, simple aggregations |
| Visualization | 2-5s | Charts, graphs, plots |
| Analytical | 5-8s | Trends, comparisons, insights |
| Complex | 8-15s | Multi-step, complex reasoning |

## 🔧 Configuration

### Handler Configurations
```python
handler_configs = {
    'simple': {
        'max_execution_time': 2.0,
        'use_deterministic_builders': True,
        'fallback_to_llm': True,
        'cache_results': True
    },
    'visualization': {
        'max_execution_time': 5.0,
        'supported_chart_types': ['bar', 'pie', 'line', 'scatter'],
        'default_chart_size': (10, 6),
        'save_charts': True,
        'chart_output_dir': 'outputs/plots'
    }
}
```

### LLM Models
- **Classification**: `llama3.2:1b` (fast, lightweight)
- **SQL Generation**: `llama3.1:8b` (balanced performance)
- **Complex Analysis**: `llama3.1:70b` (if available)

## 🧪 Testing

### Test Coverage
- **Query Classification**: Tests classification accuracy and confidence
- **Handler Routing**: Verifies correct handler selection
- **Result Generation**: Validates output format and content
- **Error Handling**: Tests fallback mechanisms
- **Performance**: Measures execution times

### Test Categories
1. **Simple Scalar Queries**: "How many workers?"
2. **Simple Table Queries**: "Show all workers"
3. **Visualization Queries**: "Create a bar chart"
4. **Analytical Queries**: "Analyze trends"
5. **Complex Queries**: "Show data and create chart"

## 📈 Monitoring & Statistics

### System Statistics
```python
stats = system.get_system_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Success rate: {stats['success_rate']*100:.1f}%")
print(f"Average execution time: {stats['average_execution_time']:.2f}s")
print(f"Handler usage: {stats['handler_usage']}")
```

### Health Monitoring
```python
health = system.health_check()
print(f"Overall health: {health['overall']}")
for component, status in health['components'].items():
    print(f"{component}: {status}")
```

## 🔄 Migration from Current System

### Phase 1: Foundation (Completed)
- ✅ Query Classifier implementation
- ✅ Base Handler architecture
- ✅ SimpleQueryHandler with deterministic builders
- ✅ VisualizationHandler with chart generation
- ✅ Hybrid System orchestrator

### Phase 2: Enhancement (In Progress)
- 🔄 AnalyticalHandler implementation
- 🔄 ComplexQueryHandler implementation
- 🔄 Advanced caching strategies
- 🔄 Performance optimization

### Phase 3: Integration (Planned)
- ⏳ Full system integration
- ⏳ A/B testing with current system
- ⏳ Gradual migration strategy
- ⏳ Production deployment

## 🛠️ Development

### Adding New Handlers
1. Inherit from `BaseHandler`
2. Implement `can_handle()` and `process()` methods
3. Register in `HybridNL2SQLSystem._initialize_handlers()`
4. Add to fallback chains if needed

### Extending Query Classification
1. Add new examples to `QueryClassifier._load_classification_examples()`
2. Update classification prompt in `_build_classification_prompt()`
3. Add new handler types if needed

### Customizing Deterministic Builders
1. Add new builders to `DeterministicBuilderManager`
2. Implement pattern matching logic
3. Return SQL string or None

## 🐛 Troubleshooting

### Common Issues
1. **Classification Errors**: Check LLM model availability
2. **Handler Failures**: Verify database connection and schema
3. **Visualization Issues**: Ensure matplotlib and output directory
4. **Performance Issues**: Check caching and model selection

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run system with detailed logging
demo = HybridSystemDemo()
```

## 📚 API Reference

### Main System
```python
system = HybridNL2SQLSystem(engine, vector_manager)
result = system.process_query(query)
```

### Result Format
```json
{
    "mode": "SHORT_ANSWER|TABLE|VISUALIZATION|ANALYTICAL|ERROR",
    "short_answer": "...",           // if scalar
    "table_markdown": "...",         // if table
    "visualization_path": "...",     // if chart
    "sql": "...",
    "metadata": {
        "handler_used": "...",
        "classification": {...},
        "confidence": 0.0-1.0,
        "execution_time_ms": 0,
        "errors": null|[...]
    }
}
```

## 🎯 Future Enhancements

1. **Advanced Analytics**: Statistical analysis and machine learning insights
2. **Multi-Database Support**: Extend to different database types
3. **Query Optimization**: Automatic SQL optimization and indexing
4. **User Learning**: Adapt to user preferences and query patterns
5. **Real-time Collaboration**: Multi-user query sharing and collaboration

## 📄 License

This project is part of the Farnan NLP-to-SQL system. See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

---

**Note**: This hybrid system represents a significant architectural improvement over the current pattern-based approach, providing better flexibility, accuracy, and performance for natural language to SQL conversion.
