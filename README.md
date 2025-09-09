## Natural Language → SQL Assistant with Advanced Features

This project provides a production-ready Python 3.11+ assistant that converts natural language into safe SQL queries, executes them with strict security guards, and returns intelligent multi-mode responses. The system includes advanced features like query caching, comprehensive history management, a modern web interface, enhanced visualizations, and extensive testing.

### 🚀 Key Features

**Core Functionality:**
- 🛡️ **Safe SQL Execution**: SELECT-only queries with automatic LIMIT injection, timeout protection, and SQL injection prevention
- 🤖 **Intelligent Mode Detection**: Automatically chooses the best response format (TABLE, SHORT_ANSWER, ANALYTICAL, VISUALIZATION, COMBO)
- 📊 **Advanced Visualizations**: Smart chart type detection with support for bar charts, line plots, scatter plots, pie charts, and histograms
- 🔍 **RAG-Enhanced**: Vector embeddings with ChromaDB and mxbai-embed-large for schema/sample/query context retrieval

**Performance & Reliability:**
- ⚡ **Query Result Caching**: File-based TTL cache for improved performance on repeated queries
- 🔄 **Enhanced Error Handling**: Improved connection handling with clear error messages and automatic retry logic
- 📈 **Multi-Database Support**: Extended beyond MySQL to support PostgreSQL and SQLite
- 🧪 **Comprehensive Testing**: Full test suite covering all major components

**User Experience:**
- 🌐 **Modern Web Interface**: Beautiful, responsive web UI with real-time query processing and visualization display
- 📚 **Query History Management**: Complete history with search, favorites, tags, statistics, and export capabilities
- 🖥️ **Multiple Interfaces**: CLI, web interface, and MCP stdio server support
- 📊 **Rich Analytics**: Query performance metrics, success rates, and usage patterns

**Integration & Extensibility:**
- 🔌 **MCP Protocol Support**: Full MCP stdio server with comprehensive database tools
- 🎯 **Advanced Prompting**: Optimized few-shot prompts for better SQL generation and intent detection
- 📁 **Export Capabilities**: CSV and JSON export with automatic file management
- 🔧 **Developer Tools**: Extensive logging, debugging support, and configuration options

### 📋 Requirements

**System Requirements:**
- Python 3.11+
- Database: MySQL, PostgreSQL, or SQLite
- Ollama with required models:
  ```bash
  ollama pull llama3.2          # Main LLM for query processing
  ollama pull mxbai-embed-large # Embeddings for RAG
  ```

**Installation:**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Set your database URL
export DATABASE_URL='mysql+pymysql://user:pass@host:3306/yourdb'
# or for PostgreSQL:
# export DATABASE_URL='postgresql+psycopg2://user:pass@host:5432/yourdb'
# or for SQLite:
# export DATABASE_URL='sqlite:///path/to/your/database.db'
```

**Optional: Create .env file for easier configuration:**
```bash
echo "DATABASE_URL=your_database_url_here" > .env
```

### 🚀 Quick Start

**1. Build Vector Index (Schema + Sample Data):**
```bash
python app.py --build-index
```

**2. Start the Web Interface (Recommended):**
```bash
python app.py --web
# Open http://localhost:8080 in your browser
```

**3. Use CLI for One-shot Queries:**
```bash
python app.py --query "Show top 10 customers by revenue" --mode VISUALIZATION --export csv
```

**4. Interactive CLI Mode:**
```bash
python app.py
# Then type your questions at the NL> prompt
```

**5. Run as MCP Server:**
```bash
python app.py --server
# Your MCP client can now connect via stdio
```

### 📊 Query History Commands

```bash
# View recent query history
python app.py --history

# Search history
python app.py --history-search "customers"

# Show statistics
python app.py --history-stats

# Export history
python app.py --history-export json

# Clear all history
python app.py --history-clear
```

### 🎯 Output Modes

The system automatically detects the best response format or you can force a specific mode:

- **TABLE**: Full data table in Markdown format with pagination
- **SHORT_ANSWER**: Concise scalar value (perfect for counts, sums, averages)
- **ANALYTICAL**: Structured business insights (Insights, Gaps, Risks, Recommendations)
- **VISUALIZATION**: Smart chart generation (bar, line, pie, scatter, histogram)
- **COMBO**: Combines multiple modes for comprehensive analysis

### 📁 Project Structure

```
├── app.py                 # Main CLI application and server runner
├── query_processor.py     # Core query processing with caching and history
├── query_history.py       # Comprehensive history management
├── web_server.py          # Modern web interface
├── vector.py              # Vector store and semantic search
├── mcp_handler.py         # LLM prompt engineering
├── mcp_sql_server.py      # MCP protocol server
├── test_nl2sql.py         # Comprehensive test suite
└── outputs/
    ├── cache/             # Query result cache
    ├── exports/           # CSV/JSON exports
    ├── logs/             # Application logs
    ├── plots/            # Generated visualizations
    ├── query_history.json # Query history database
    └── synonyms.json     # Domain-specific synonyms
```

### 💡 Example Queries

**Data Exploration:**
- "Show me the top 10 customers by total revenue"
- "List all products with price greater than $100"
- "What are the most recent orders from this month?"

**Analytics & Insights:**
- "Compare Q3 vs Q2 sales performance and identify key drivers"
- "Analyze customer churn patterns over the last year"
- "What are the main factors affecting product returns?"

**Visualizations:**
- "Create a bar chart of monthly sales for 2024"
- "Show me a pie chart of revenue breakdown by product category"
- "Plot the trend of customer acquisitions over time"

**Quick Answers:**
- "How many active customers do we have?"
- "What's our total revenue this quarter?"
- "What's the average order value?"

### 🔒 Security & Performance Features

**SQL Safety:**
- ✅ SELECT-only query validation with comprehensive blocked operation list
- ✅ Automatic LIMIT injection (default: 50 rows, max: 1000)
- ✅ Query timeout protection with MySQL execution hints
- ✅ Multiple statement detection and blocking
- ✅ SQL injection prevention through parameterized queries

**Performance Optimizations:**
- ⚡ TTL-based query result caching (1-hour default)
- ⚡ Connection pooling with pre-ping health checks
- ⚡ Retry logic for transient database errors
- ⚡ Efficient vector similarity search with ChromaDB

**Monitoring & Observability:**
- 📊 Comprehensive query statistics and success rates
- 📈 Execution time tracking and performance analytics
- 🔍 Detailed logging with structured error reporting
- 📋 Query history with search and filtering capabilities

### 🧪 Testing

Run the comprehensive test suite:
```bash
python test_nl2sql.py
```

Tests cover:
- SQL validation and safety
- Query caching functionality  
- String parsing utilities
- History management
- Vector operations
- Web API endpoints
- Full integration scenarios

### 🔧 Configuration Options

**Environment Variables:**
```bash
DATABASE_URL=your_database_connection_string
HOST=0.0.0.0              # Web server host (default: 127.0.0.1)
PORT=8080                 # Web server port (default: 8080)
DEBUG=true                # Enable debug mode
```

**Customization:**
- Modify `outputs/synonyms.json` for domain-specific term expansion
- Adjust cache TTL and limits in `QueryCache` class
- Customize visualization styles in `_maybe_visualize` method
- Add new output modes by extending the mode detection logic

### 🤝 Contributing

This project follows modern Python development practices:
- Type hints throughout codebase
- Comprehensive error handling
- Modular, testable architecture
- Clear separation of concerns
- Extensive documentation

### 📄 License

See LICENSE file for details.

---

*Built with ❤️ for data analysts, business users, and developers who want to democratize database access through natural language.*
