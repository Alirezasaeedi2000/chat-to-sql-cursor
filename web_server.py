#!/usr/bin/env python3
"""
Web interface for the Natural Language to SQL Assistant.
Provides a simple, modern web UI for interacting with the system.
"""

import logging
import os
import traceback
from datetime import datetime, UTC
from typing import Optional

from flask import Flask, render_template_string, request, jsonify, send_file
from werkzeug.exceptions import BadRequest

from query_processor import QueryProcessor, create_engine_from_env
from vector import VectorStoreManager
from query_history import QueryHistoryManager


LOGGER = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)

# Global instances
qp: Optional[QueryProcessor] = None
history: Optional[QueryHistoryManager] = None


def init_app():
    """Initialize the application components."""
    global qp, history
    try:
        engine = create_engine_from_env()
        vector = VectorStoreManager()
        qp = QueryProcessor(engine, vector)
        history = QueryHistoryManager()
        LOGGER.info("Application initialized successfully")
    except Exception as e:
        LOGGER.error(f"Failed to initialize application: {e}")
        raise


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NL to SQL Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-sql.min.js"></script>
</head>
<body class="bg-gray-50 min-h-screen">
    <div x-data="nlToSqlApp()" class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="bg-white shadow rounded-lg p-6 mb-6">
            <h1 class="text-3xl font-bold text-gray-900 mb-2">Natural Language to SQL Assistant</h1>
            <p class="text-gray-600">Convert natural language questions into SQL queries and get instant results</p>
        </div>

        <!-- Query Input -->
        <div class="bg-white shadow rounded-lg p-6 mb-6">
            <form @submit.prevent="submitQuery">
                <div class="mb-4">
                    <label for="query" class="block text-sm font-medium text-gray-700 mb-2">Your Question</label>
                    <textarea 
                        id="query" 
                        x-model="userQuery"
                        rows="3" 
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        placeholder="e.g., Show me the top 10 customers by revenue last year"
                        required
                    ></textarea>
                </div>
                
                <div class="flex flex-wrap gap-4 mb-4">
                    <div>
                        <label for="mode" class="block text-sm font-medium text-gray-700 mb-1">Mode</label>
                        <select id="mode" x-model="mode" class="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <option value="">Auto-detect</option>
                            <option value="TABLE">Table</option>
                            <option value="SHORT_ANSWER">Short Answer</option>
                            <option value="ANALYTICAL">Analytical</option>
                            <option value="VISUALIZATION">Visualization</option>
                            <option value="COMBO">Combo</option>
                        </select>
                    </div>
                    
                    <div>
                        <label for="export" class="block text-sm font-medium text-gray-700 mb-1">Export</label>
                        <select id="export" x-model="exportFormat" class="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <option value="">None</option>
                            <option value="csv">CSV</option>
                            <option value="json">JSON</option>
                        </select>
                    </div>
                </div>
                
                <button 
                    type="submit" 
                    :disabled="loading || !userQuery.trim()"
                    class="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    <span x-show="!loading">Execute Query</span>
                    <span x-show="loading">Processing...</span>
                </button>
            </form>
        </div>

        <!-- Results -->
        <div x-show="result || error" class="bg-white shadow rounded-lg p-6 mb-6">
            <!-- Error -->
            <div x-show="error" class="bg-red-50 border border-red-200 rounded-md p-4 mb-4">
                <h3 class="text-lg font-medium text-red-800 mb-2">Error</h3>
                <p class="text-red-700" x-text="error"></p>
            </div>

            <!-- Success -->
            <div x-show="result && !error">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-medium text-gray-900">Results</h3>
                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800" x-text="result?.mode"></span>
                </div>

                <!-- SQL Query -->
                <div x-show="result?.sql" class="mb-6">
                    <h4 class="text-md font-medium text-gray-700 mb-2">Generated SQL</h4>
                    <pre class="bg-gray-50 p-4 rounded-md overflow-x-auto"><code class="language-sql" x-text="result?.sql"></code></pre>
                </div>

                <!-- Table -->
                <div x-show="result?.table_markdown && (result?.mode === 'TABLE' || result?.mode === 'COMBO' || result?.mode === 'VISUALIZATION')" class="mb-6">
                    <h4 class="text-md font-medium text-gray-700 mb-2">Data Table</h4>
                    <div class="overflow-x-auto">
                        <div class="prose prose-sm max-w-none" x-html="renderMarkdown(result?.table_markdown)"></div>
                    </div>
                </div>

                <!-- Short Answer -->
                <div x-show="result?.short_answer && (result?.mode === 'SHORT_ANSWER' || result?.mode === 'COMBO')" class="mb-6">
                    <h4 class="text-md font-medium text-gray-700 mb-2">Answer</h4>
                    <p class="text-2xl font-bold text-blue-600" x-text="result?.short_answer"></p>
                </div>

                <!-- Analysis -->
                <div x-show="result?.analysis && (result?.mode === 'ANALYTICAL' || result?.mode === 'COMBO')" class="mb-6">
                    <h4 class="text-md font-medium text-gray-700 mb-2">Analysis</h4>
                    <div class="prose prose-sm max-w-none" x-html="renderMarkdown(result?.analysis)"></div>
                </div>

                <!-- Visualization -->
                <div x-show="result?.visualization_path" class="mb-6">
                    <h4 class="text-md font-medium text-gray-700 mb-2">Visualization</h4>
                    <img :src="'/static/' + result?.visualization_path" class="max-w-full h-auto rounded-lg shadow" alt="Query Visualization">
                </div>

                <!-- Metadata -->
                <div class="text-sm text-gray-500">
                    <span>Rows: </span><span x-text="result?.metadata?.row_count || 0"></span>
                    <span class="mx-2">•</span>
                    <span>Time: </span><span x-text="(result?.metadata?.execution_time_ms || 0).toFixed(1)"></span><span>ms</span>
                </div>
            </div>
        </div>

        <!-- History -->
        <div class="bg-white shadow rounded-lg p-6">
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-lg font-medium text-gray-900">Query History</h3>
                <div class="flex gap-2">
                    <button @click="loadHistory" class="text-sm text-blue-600 hover:text-blue-800">Refresh</button>
                    <button @click="clearHistory" class="text-sm text-red-600 hover:text-red-800">Clear</button>
                </div>
            </div>

            <div x-show="historyEntries.length === 0" class="text-gray-500 text-center py-8">
                No queries yet. Try asking a question above!
            </div>

            <div x-show="historyEntries.length > 0" class="space-y-3">
                <template x-for="entry in historyEntries.slice(0, 10)" :key="entry.id">
                    <div class="border border-gray-200 rounded-lg p-4 hover:bg-gray-50">
                        <div class="flex items-start justify-between mb-2">
                            <p class="font-medium text-gray-900" x-text="entry.user_query"></p>
                            <div class="flex items-center gap-2">
                                <span x-show="entry.error" class="text-red-500">✗</span>
                                <span x-show="!entry.error" class="text-green-500">✓</span>
                                <span class="text-xs text-gray-500" x-text="new Date(entry.timestamp).toLocaleString()"></span>
                            </div>
                        </div>
                        <div class="flex items-center gap-4 text-sm text-gray-500">
                            <span x-text="entry.mode"></span>
                            <span>Rows: <span x-text="entry.row_count || 0"></span></span>
                            <span>Time: <span x-text="(entry.execution_time_ms || 0).toFixed(1)"></span>ms</span>
                        </div>
                        <div x-show="entry.error" class="mt-2 text-sm text-red-600" x-text="entry.error.substring(0, 200) + (entry.error.length > 200 ? '...' : '')"></div>
                    </div>
                </template>
            </div>
        </div>
    </div>

    <script>
        function nlToSqlApp() {
            return {
                userQuery: '',
                mode: '',
                exportFormat: '',
                loading: false,
                result: null,
                error: null,
                historyEntries: [],

                init() {
                    this.loadHistory();
                },

                async submitQuery() {
                    if (!this.userQuery.trim()) return;
                    
                    this.loading = true;
                    this.result = null;
                    this.error = null;

                    try {
                        const response = await fetch('/api/query', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                query: this.userQuery,
                                mode: this.mode || null,
                                export: this.exportFormat || null
                            })
                        });

                        if (!response.ok) {
                            const errorData = await response.json();
                            throw new Error(errorData.error || 'Request failed');
                        }

                        this.result = await response.json();
                        this.loadHistory(); // Refresh history after new query
                        
                        // Highlight code
                        this.$nextTick(() => {
                            if (window.Prism) {
                                Prism.highlightAll();
                            }
                        });
                        
                    } catch (err) {
                        this.error = err.message;
                    } finally {
                        this.loading = false;
                    }
                },

                async loadHistory() {
                    try {
                        const response = await fetch('/api/history');
                        if (response.ok) {
                            this.historyEntries = await response.json();
                        }
                    } catch (err) {
                        console.error('Failed to load history:', err);
                    }
                },

                async clearHistory() {
                    if (!confirm('Are you sure you want to clear all query history?')) return;
                    
                    try {
                        const response = await fetch('/api/history', { method: 'DELETE' });
                        if (response.ok) {
                            this.historyEntries = [];
                        }
                    } catch (err) {
                        console.error('Failed to clear history:', err);
                    }
                },

                renderMarkdown(text) {
                    if (!text) return '';
                    // Simple markdown-to-HTML conversion for tables
                    return text
                        .replace(/\\\\n/g, '\\n')
                        .replace(/\\|/g, '|')
                        .replace(/^(.+)$/gm, '<div>$1</div>')
                        .replace(/<div>(\\|[^|]+\\|.*)<\\/div>/g, '<table class="min-w-full divide-y divide-gray-200"><tbody>$1</tbody></table>')
                        .replace(/\\|([^|]*)\\|/g, '<td class="px-3 py-2 border">$1</td>')
                        .replace(/<td class="px-3 py-2 border">([^<]+)<\\/td>/g, (match, content) => {
                            return content.trim().startsWith('-') ? 
                                '<td class="px-3 py-2 border bg-gray-50"></td>' : 
                                '<td class="px-3 py-2 border">' + content + '</td>';
                        });
                }
            };
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Serve the main application page."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/query", methods=["POST"])
def api_query():
    """Handle query execution API endpoint."""
    if not qp:
        return jsonify({"error": "Application not initialized"}), 500

    try:
        data = request.get_json()
        if not data or "query" not in data:
            raise BadRequest("Missing query parameter")

        user_query = data["query"].strip()
        if not user_query:
            raise BadRequest("Query cannot be empty")

        mode = data.get("mode")
        export_format = data.get("export")

        # Process the query
        result = qp.process(user_query, prefer_mode=mode, export=export_format)

        # Convert to JSON-serializable format
        response_data = {
            "mode": result.mode,
            "sql": result.sql,
            "table_markdown": result.table_markdown,
            "short_answer": result.short_answer,
            "analysis": result.analysis,
            "visualization_path": os.path.basename(result.visualization_path)
            if result.visualization_path
            else None,
            "metadata": result.metadata,
        }

        return jsonify(response_data)

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        LOGGER.error(f"Query processing failed: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Query processing failed: {str(e)}"}), 500


@app.route("/api/history", methods=["GET"])
def api_history_get():
    """Get query history."""
    if not history:
        return jsonify({"error": "History not initialized"}), 500

    try:
        entries = history.search_history(limit=50)
        return jsonify(
            [
                {
                    "id": entry.id,
                    "timestamp": entry.timestamp,
                    "user_query": entry.user_query,
                    "sql_query": entry.sql_query,
                    "mode": entry.mode,
                    "execution_time_ms": entry.execution_time_ms,
                    "row_count": entry.row_count,
                    "error": entry.error,
                    "is_favorite": entry.is_favorite,
                    "tags": entry.tags,
                }
                for entry in entries
            ]
        )
    except Exception as e:
        LOGGER.error(f"Failed to get history: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["DELETE"])
def api_history_clear():
    """Clear query history."""
    if not history:
        return jsonify({"error": "History not initialized"}), 500

    try:
        history.clear_history()
        return jsonify({"status": "success"})
    except Exception as e:
        LOGGER.error(f"Failed to clear history: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static files (visualizations)."""
    if filename.startswith("viz_") and filename.endswith(".png"):
        full_path = os.path.join("outputs", "plots", filename)
        if os.path.exists(full_path):
            return send_file(full_path, mimetype="image/png")

    return jsonify({"error": "File not found"}), 404


@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "initialized": qp is not None and history is not None,
        }
    )


def main():
    """Main entry point for the web server."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    try:
        init_app()

        port = int(os.environ.get("PORT", 8080))
        host = os.environ.get("HOST", "127.0.0.1")
        debug = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")

        LOGGER.info(f"Starting web server on {host}:{port}")
        app.run(host=host, port=port, debug=debug)

    except Exception as e:
        LOGGER.error(f"Failed to start web server: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
