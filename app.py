import argparse
import re
import logging
import os
import sys
from typing import Optional

from query_processor import QueryProcessor, create_engine_from_url
from vector import VectorStoreManager
from query_history import QueryHistoryManager


def setup_logging() -> None:
    os.makedirs("outputs/logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("outputs/logs/nl2sql.log", encoding="utf-8"),
        ],
    )


def run_cli(
    db_url: str,
    query: Optional[str],
    mode: Optional[str],
    export: Optional[str],
    build_index: bool,
) -> int:
    setup_logging()
    logger = logging.getLogger("cli")

    engine = create_engine_from_url(db_url)
    vector = VectorStoreManager()
    if build_index:
        added = vector.upsert_schema_and_samples(engine, sample_rows_per_table=5)
        logger.info("Indexed %d items into vector store", added)

    qp = QueryProcessor(engine, vector)

    if query:
        result = qp.process(query, prefer_mode=mode, export=export)
        print("Mode:", result.mode)
        if result.sql:
            print("SQL:")
            print(result.sql)
        if result.table_markdown and result.mode in ("TABLE", "COMBO", "VISUALIZATION"):
            print("\nTable:")
            print(result.table_markdown)
        if result.short_answer and result.mode in ("SHORT_ANSWER", "COMBO"):
            print("\nShort Answer:")
            print(result.short_answer)
        if result.analysis and result.mode in ("ANALYTICAL", "COMBO"):
            print("\nAnalytical:")
            print(result.analysis)
        if result.visualization_path:
            print("\nVisualization saved to:", result.visualization_path)
        return 0

    print("Interactive mode. Type 'exit' to quit.")
    while True:
        try:
            user_q = input("NL> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_q.lower() in {"exit", "quit"}:
            break
        if not user_q:
            continue
        # Per-query inline mode override: "mode:COMBO your query"
        prefer_mode_iter = mode
        m = re.match(r"^\s*mode\s*:\s*([a-z_ ]+)\s*(.*)$", user_q, flags=re.IGNORECASE)
        if m:
            requested = m.group(1).strip().upper().replace(" ", "_")
            if requested in {
                "TABLE",
                "SHORT_ANSWER",
                "ANALYTICAL",
                "VISUALIZATION",
                "COMBO",
            }:
                prefer_mode_iter = requested
                remainder = m.group(2).strip()
                if remainder:
                    user_q = remainder
        res = qp.process(user_q, prefer_mode=prefer_mode_iter, export=export)
        print("Mode:", res.mode)
        if res.sql:
            print("SQL:\n", res.sql)
        if res.table_markdown and res.mode in ("TABLE", "COMBO", "VISUALIZATION"):
            print("\nTable:\n", res.table_markdown)
        if res.short_answer and res.mode in ("SHORT_ANSWER", "COMBO"):
            print("\nShort Answer:\n", res.short_answer)
        if res.analysis and res.mode in ("ANALYTICAL", "COMBO"):
            print("\nAnalytical:\n", res.analysis)
        if res.visualization_path:
            print("\nVisualization saved to:", res.visualization_path)
    return 0


def handle_history_commands(args) -> int:
    """Handle history-related commands."""
    setup_logging()
    history = QueryHistoryManager()

    if args.history_clear:
        history.clear_history()
        print("Query history cleared.")
        return 0

    if args.history_stats:
        stats = history.get_statistics()
        print("\n=== Query Statistics ===")
        print(f"Total queries: {stats['total_queries']}")
        print(f"Successful: {stats['successful_queries']}")
        print(f"Failed: {stats['failed_queries']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Average execution time: {stats['average_execution_time_ms']:.1f}ms")
        print(f"Recent queries (7 days): {stats['recent_queries_7_days']}")
        print(f"Favorites: {stats['favorites_count']}")

        if stats["mode_distribution"]:
            print("\nMode distribution:")
            for mode, count in stats["mode_distribution"].items():
                print(f"  {mode}: {count}")

        print("\n=== Popular Query Patterns ===")
        popular = history.get_popular_queries()
        for i, pattern in enumerate(popular, 1):
            print(
                f"{i}. {pattern['example']} (used {pattern['count']} times, {pattern['success_rate']:.1%} success)"
            )

        return 0

    if args.history_export:
        try:
            filepath = history.export_history(args.history_export)
            print(f"History exported to: {filepath}")
        except Exception as e:
            print(f"Export failed: {e}")
            return 1
        return 0

    # Show recent history
    search_term = args.history_search if args.history_search else ""
    entries = history.search_history(query=search_term, limit=20)

    if not entries:
        print("No history entries found.")
        return 0

    print(
        f"\n=== Query History {'(Search: ' + search_term + ')' if search_term else ''} ==="
    )
    for entry in entries:
        status = "✓" if entry.error is None else "✗"
        favorite = "★" if entry.is_favorite else " "
        timestamp = entry.timestamp.split("T")[0]  # Just date

        print(f"\n{status} {favorite} [{entry.id}] {timestamp}")
        print(f"Query: {entry.user_query}")
        if entry.sql_query:
            print(
                f"SQL: {entry.sql_query[:100]}{'...' if len(entry.sql_query) > 100 else ''}"
            )
        print(
            f"Mode: {entry.mode} | Rows: {entry.row_count or 0} | Time: {entry.execution_time_ms or 0:.1f}ms"
        )
        if entry.tags:
            print(f"Tags: {', '.join(entry.tags)}")
        if entry.error:
            print(
                f"Error: {entry.error[:200]}{'...' if len(entry.error) > 200 else ''}"
            )

    return 0


def run_web_server() -> int:
    """Run the web interface server."""
    setup_logging()
    from web_server import main as web_main

    return web_main()


def run_server() -> int:
    setup_logging()
    # Run the MCP stdio server in this process
    from mcp_sql_server import main as mcp_main
    import asyncio

    asyncio.run(mcp_main())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Natural Language to SQL Assistant for MySQL"
    )
    parser.add_argument(
        "--db-url",
        help="SQLAlchemy DB URL, e.g., mysql+pymysql://user:pass@host:3306/db",
        required=False,
    )
    parser.add_argument("--query", help="One-shot NL query", required=False)
    parser.add_argument(
        "--mode",
        help="Force mode: TABLE|SHORT_ANSWER|ANALYTICAL|VISUALIZATION|COMBO",
        required=False,
    )
    parser.add_argument("--export", help="Export format: csv|json", required=False)
    parser.add_argument(
        "--build-index", help="Build or refresh the vector index", action="store_true"
    )
    parser.add_argument("--server", help="Run MCP stdio server", action="store_true")
    parser.add_argument("--web", help="Run web interface", action="store_true")

    # History commands
    parser.add_argument("--history", help="Show query history", action="store_true")
    parser.add_argument("--history-search", help="Search query history", type=str)
    parser.add_argument(
        "--history-stats", help="Show query statistics", action="store_true"
    )
    parser.add_argument("--history-export", help="Export history (json|csv)", type=str)
    parser.add_argument(
        "--history-clear", help="Clear all history", action="store_true"
    )

    args = parser.parse_args()

    if args.server:
        return run_server()

    if args.web:
        return run_web_server()

    # Handle history commands that don't need database
    if (
        args.history
        or args.history_search
        or args.history_stats
        or args.history_export
        or args.history_clear
    ):
        return handle_history_commands(args)

    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        print("Please provide --db-url or set DATABASE_URL.")
        return 2

    return run_cli(db_url, args.query, args.mode, args.export, args.build_index)


if __name__ == "__main__":
    raise SystemExit(main())
