from typing import List, Tuple


def build_intent_prompt() -> Tuple[str, List[str]]:
    system = (
        "You are an expert MySQL analyst. Analyze the user's question and determine the SINGLE best response format.\n"
        "Output ONLY a JSON object with keys: mode and reason.\n"
        "\nSTRICT MODE RULES:\n"
        "- SHORT_ANSWER: Questions asking 'how many', 'what is total', 'count of' - user wants ONE number\n"
        "- TABLE: Questions asking 'show me', 'list', 'top X', 'display' - user wants detailed rows\n"
        "- VISUALIZATION: Questions asking 'plot', 'chart', 'graph', 'distribution', 'count by' - user wants visual\n"
        "- ANALYTICAL: Questions asking 'compare', 'analyze', 'insights', 'trends' - user wants analysis\n"
        "- COMBO: Only for very complex questions explicitly asking for multiple outputs\n"
        "\nDO NOT default to COMBO unless question explicitly needs multiple views!"
    )
    few_shots = [
        '{"mode": "SHORT_ANSWER", "reason": "Question asks how many - single number expected."}',
        '{"mode": "TABLE", "reason": "Question asks to show/list - tabular data expected."}',
        '{"mode": "VISUALIZATION", "reason": "Question asks for plot/chart - visual output expected."}',
        '{"mode": "ANALYTICAL", "reason": "Question asks to compare/analyze - insights expected."}',
        '{"mode": "TABLE", "reason": "Question asks for top X - ranked list expected."}',
    ]
    return system, few_shots


def build_sql_prompt() -> str:
    return (
        "You are a SQL expert. Generate ONLY a MySQL SELECT query.\n"
        "\nCRITICAL: Output ONLY SQL code wrapped in ```sql``` fences. No explanations, no text, no analysis.\n"
        "\nSCHEMA CONTEXT:\n"
        "Tables: production_info, person_hyg, packaging_info, pack_waste, production_test, workers, prices, packs, repo_nc, transtatus, users\n"
        "\nEXAMPLES:\n"
        "Question: How many workers?\n"
        "SQL: ```sql\nSELECT COUNT(*) FROM workers LIMIT 50\n```\n"
        "\nQuestion: Production volumes this month?\n"
        "SQL: ```sql\nSELECT SUM(totalUsage) FROM production_info WHERE MONTH(date) = MONTH(CURDATE()) LIMIT 50\n```\n"
        "\nQuestion: Show packaging types?\n"
        "SQL: ```sql\nSELECT bakeType, COUNT(*) FROM packaging_info GROUP BY bakeType ORDER BY COUNT(*) DESC LIMIT 50\n```\n"
        "\nQuestion: Waste distribution by type?\n"
        "SQL: ```sql\nSELECT type, COUNT(*) FROM pack_waste GROUP BY type LIMIT 50\n```\n"
        "\nRULES:\n"
        "- Use ONLY the provided schema context\n"
        "- Always add LIMIT 50\n"
        "- Use backticks around table/column names\n"
        "- Output ONLY SQL in ```sql``` fences\n"
        "- NO explanatory text\n"
    )


def build_sql_repair_prompt() -> str:
    return (
        "The previous SQL failed. Produce a corrected MySQL SELECT query.\n"
        "\nCRITICAL: Generate ONLY SQL code, no explanations or text.\n"
        "\nConstraints:\n"
        "- Single statement, SELECT-only.\n"
        "- Keep semantics close to the original question.\n"
        "- Use backticks around identifiers.\n"
        "- Use ONLY table/column names from the provided schema context.\n"
        "- Always add LIMIT (default 50).\n"
        "- Output ONLY the SQL inside ```sql fences.\n"
        "\nFORBIDDEN:\n"
        "- Do NOT generate explanatory text\n"
        "- Do NOT add comments or analysis\n"
        "- Do NOT guess column names not in context\n"
    )


def build_analytical_prompt() -> str:
    return (
        "You are a data analyst. Analyze the SQL results and write a focused, accurate analysis.\n"
        "\nSTRUCTURE: Insights, Gaps, Risks, Recommendations\n"
        "\nCRITICAL RULES:\n"
        "- Base insights ONLY on the actual SQL results shown\n"
        "- Do NOT mention generic things like 'concentration risk in two regions' unless relevant\n"
        "- Do NOT mention 'February dip' unless the data actually shows time-series information\n"
        "- Focus on what the numbers actually reveal\n"
        "- Be specific to the query and results\n"
        "\nEXAMPLE GOOD INSIGHTS:\n"
        "- For salary data: 'Average salary in HR ($91K) is highest, Sales lowest ($89K)'\n"
        "- For counts: 'Engineering has most employees (634), Sales has fewest (555)'\n"
        "- For distributions: 'Top 10% of salaries range from $119K to $120K'\n"
        "\nAVOID GENERIC STATEMENTS:\n"
        "- Don't mention regions unless location data is shown\n"
        "- Don't mention February unless time data is present\n"
        "- Don't mention overtime unless hours data is shown\n"
    )