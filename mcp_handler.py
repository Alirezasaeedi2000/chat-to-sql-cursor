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
        "You are an expert SQL generator. Convert the natural language question into a single, precise MySQL SELECT query.\n"
        "\nCRITICAL ANALYSIS:\n"
        "1. READ THE QUESTION WORD BY WORD\n"
        "2. IDENTIFY the key intent: COUNT, GROUP BY, ORDER BY, simple SELECT, or CHART\n"
        "3. MATCH SQL structure to the question type\n"
        "\nQUESTION PATTERNS & REQUIRED SQL:\n"
        "- 'how many employees' → SELECT COUNT(*) FROM `employe` (not `employee_projects`!)\n"
        "- 'how many projects' → SELECT COUNT(*) FROM `projects` (do NOT join to `employee_projects`)\n"
        "- 'count by department' → SELECT d.name, COUNT(*) FROM `employe` e JOIN `departments` d ON e.department_id=d.id GROUP BY d.name\n"
        "- 'top X employees by salary' → SELECT first_name, last_name, salary FROM `employe` ORDER BY salary DESC LIMIT X\n"
        "- 'top X employees by salary in each department' → Use window function: ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) and filter <= X; join to `departments` for names if needed\n"
        "- 'average salary by department' → SELECT d.name, AVG(e.salary) FROM `employe` e JOIN `departments` d ON e.department_id=d.id GROUP BY d.name\n"
        "- 'list departments' → SELECT name FROM `departments`\n"
        "- 'pie chart of project budgets' → SELECT name, budget FROM `projects` (no JOIN needed unless budget is external)\n"
        "- 'histogram of employee salaries' → SELECT salary FROM `employe` (return a single numeric column suitable for a histogram)\n"
        "\nSTRICT RULES:\n"
        "- Use ONLY table/column names from the provided context\n"
        "- JOIN only when absolutely necessary for the question (e.g., to fetch a name from another table)\n"
        "- For COUNT questions, always COUNT from the base entity table mentioned by the user (e.g., projects → `projects`)\n"
        "- For per-group top-X, prefer window functions on the base entity table and lightweight joins for labels (e.g., departments)\n"
        "- For pie charts, return two columns: label and value. Prefer a direct table with numeric value (e.g., project name + budget). Avoid dividing by employee counts or joining unless required by the question.\n"
        "- Keep queries as simple as possible\n"
        "- Always add LIMIT (default 50)\n"
        "- Use backticks around identifiers\n"
        "- Output ONLY SQL in ```sql fences\n"
        "\nFORBIDDEN:\n"
        "- Do NOT add unnecessary JOINs\n"
        "- Do NOT join to unrelated tables for the metric (e.g., don't join `projects` to get salaries)\n"
        "- Do NOT use complex subqueries unless required\n"
        "- Do NOT guess column names not in context\n"
        "- Do NOT turn a simple total (COUNT(*)) into a JOIN-based count on link tables\n"
    )


def build_sql_repair_prompt() -> str:
    return (
        "The previous SQL failed. Produce a corrected MySQL SELECT.\n"
        "Constraints:\n"
        "- Single statement, SELECT-only.\n"
        "- Keep semantics close to the question.\n"
        "- Use backticks around identifiers.\n"
        "- Output ONLY the SQL inside ```sql fences.\n"
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
