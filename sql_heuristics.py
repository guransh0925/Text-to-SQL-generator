import re


def _split_columns(column_block):
    columns = []
    for raw_column in column_block.split(","):
        column = raw_column.strip().split()
        if column:
            columns.append(column[0].strip('"`[]'))
    return columns


def parse_schema(schema):
    tables = {}

    for match in re.finditer(
        r"CREATE\s+TABLE\s+([A-Za-z_][\w]*)\s*\((.*?)\)\s*;",
        schema,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        tables[match.group(1).lower()] = _split_columns(match.group(2))

    for match in re.finditer(
        r"Table:\s*([A-Za-z_][\w]*)\s*\n\s*Columns:\s*([^\n]+)",
        schema,
        flags=re.IGNORECASE,
    ):
        tables[match.group(1).lower()] = [
            column.strip().strip('"`[]')
            for column in match.group(2).split(",")
            if column.strip()
        ]

    return tables


def _column_for_phrase(columns, phrases):
    lower_to_original = {column.lower(): column for column in columns}
    for phrase in phrases:
        if phrase in lower_to_original:
            return lower_to_original[phrase]
    return None


def _number_after(question, words):
    pattern = r"(?:%s)\s+\$?([0-9][0-9,]*(?:\.\d+)?)" % "|".join(words)
    match = re.search(pattern, question, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).replace(",", "")


def _quote(value):
    return "'" + value.replace("'", "''") + "'"


def _status_value(question):
    if "completed" in question:
        return "completed"
    if "pending" in question:
        return "pending"
    if "cancelled" in question or "canceled" in question:
        return "cancelled"
    if "shipped" in question:
        return "shipped"
    return None


def _city_value(question):
    match = re.search(r"\b(?:from|in)\s+([A-Za-z][A-Za-z\s]+?)(?:\.|$)", question, flags=re.IGNORECASE)
    if not match:
        return None
    city = match.group(1).strip()
    blocked = {"the sales department", "sales department", "department"}
    return None if city.lower() in blocked else city


def _table_columns(tables, table_name):
    return tables.get(table_name, [])


def _has_table(tables, table_name):
    return table_name in tables


def _singular(name):
    return name[:-1] if name.endswith("s") else name


def _mentioned_table(tables, question):
    for table in tables:
        table_words = {table, _singular(table)}
        if any(re.search(rf"\b{re.escape(word)}\b", question, flags=re.IGNORECASE) for word in table_words):
            return table
    return next(iter(tables)) if len(tables) == 1 else None


def _quoted_value_after_column(question, column):
    pattern = rf"\b{re.escape(column)}\b\s*(?:=|is|named|with|as)?\s*[\"']([^\"']+)[\"']"
    match = re.search(pattern, question, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _bare_value_after_column(question, column):
    pattern = rf"\b{re.escape(column)}\b\s*(?:=|is|named|with|as)\s+([A-Za-z][\w-]*)"
    match = re.search(pattern, question, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _number_value_after_column(question, column):
    pattern = rf"\b{re.escape(column)}\b\s*(?:=|is|with|as)?\s+\$?([0-9][0-9,]*(?:\.\d+)?)"
    match = re.search(pattern, question, flags=re.IGNORECASE)
    return match.group(1).replace(",", "") if match else None


def _generic_conditions(columns, question):
    conditions = []
    q = question.lower()

    for column in columns:
        column_lower = column.lower()
        value = _quoted_value_after_column(question, column_lower) or _bare_value_after_column(question, column_lower)
        if value:
            conditions.append(f"{column} = {_quote(value)}")
            continue

        number_value = _number_value_after_column(question, column_lower)
        if number_value:
            conditions.append(f"{column} = {number_value}")
            continue

        number = _number_after(q, [f"{column_lower} over", f"{column_lower} above", f"{column_lower} greater than"])
        if number:
            conditions.append(f"{column} > {number}")

    return conditions


def _generic_simple_sql(tables, question):
    table = _mentioned_table(tables, question)
    if not table:
        return None

    columns = tables[table]
    conditions = _generic_conditions(columns, question)

    select_expr = "*"
    for column in columns:
        if column.lower() in question.lower() and re.search(r"\b(show|select|get|list)\b", question, flags=re.IGNORECASE):
            if " with " not in question.lower() and " where " not in question.lower():
                select_expr = column
            break

    sql = f"SELECT {select_expr} FROM {table}"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    return sql + ";"


def heuristic_sql(schema, question):
    tables = parse_schema(schema)
    if not tables:
        return None

    q = question.strip().lower()

    if _has_table(tables, "orders") and "order" in q:
        columns = _table_columns(tables, "orders")
        where = []

        status_col = _column_for_phrase(columns, ["status"])
        status = _status_value(q)
        if status_col and status:
            where.append(f"{status_col} = {_quote(status)}")

        total_col = _column_for_phrase(columns, ["total", "amount", "order_total"])
        total = _number_after(q, ["over", "above", "greater than", "more than"])
        if total_col and total:
            where.append(f"{total_col} > {total}")

        sql = "SELECT * FROM orders"
        if where:
            sql += " WHERE " + " AND ".join(where)
        return sql + ";"

    if _has_table(tables, "customers") and "customer" in q:
        columns = _table_columns(tables, "customers")
        select_col = _column_for_phrase(columns, ["name", "customer_name"])
        select_expr = select_col if ("name" in q and select_col) else "*"
        where = []

        city_col = _column_for_phrase(columns, ["city"])
        city = _city_value(q)
        if city_col and city:
            where.append(f"{city_col} = {_quote(city)}")

        sql = f"SELECT {select_expr} FROM customers"
        if where:
            sql += " WHERE " + " AND ".join(where)
        return sql + ";"

    if _has_table(tables, "employees") and "employee" in q:
        columns = _table_columns(tables, "employees")
        name_col = _column_for_phrase(columns, ["employee_name", "name"])
        salary_col = _column_for_phrase(columns, ["salary"])
        department_col = _column_for_phrase(columns, ["department"])

        select_expr = "*"
        if "name" in q and "salary" in q and name_col and salary_col:
            select_expr = f"{name_col}, {salary_col}"
        elif "name" in q and name_col:
            select_expr = name_col

        where = []
        salary = _number_after(q, ["over", "above", "greater than", "more than", "earning more than"])
        if salary_col and salary:
            where.append(f"{salary_col} > {salary}")

        department_match = re.search(r"\b(?:in|from)\s+the\s+([A-Za-z]+)\s+department\b", q)
        if department_col and department_match:
            where.append(f"{department_col} = {_quote(department_match.group(1))}")

        sql = f"SELECT {select_expr} FROM employees"
        if where:
            sql += " WHERE " + " AND ".join(where)
        return sql + ";"

    return _generic_simple_sql(tables, question)
