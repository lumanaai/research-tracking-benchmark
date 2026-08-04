import json
import sqlite3
from pathlib import Path

# Set this to True to generate all tables in a single file called analytic_db.sqlite
single_file_mode = True  # Change to False for file-per-table mode


def infer_sqlite_type(value):
    if isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "REAL"
    elif isinstance(value, (bytes, bytearray)):
        return "BLOB"
    elif value is None:
        return "TEXT"  # Default to TEXT for None
    elif isinstance(value, dict) and value.get("type") == "Buffer" and isinstance(value.get("data"), list):
        return "BLOB"
    else:
        return "TEXT"

def sanitize_value(val):
    # Handle Buffer dict to bytes
    if isinstance(val, dict) and val.get("type") == "Buffer" and isinstance(val.get("data"), list):
        return bytes(val["data"])
    # Handle dict/list as JSON string
    if isinstance(val, (dict, list)):
        return json.dumps(val)
    # Handle bytes/bytearray
    elif isinstance(val, (bytes, bytearray)):
        return val
    # Handle base64-encoded strings for blobs
    elif isinstance(val, str):
        return val
    # Handle None
    elif val is None:
        return None
    # Handle tuples/sets as JSON string
    elif isinstance(val, (tuple, set)):
        return json.dumps(list(val))
    # Fallback: convert to string
    else:
        return str(val)

def main(db_file_in):
    # Load JSON
    data = None
    with open(db_file_in) as f:
        data = json.load(f)

    metadata = data.pop("metadata", {})

    if single_file_mode:
        db_file = db_file_in.parent.joinpath("analytic.db")
        con = sqlite3.connect(db_file)
        cur = con.cursor()

    for table_name, rows in data.items():
        if not single_file_mode:
            db_file = db_file_in.parent.joinpath(f"{table_name}.sqlite")
            con = sqlite3.connect(db_file)
            cur = con.cursor()
        # Infer columns and types from first row, or fallback to id only
        if rows and len(rows) > 0:
            first_row = rows[0]
            columns = list(first_row.keys())
            col_types = [infer_sqlite_type(first_row[col]) for col in columns]
            col_defs = [f'{col} {col_type}' for col, col_type in zip(columns, col_types)]
            # Add id INTEGER PRIMARY KEY AUTOINCREMENT if not present
            if "id" not in columns:
                col_defs.insert(0, "id INTEGER PRIMARY KEY AUTOINCREMENT")
        else:
            # No rows: create table with only id column
            col_defs = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
        create_stmt = f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join(col_defs)})'
        cur.execute(create_stmt)
        # Insert data if any
        if rows and len(rows) > 0:
            placeholders = ", ".join(["?" for _ in columns])
            insert_stmt = f'INSERT INTO {table_name} ({", ".join(columns)}) VALUES ({placeholders})'
            for row in rows:
                try:
                    values = [sanitize_value(row.get(col)) for col in columns]
                    cur.execute(insert_stmt, values)
                except Exception as e:
                    print(f"Error inserting row in table '{table_name}': {row}\nColumn types: {[type(row.get(col)) for col in columns]}\nException: {e}")
                    raise
        if not single_file_mode:
            con.commit()
            con.close()
    if single_file_mode:
        con.commit()
        con.close()

if __name__ == "__main__":
    db_file = Path(__file__).parent.parent.joinpath("analyzer_manager", "app", "assets", "test_device", "analytics_export_2025-11-12T10-03-29-218Z.json")
    main(db_file)