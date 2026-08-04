import csv
import datetime
from typing import Dict
import pytz
import pymysql


# Define your connection details
HOST = "svc-a519c79a-48fd-4531-9e71-5220fc82e61c-dml.gcp-iowa-1.svc.singlestore.com"
PORT = 3306  # Default port for SingleStore
USER = "prod-readonly"
PASSWORD = "DAI4VhetILRgc6CFXY5Dix9O7VbWw9SZ"
DATABASE = "lumix_production"
base_path = "/mnt/c/Temp/blacklist/"

training_bucket = "lumixai-training-thumbnails"


def timestamp_to_tz(timestamp: int, tz: str = "US/Eastern"):
    return datetime.datetime.fromtimestamp(timestamp // 1000).astimezone(pytz.timezone(tz)).replace(tzinfo=None)


def run_singlestore_query(query) -> Dict[str, Dict]:
    # Connect to the SingleStore database
    query_results = {}
    connection = pymysql.connect(
        host=HOST, port=PORT, user=USER, password=PASSWORD, database=DATABASE, cursorclass=pymysql.cursors.DictCursor
    )
    try:
        with connection.cursor() as cursor:
            # Execute the query
            cursor.execute(query)
            # Fetch all rows (use fetchone() to get one row, or fetchmany(size) to get a specified number of rows)
            query_results = cursor.fetchall()
            # Printing the results (you might want to process these differently)
            # result_parsing_quarry(results, local_path)
    finally:
        # Close the connection
        connection.close()
    return query_results  # noqa


def save_dict_list_to_csv(data, file_name, delimiter=","):
    keys = data[0].keys()
    with open(file_name, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys, delimiter=delimiter)
        dict_writer.writeheader()
        dict_writer.writerows(data)


def datetime_to_timestamp(year, month, day, hour, minute):
    return int(pytz.UTC.localize(datetime.datetime(year, month, day, hour, minute)).timestamp()) * 1000
