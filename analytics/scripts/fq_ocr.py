import subprocess
import os
import glob
import json
import datetime
from typing import List

import pytz
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

dictionary = {
    "CMPC HW": ["cmpc", "g3109380", "guaiba", "eucalyptus", "brazil"],
    "Domtar Dryden SW": ["domatar", "sdomatar", "dryden", "drvden"],
    "International Paper SW": ["international", "grande", "prairie"],
    "Metsa SW": ["metsa", "pine", "rma", "71602"],
    "Resolute Forest Products": ["resolute", "forest", "products", "thunder", "softwood"],
    "SCA Pure": ["sca", "pure"],
}

unrecognized_str = "Unrecognized"

def smallest_distance(data_str, query_str):
    m, n = len(data_str), len(query_str)

    # Initialize a matrix to store Levenshtein distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialization for base cases
    for i in range(n + 1):
        dp[0][i] = i

    min_distance = n * m

    # Build the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if data_str[i - 1] == query_str[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

        # Update min_distance with the minimum value in the current row
        min_distance = min(min_distance, dp[i][n])

    return min_distance


def is_close_match(line: List[str], query: str):
    # both query and line assumed to be lower case
    dist = smallest_distance("".join(line).lower(), query)
    return dist < 0.25 * len(query)


def utc_now_milliseconds():
    utc_now = datetime.datetime.now(pytz.utc)
    return int(utc_now.timestamp() * 1000)


def read_first_alert_file():
    # Use glob to find all files that match the pattern "alert_*.json"
    files = glob.glob(os.path.join(base_dir, "alert_*.json"))

    if not files:
        print("No matching files found!")
        return None

    # Read the first file found
    with open(files[0], "r") as f:
        content = json.load(f)

    return content


def delete_alert_files():
    # Use glob to find all files that match the pattern "alert_*.json"
    files_to_delete = glob.glob("alert_*.json")

    # Iterate over each matching file and delete it
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")


def sorted_data(data):
    data = [
        {
            "timestamp": item["timestamp"],
            "msg": item["alertMessage"],
            "text": [text.lower() for text in item["alertData"]["text"]],
            "confidence": item["alertData"]["confidence"],
        }
        for item in data
        if "alertData" in item
    ]
    sorted_data = sorted(data, key=lambda x: x["timestamp"])

    return sorted_data


def filter_by_dictionary(data):
    filtered_data = []
    match_by_patt = 0
    match_by_in = 0
    no_match = 0

    for idx, item in enumerate(data):
        txt = [t.lower() for t in item["text"]]
        key_found = False
        for key in dictionary.keys():
            is_match = False
            for word in dictionary[key]:
                if word in txt:
                    match_by_in += 1
                    is_match = True
                    break
                elif is_close_match(txt, word):
                    match_by_patt += 1
                    is_match = True
                    break
            if is_match:
                filtered_data.append({"timestamp": item["timestamp"], "key": key, "confidence": item["confidence"]})
                key_found = True
                break
        if not key_found:
            filtered_data.append({"timestamp": item["timestamp"], "key": unrecognized_str, "confidence": item["confidence"]})
            no_match += 1

    print(f"stats: pattern {match_by_patt}, in {match_by_in}, no match {no_match}")
    return filtered_data


def filter_by_start_end(data):
    filtered_data = []
    for i, item in enumerate(data):

        # Check timestamp difference and average confidence for subsequent elements
        if item["timestamp"] >= start_time and item["timestamp"] <= end_time:
            filtered_data.append(item)
    return filtered_data


def filter_by_time(data):
    filtered_data = []
    last_recognized_idx = -1
    package_switch_duration = 7000
    valid_package_duration = 3000
    allow_unrecognized = False
    for i, item in enumerate(data):

        # Check for the first element
        if last_recognized_idx < 0:
            if item["key"] != unrecognized_str:
                filtered_data.append(item)
                last_recognized_idx = i
            continue

        if item["key"] == unrecognized_str:
            time_diff = item["timestamp"] - data[i - 1]["timestamp"]
            if time_diff >= package_switch_duration:
                if allow_unrecognized:
                    filtered_data.append(item)
        else:
            time_diff = item["timestamp"] - data[last_recognized_idx]["timestamp"]
            # Check timestamp difference and average confidence for subsequent elements
            if time_diff >= package_switch_duration:
                if filtered_data[-1]["key"] == unrecognized_str and item["timestamp"] - filtered_data[-1]["timestamp"] < package_switch_duration:
                    filtered_data.pop(-1)
                filtered_data.append(item)
            last_recognized_idx = i

    return filtered_data


def find_in_msg(data, value):
    for item in data:
        if value in item["msg"]:
            print(item)


def timestamp_to_edt(timestamp: int):
    return (
        datetime.datetime.fromtimestamp(timestamp // 1000).astimezone(pytz.timezone("US/Eastern")).replace(tzinfo=None)
    )


def port_to_excel(data):
    data_to_excel = []
    for item in data:
        utc_dt = datetime.datetime.utcfromtimestamp(item["timestamp"] / 1000)  # Convert to seconds first
        utc_dt = pytz.utc.localize(utc_dt)

        # Convert UTC to EDT
        edt_dt = utc_dt.astimezone(pytz.timezone("US/Eastern"))
        edt_dt_naive = edt_dt.replace(tzinfo=None)

        data_to_excel.append({"EDT time": edt_dt_naive, "timestamp": item["timestamp"], "key": item["key"]})
    df = pd.DataFrame(data_to_excel)

    # Save to Excel
    df.to_excel("output.xlsx", index=False, engine="openpyxl")


if __name__ == "__main__":

    read_from_mongo = False
    base_dir = "/mnt/c/Temp/ocr"

    # make sure that the dictionary is lower case
    for k in dictionary:
        dictionary[k] = [t.lower() for t in dictionary[k]]

    if read_from_mongo:
        print("Deleting old jsons")
        delete_alert_files()
        duration = input("Enter the period in seconds: ")

        end_time = utc_now_milliseconds()  # current time in milliseconds
        start_time = end_time - int(duration) * 1000

        # Run the JavaScript script using Node.js with startTime and endTime as arguments

        result = subprocess.run(["node", "alerts.js", str(start_time), str(end_time)], capture_output=True, text=True)

        # Print the output (and errors if any)
        print(result.stdout)
        if result.stderr:
            print("Error:", result.stderr)
    if not read_from_mongo or not result.stderr:

        # Read the json
        data = read_first_alert_file()
        if data:
            # data = filter_by_start_end(data)
            sdata = sorted_data(data)
            dict_data = filter_by_dictionary(sdata)
            time_data = filter_by_time(dict_data)

            start_time = time_data[0]["timestamp"]
            end_time = time_data[-1]["timestamp"]
            print(f"collected {len(time_data)} over {(end_time - start_time) / 1000} seconds")

            unique_texts = sorted(dictionary.keys())
            text_to_num = {text: i for i, text in enumerate(unique_texts)}
            text_to_num[unrecognized_str] = len(unique_texts)
            for item in time_data:
                item["text_num"] = text_to_num[item["key"]]
            x_values = [timestamp_to_edt(item["timestamp"]) for item in time_data]
            y_values = [item["text_num"] for item in time_data]

            plt.figure(figsize=(10, 6))
            plt.scatter(x_values, y_values, marker="o", color="blue")
            # for i, txt in enumerate([item['key'] for item in data]):
            #     plt.annotate(txt, (x_values[i], y_values[i]), fontsize=12)

            # Labeling and Displaying the Plot
            plt.xlabel(f"Timestamps {x_values[0].strftime('%Y-%m-%d')}")
            plt.ylabel("Package Values")
            plt.title("First quality packages")
            plt.yticks(list(text_to_num.values()), list(text_to_num.keys()))  # Set y-axis ticks to text values
            plt.xticks(rotation=45)
            date_format = mdates.DateFormatter("%H:%M:%S")
            plt.gca().xaxis.set_major_formatter(date_format)

            plt.tight_layout()
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.show()

            # port to excel
            port_to_excel(time_data)
