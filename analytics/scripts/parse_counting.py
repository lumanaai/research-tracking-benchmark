import os
import json
import csv
from collections import Counter

def count_alert_values_in_folder(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            counter = Counter()
            for entry in data:
                value = entry.get('alertData', {}).get('value')
                if value is not None:
                    counter[value] += 1
            print(f"{filename}:")
            for value, count in counter.items():
                results.append({'filename': filename, 'value': value, 'count': count})
                print(f"  value {value}: {count}")

    # Write results to CSV
    csv_path = os.path.join(folder_path, 'alert_value_counts.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'value', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nResults written to {csv_path}")

if __name__ == "__main__":
    folder = "/mnt/d/west_wall/out"  # Change this to your folder path
    count_alert_values_in_folder(folder)