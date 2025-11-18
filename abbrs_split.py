import csv
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

with open('.\\abbrs.csv', 'r', newline='', encoding='utf-8') as abbrs_file:
    csv_reader = csv.reader(abbrs_file)
    prev_abbr_file_name = ''
    rows = []

    for row in csv_reader:
        abbr_file_name = row.pop()

        if prev_abbr_file_name != '' and abbr_file_name != prev_abbr_file_name:
            prev_abbr_file_path = '.\\.cache\\abbrs\\' + prev_abbr_file_name
            if not os.path.exists(prev_abbr_file_path):
                print(prev_abbr_file_path)
            with open(prev_abbr_file_path, 'w', newline='', encoding='utf-8') as abbr_file:
                csv_writer = csv.writer(abbr_file)
                csv_writer.writerows(rows)
            rows.clear()

        rows.append(row)

        prev_abbr_file_name = abbr_file_name

