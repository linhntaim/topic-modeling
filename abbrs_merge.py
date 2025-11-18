import csv
import glob
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

rows = []
for csv_file_path in glob.glob('.\\.cache\\abbrs\\*.csv'):
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            row.append(os.path.basename(csv_file_path))
            rows.append(row)

with open('.\\abbrs.csv', 'w', newline='', encoding='utf-8') as abbrs_file:
    writer = csv.writer(abbrs_file)
    writer.writerows(rows)