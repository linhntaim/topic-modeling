import csv
import os
import re

from docs_manager.docs_db import DocsDB
from docs_manager.docs_extractor import DocsExtractor


class AbbreviationsExtractor(DocsExtractor):
    def from_db(self, docs_db: DocsDB, dir_path, raw_dir_path: str = None):
        docs = super().from_db(docs_db, dir_path, raw_dir_path)

        for doc in docs:
            cache_file_path = './.cache/abbrs/' + doc['file'] + '.csv'

            abbrs = {}  # id(name@lower) -> [id, name, expansion@n-gram, verified(v|x)]

            if os.path.isfile(cache_file_path):
                with open(cache_file_path, 'r', newline='', encoding='utf-8') as abbr_file:
                    abbr_csv_reader = csv.reader(abbr_file)
                    for row in abbr_csv_reader:
                        abbrs[row[0]] = row

            text = doc['text']
            abbr_matches = re.finditer(r'\(([A-Z]{2,}[a-z]*|[A-Z]+[a-z]+[A-Z]+)\)', text)
            for abbr_match in abbr_matches:
                abbr_name = abbr_match.group()[1:-1]
                abbr_id = abbr_name.lower()

                # Extract abbr expansion in n-gram form
                if re.fullmatch(r'[A-Z]+[a-z]+[A-Z]+', abbr_name) is not None:
                    abbr_main_name = abbr_name
                else:
                    abbr_main_name = re.sub(r'[a-z]+', '', abbr_name)
                abbr_expansion_tokens = [
                    re.sub(r'[^a-z\'\-_]+', '', token.lower())
                    for token in text[:abbr_match.start() - 1].split()[-len(abbr_main_name):]
                ]
                abbr_should_be_main_name = ''.join([token[:1] for token in abbr_expansion_tokens])
                abbr_expansion = '_'.join(abbr_expansion_tokens)
                matched_expansion = abbr_should_be_main_name.lower() == abbr_main_name.lower()

                if abbr_id not in abbrs:
                    if not matched_expansion:
                        print(
                            f'WARN: Abbreviation [{abbr_name}] may have unmatched expansion [{abbr_expansion}] in [{doc["file"]}]'
                        )
                    abbrs[abbr_id] = [abbr_id, abbr_name, abbr_expansion, 'v' if matched_expansion else 'x']
                else:
                    if abbrs[abbr_id][3] == 'x':  # not verified
                        if abbr_expansion == abbrs[abbr_id][2]:
                            if not matched_expansion:
                                print(
                                    f'WARN: Abbreviation [{abbr_name}] may have unmatched expansion [{abbr_expansion}] in [{doc["file"]}]'
                                )
                        else:
                            print(
                                f'WARN: Same abbreviation [{abbr_name}] represented by different expansions ([{abbr_expansion}] vs. [{abbrs[abbr_id][2]}]) in [{doc["file"]}]'
                            )

            with open(cache_file_path, 'w', newline='', encoding='utf-8') as abbr_file:
                abbr_csv_writer = csv.writer(abbr_file)
                abbr_csv_writer.writerows(abbrs.values())

        return docs
