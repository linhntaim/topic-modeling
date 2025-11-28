import glob
import os
import re

import pandas


class DocsDB:
    def __init__(self, db_file_path, sheet_name='Sheet1'):
        if not os.path.isfile(db_file_path):
            raise FileNotFoundError(f'File {db_file_path} does not exist')

        self.__db_file_path = db_file_path
        self.__sheet_name = sheet_name
        self.__db = pandas.read_excel(self.__db_file_path,
                                      sheet_name=self.__sheet_name,
                                      converters={'eid': str, 'title': str, 'file': str, 'PROCESSING': str},
                                      na_filter=False,
                                      engine='openpyxl')

    def __flush(self):
        self.__db.to_excel(self.__db_file_path,
                           sheet_name=self.__sheet_name,
                           index=False)

    def get_all(self):
        return self.__db.to_dict(orient='records')

    def get_processed(self):
        return self.__db[self.__db['PROCESSING'] == '1'].to_dict(orient='records')

    def get_non_processed(self):
        return self.__db[self.__db['PROCESSING'] == '0'].to_dict(orient='records')

    def get_non_deleted(self):
        return self.__db[self.__db['PROCESSING'] != 'delete'].to_dict(orient='records')

    def get_deleted(self):
        return self.__db[self.__db['PROCESSING'] == 'delete'].to_dict(orient='records')

    def update_files(self, docs_dir_path):
        def to_slug(text):
            return re.sub(r'\s+', '-', re.sub(r'[^a-z]', ' ', text.lower()).strip())

        if not os.path.isdir(docs_dir_path):
            raise FileNotFoundError(f'Directory {docs_dir_path} does not exist')

        file_paths = glob.glob(os.path.join(docs_dir_path, "*.pdf"))
        print('Storage has %d files' % len(file_paths))

        file_slugs = list(map(lambda file_path: to_slug(os.path.splitext(os.path.basename(file_path))[0]), file_paths))

        updated = 0
        for index, row in self.__db.iterrows():
            eid = row['eid']
            title = row['title']
            file = row['file']

            slug = to_slug(title)
            matched_indices = [i for i, file_slug in enumerate(file_slugs) if file_slug == slug]
            if len(matched_indices) > 0:
                updating_file_name = os.path.basename(file_paths[matched_indices[0]])
                if updating_file_name != file:
                    self.__db.at[index, 'file'] = updating_file_name
                    updated += 1
                    print(f'Eid "{eid}": Updating file {updating_file_name}')
            else:
                if file == '':
                    print(f'WARN: The file mapping to the eid "{eid}" does not exist')

        if updated > 0:
            self.__flush()
        print('Updated %d files' % updated)
