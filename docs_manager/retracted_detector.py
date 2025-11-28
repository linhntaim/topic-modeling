from docs_manager.docs_db import DocsDB
from docs_manager.docs_extractor import DocsExtractor
from docs_manager.pdf_extractors.retracted_extractor import RetractedExtractor


class RetractedDetector(DocsExtractor):
    def __init__(self):
        super().__init__(RetractedExtractor())

    def from_db(self, docs_db: DocsDB, dir_path: str, raw_dir_path: str = None):
        docs = super().from_db(docs_db, dir_path, raw_dir_path)
        for doc in docs:
            if doc['text'] == '<RETRACTED>':
                print(doc['title'])

        return docs

    def _get_text(self, file_path):
        return self._pdf_extractor.from_file(file_path)
