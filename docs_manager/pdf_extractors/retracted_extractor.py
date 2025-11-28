import re

import pymupdf.layout
import pymupdf4llm

from docs_manager.pdf_extractors.pdf_extractor import PdfExtractor


class RetractedExtractor(PdfExtractor):
    def from_file(self, file_path):
        with pymupdf.open(file_path) as pdf:
            if self._retracted(pdf, pdf.page_count, lambda page: self._search_simple(page)):
                return '<RETRACTED>'

            doc = pymupdf4llm.parse_document(pdf)
            if self._retracted(doc.pages, doc.page_count, lambda page: self._search(page)):
                return '<RETRACTED>'

            return ''

    def _retracted(self, doc, page_count, search_method):
        found = 0
        for page in doc:
            retracted_found = search_method(page)
            if retracted_found:
                found += 1

        return True if found >= page_count - 2 else False

    def _search_simple(self, page):
        l = page.search_for('retracted')
        return len(l) > 0

    def _search(self, page):
        for box in page.boxes:
            for text_box in box.textlines:
                for span_box in text_box['spans']:
                    text = span_box['text']
                    if re.search('retracted', text, re.IGNORECASE) is not None:
                        return True
        return False
