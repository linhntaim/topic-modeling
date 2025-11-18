from pdfminer.high_level import extract_text as extract_text_from_pdf

from docs_manager.pdf_extractors.pdf_extractor import PdfExtractor


class PdfMinerExtractor(PdfExtractor):
    def from_file(self, file_path) -> str:
        return extract_text_from_pdf(file_path)