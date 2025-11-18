import os

from dotenv import load_dotenv

from docs_manager.docs_db import DocsDB
from docs_manager.pdf_extractors.pymupdf_extractor import PyMuPDFExtractor
from docs_manager.special_chars_extractor import SpecialCharsExtractor

# Load environment variables from .env file
load_dotenv()

# Note: Extract special chars from raw texts
(SpecialCharsExtractor(PyMuPDFExtractor())
 .from_db(DocsDB(os.getenv('DOCS_FILE')), os.getenv('DOCS_DIR'), os.getenv('DOCS_RAW_DIR')))
