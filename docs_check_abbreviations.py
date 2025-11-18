import os

from dotenv import load_dotenv

from docs_manager.docs_db import DocsDB
from docs_manager.abbreviations_extractor import AbbreviationsExtractor
from docs_manager.pdf_extractors.pymupdf_extractor import PyMuPDFExtractor

# Load environment variables from .env file
load_dotenv()

# Note: Extract abbreviations from clean texts
(AbbreviationsExtractor(PyMuPDFExtractor())
 .from_db(DocsDB(os.getenv('DOCS_FILE')), os.getenv('DOCS_DIR'), os.getenv('DOCS_RAW_DIR')))
