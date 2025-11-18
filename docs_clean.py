import os

from dotenv import load_dotenv

from docs_manager.docs_db import DocsDB
from docs_manager.docs_extractor import DocsExtractor
from docs_manager.pdf_extractors.pymupdf_extractor import PyMuPDFExtractor

# Load environment variables from .env file
load_dotenv()

def remove_files(dir_path: str):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

clean_dir = '.\\.cache\\clean'

remove_files(clean_dir)

(DocsExtractor(PyMuPDFExtractor())
 .from_db(DocsDB(os.getenv('DOCS_FILE')), os.getenv('DOCS_DIR'), os.getenv('DOCS_RAW_DIR')))
