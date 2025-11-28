import argparse
import os

from dotenv import load_dotenv

from docs_manager.docs_db import DocsDB
from docs_manager.docs_extractor import DocsExtractor
from docs_manager.pdf_extractors.pymupdf_extractor import PyMuPDFExtractor

# Load environment variables from .env file
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--reset', action='store_true', help='Remove all raw & cleaned files')
args = parser.parse_args()

reset = args.reset


def remove_files(dir_path: str):
    for filename in os.listdir(dir_path):
        if filename == '.gitignore':
            continue
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


clean_dir = '.\\.cache\\clean'
raw_dir = '.\\.cache\\raw'

if reset:
    print('Removing all raw & cleaned files')
    remove_files(raw_dir)
    remove_files(clean_dir)

(DocsExtractor(PyMuPDFExtractor())
 .from_db(DocsDB(os.getenv('DOCS_FILE')), os.getenv('DOCS_DIR'), os.getenv('DOCS_RAW_DIR')))
