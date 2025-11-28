import os

from dotenv import load_dotenv

from docs_manager.docs_db import DocsDB
from docs_manager.retracted_detector import RetractedDetector

# Load environment variables from .env file
load_dotenv()

# Note: Extract retracted from raw pdfs
(RetractedDetector()
 .from_db(DocsDB(os.getenv('DOCS_FILE')), os.getenv('DOCS_DIR'), os.getenv('DOCS_RAW_DIR')))

