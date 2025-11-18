import os

from dotenv import load_dotenv

from docs_manager.docs_db import DocsDB

# Load environment variables from .env file
load_dotenv()

DocsDB(os.getenv('DOCS_FILE')).update_files(os.getenv('DOCS_DIR'))
