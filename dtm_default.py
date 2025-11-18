import os
import warnings

from dotenv import load_dotenv

from docs_manager.docs_db import DocsDB
from docs_manager.docs_extractor import DocsExtractor
from docs_manager.pdf_extractors.pymupdf_extractor import PyMuPDFExtractor
from topic_model_runners.base.default_dtm_runner import DefaultDtmRunner
from topic_model_runners.base.dtm_option import DtmOption


def main():
    warnings.filterwarnings('ignore')

    # Load environment variables from .env file
    load_dotenv()

    dtm_option = DtmOption({
        'ngram': True,
        'tfidf': True,
        'abbr_as_ngram': False,
    })

    DefaultDtmRunner(
        DocsExtractor(PyMuPDFExtractor()).from_db(
            DocsDB(os.getenv('DOCS_FILE')),
            os.getenv('DOCS_DIR'),
            os.getenv('DOCS_RAW_DIR')
        ),
        dtm_option
    ).run()


if __name__ == '__main__':
    main()
