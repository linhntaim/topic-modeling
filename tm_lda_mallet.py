import os
import warnings

from dotenv import load_dotenv

from docs_manager.docs_db import DocsDB
from docs_manager.docs_extractor import DocsExtractor
from docs_manager.pdf_extractors.pymupdf_extractor import PyMuPDFExtractor
from topic_model_runners.base.dtm_option import DtmOption
from topic_model_runners.mallet_lda_option import MalletLdaOption
from topic_model_runners.mallet_lda_runner import MalletLdaRunner


def main():
    warnings.filterwarnings('ignore')

    # Load environment variables from .env file
    load_dotenv()

    dtm_option = DtmOption({
        'ngram': True,
        'tfidf': True,
        'abbr_as_ngram': False,
    })

    topic_model_options = []
    # CAUTION: Remember to split loop to reduce running time

    # nums_topics = range(2, 16, 2)
    # nums_topics = range(16, 30, 2)
    # nums_topics = range(30, 42, 2)
    # nums_topics = range(42, 52, 2)
    # nums_topics = range(52, 62, 2)
    # nums_topics = range(62, 72, 2)
    # nums_topics = range(72, 82, 2)
    # nums_topics = range(82, 92, 2)
    # nums_topics = range(92, 102, 2)

    # nums_topics = range(4, 44, 4)
    # nums_topics = range(44, 76, 4)
    # nums_topics = range(76, 104, 4)

    # nums_topics = range(2, 50, 4)
    # nums_topics = range(50, 82, 4)
    # nums_topics = range(82, 102, 4)

    nums_topics = [10]

    alphas = [5]

    betas = [0.01]

    print('Topics:', end=' ')
    for num_topics in nums_topics:
        print(num_topics, end=' ')
        for alpha in alphas:
            for beta in betas:
                topic_model_options.append(
                    MalletLdaOption({
                        'num_topics': num_topics,
                        'terms_per_topic': 15,

                        'alpha': alpha,
                        'beta': beta,
                        'random_seed': 100,

                        'num_iterations': 1000,
                        'optimize_interval': 0,
                        'optimize_burn_in': 200,
                    })
                )
    print('')

    (MalletLdaRunner(
        DocsExtractor(PyMuPDFExtractor()).from_db(
            DocsDB(os.getenv('DOCS_FILE')),
            os.getenv('DOCS_DIR'),
            os.getenv('DOCS_RAW_DIR')
        ),
        dtm_option,
        topic_model_options,
        cache_enabled=True
    )
     .run())


if __name__ == '__main__':
    main()
