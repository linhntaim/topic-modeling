import csv
import re
from statistics import median

import nltk
import spacy
from gensim.corpora import Dictionary
from gensim.models import Phrases, TfidfModel
from gensim.models.phrases import Phraser
from gensim.utils import tokenize
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

from helpers.english import English as EnglishHelper
from topic_model_runners.base.dtm_option import DtmOption
from topic_model_runners.base.dtm_runner import DtmRunner


class DefaultDtmRunner(DtmRunner):
    def __init__(self, docs: list, option: DtmOption = None, cache_enabled: bool = True):
        super().__init__(docs, option, cache_enabled)

        nltk.download('stopwords')
        nltk.download('wordnet')

    # @see https://github.com/kapadias/medium-articles/blob/master/natural-language-processing/topic-modeling/Introduction%20to%20Topic%20Modeling.ipynb#Step-4:-Prepare-text-for-LDA-analysis-
    def _tokenize(self):
        super()._tokenize()

        en_helper = EnglishHelper()
        en_helper.prepare_us_to_gb()

        for doc in self._docs:
            replacing_abbrs = {}
            deleting_abbrs = []

            abbr_file_path = '.\\.cache\\abbrs\\' + doc['file'] + '.csv'
            with open(abbr_file_path, 'r', newline='', encoding='utf-8') as abbr_file:
                abbr_csv_reader = csv.reader(abbr_file)
                for row in abbr_csv_reader:
                    if row[3] == 'v':
                        replacing_abbrs[row[1]] = row[2]
                    elif row[3] == 'd':
                        deleting_abbrs.append(row[1])

            # Prepare
            doc['tokens'] = []
            text = doc['text']

            # Handle abbrs
            # - remove unnecessary
            for abbr_name in deleting_abbrs:
                text = re.sub(r'\b' + abbr_name + '\\b', '', text)
            # - restore
            if self._option.abbr_as_ngram:
                for abbr_name, abbr_expansion in replacing_abbrs.items():
                    text = re.sub(r'\b' + abbr_name + '\\b', abbr_expansion, text)
            else:
                for abbr_name, abbr_expansion in replacing_abbrs.items():
                    spaced_abbr_expansion = ' '.join(abbr_expansion.split('_'))
                    text = re.sub(r'\b' + abbr_name + '\\b', spaced_abbr_expansion, text)

            # Tokenize
            tokens = tokenize(text, lowercase=False, deacc=True, errors='ignore')

            # Transform tokens:
            # - fix '_' glitch: token starts or ends with '_'
            # - convert en-US to en-GB
            # - lowercase
            for token in tokens:
                token = token.strip('_')
                token = en_helper.convert_us_to_gb(token)
                doc['tokens'].append(token.lower())

    # @see https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#pre-process-and-vectorize-the-documents
    def _filter(self):
        super()._filter()

        for doc in self._docs:
            # Filter words that:
            # + have at least 3 chars
            # + meaningful word: has vowels ('a', 'e', 'i', 'o', 'y')
            # + not numeric (always OK)
            doc['tokens'] = [token for token in doc['tokens'] if
                             len(token) >= 3
                             and re.search(r'[aeiouy]+', token) is not None
                             # and not token.isnumeric()
                             ]

    # @see # @see https://github.com/kapadias/medium-articles/blob/master/natural-language-processing/topic-modeling/Introduction%20to%20Topic%20Modeling.ipynb#Step-4:-Prepare-text-for-LDA-analysis-
    def _remove_stopwords(self):
        super()._remove_stopwords()

        stop_words = stopwords.words('english')
        stop_words.extend([
            'et', 'al', 'also',
            'email', 'emails', 'email_address', 'email_addresses', 'homepage', 'homepages',
            'allrightsreserved', 'copyright',
        ])
        for doc in self._docs:
            doc['tokens'] = [token for token in doc['tokens'] if token not in stop_words]

    def _lemmatize(self):
        super()._lemmatize()

        # return self._lemmatize1(docs)
        return self._lemmatize2()

    # @see https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#pre-process-and-vectorize-the-documents
    def _lemmatize1(self):
        lemmatizer = WordNetLemmatizer()
        for doc in self._docs:
            lemmatized_tokens = []
            for token in doc['tokens']:
                if '_' in token.text:
                    lemmatized_tokens.append(token)
                else:
                    lemmatized_tokens.append(lemmatizer.lemmatize(token))

            doc['tokens'] = lemmatized_tokens

    # @see https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#10removestopwordsmakebigramsandlemmatize
    def _lemmatize2(self):
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']

        for doc in self._docs:
            lemmatized_tokens = []
            for token in nlp(' '.join(doc['tokens'])):
                if '_' in token.text:
                    lemmatized_tokens.append(token.text)
                elif token.pos_ in allowed_postags:
                    lemmatized_tokens.append(token.lemma_)

            doc['tokens'] = lemmatized_tokens

    def _stem(self):
        super()._stem()

        porter_stemmer = PorterStemmer()
        for doc in self._docs:
            doc['tokens'] = [porter_stemmer.stem(token) for token in doc['tokens']]

    def _compute_ngrams(self):
        super()._compute_ngrams()

        # return self._compute_ngrams_1(docs)
        return self._compute_ngrams_2()

    # @see https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#pre-process-and-vectorize-the-documents
    def _compute_ngrams_1(self):
        tokenized_docs = [doc['tokens'] for doc in self._docs]

        bigram = Phrases(tokenized_docs, min_count=20)
        for i in range(len(self._docs)):
            for token in bigram[tokenized_docs[i]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    tokenized_docs[i].append(token)

            self._docs[i]['tokens'] = tokenized_docs[i]

    # @see https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#9createbigramandtrigrammodels
    # @see https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/#4.-Build-the-Bigram,-Trigram-Models-and-Lemmatize
    def _compute_ngrams_2(self):
        tokenized_docs = [doc['tokens'] for doc in self._docs]

        # Build the bigram and trigram models
        bigram = Phrases(tokenized_docs, min_count=5, threshold=100)  # higher threshold fewer phrases.
        trigram = Phrases(bigram[tokenized_docs], threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = Phraser(bigram)
        trigram_mod = Phraser(trigram)

        for doc in self._docs:
            # bigram
            doc['tokens'] = bigram_mod[doc['tokens']]
            # trigram
            doc['tokens'] = trigram_mod[bigram_mod[doc['tokens']]]

    def _reduce_with_tfidf(self):
        super()._reduce_with_tfidf()

        tokenized_docs = [doc['tokens'] for doc in self._docs]

        id2word = Dictionary(tokenized_docs)
        corpus = [id2word.doc2bow(tokens) for tokens in tokenized_docs]

        tfidf = TfidfModel(corpus)
        tfidf_corpus = tfidf[corpus]
        for i, tfidf_doc in enumerate(tfidf_corpus):
            median_freq = median([token_freq for token_id, token_freq in tfidf_doc])
            strong_tokens = [id2word[token_id] for token_id, token_freq in tfidf_doc if token_freq > median_freq]
            self._docs[i]['tokens'] = [token for token in tokenized_docs[i] if token in strong_tokens]

    def _build_dtm(self):
        tokenized_docs = [doc['tokens'] for doc in self._docs]

        id2word = Dictionary(tokenized_docs)
        corpus = [id2word.doc2bow(tokens) for tokens in tokenized_docs]

        return id2word, corpus