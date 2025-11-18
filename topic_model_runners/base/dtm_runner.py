import os
import shutil

from wordcloud import WordCloud

from topic_model_runners.base.dtm_option import DtmOption
from topic_model_runners.base.runner import Runner


class DtmRunner(Runner):
    def __init__(
            self,
            docs: list,
            option: DtmOption = None,
            cache_enabled: bool = True,
            naming_prefix: str = '',
    ):
        super().__init__(
            option if option is not None else DtmOption(),
            cache_enabled,
            naming_prefix,
        )

        self._docs: list = docs
        self._num_docs = len(self._docs)

        self._count_tokens_before: int = 0
        self._count_tokens_after: int = 0
        self._count_ngram_tokens: int = 0

    def generate_cache_key(self) -> str:
        return super().generate_cache_key() + '_nd' + str(self._num_docs)

    def _generate_cache_dir(self, cache_key: str) -> str:
        return '.\\.cache\\dtm\\' + cache_key

    def get_docs(self):
        return self._docs

    def get_num_docs(self):
        return self._num_docs

    def get_dtm(self):
        return self._result[0]

    def get_stats(self):
        return (self._count_tokens_before,
                self._count_tokens_after,
                self._count_ngram_tokens)

    def _make(self):
        print(f'Analyzing {self._num_docs} docs')

        if self._cached:
            self._pre_process_from_cache()
        else:
            self._pre_process()

        return self._build_dtm()

    def _pre_process_from_cache(self):
        step = 1
        step_name = 'token'

        self._output_tokens_from_cache(step_name, step=step)

        self._count_tokens_before = sum([len(doc['tokens']) for doc in self._docs])

        if self._option.filter:
            step += 1
            step_name = 'filtr'

        if self._option.stop_words:
            step += 1
            step_name = 'stopw'

        if self._option.lemma:
            step += 1
            step_name = 'lemma'

        if self._option.stem:
            step += 1
            step_name = 'stem'

        if self._option.ngram:
            step += 1
            step_name = 'ngram'

        if self._option.tfidf:
            step += 1
            step_name = 'tfidf'

        self._output_tokens_from_cache(step_name, step=step)
        self._output_wordcloud_from_cache(step_name, step=step)

        self._count_tokens_after = sum([len(doc['tokens']) for doc in self._docs])
        self._count_ngram_tokens = sum(
            [len([1 for token in doc['tokens'] if '_' in token]) for doc in self._docs])

    def _pre_process(self):
        step = 1
        step_name = 'token'
        self._tokenize()
        self._output_tokens(step_name, step=step)
        # self._output_wordcloud(step_name, step=step)

        self._count_tokens_before = sum([len(doc['tokens']) for doc in self._docs])

        if self._option.filter:
            step += 1
            step_name = 'filtr'
            self._filter()
            # self._output_tokens(step_name, step=step)
            # self._output_wordcloud(step_name, step=step)

        if self._option.stop_words:
            step += 1
            step_name = 'stopw'
            self._remove_stopwords()
            # self._output_tokens(step_name, step=step)
            # self._output_wordcloud(step_name, step=step)

        if self._option.lemma:
            step += 1
            step_name = 'lemma'
            self._lemmatize()
            # self._output_tokens(step_name, step=step)
            # self._output_wordcloud(step_name, step=step)

        if self._option.stem:
            step += 1
            step_name = 'stem'
            self._stem()
            # self._output_tokens(step_name, step=step)
            # self._output_wordcloud(step_name, step=step)

        if self._option.ngram:
            step += 1
            step_name = 'ngram'
            self._compute_ngrams()
            # self._output_tokens(step_name, step=step)
            # self._output_wordcloud(step_name, step=step)

        if self._option.tfidf:
            step += 1
            step_name = 'tfidf'
            self._reduce_with_tfidf()

        self._output_tokens(step_name, step=step)
        self._output_wordcloud(step_name, step=step)

        self._count_tokens_after = sum([len(doc['tokens']) for doc in self._docs])
        self._count_ngram_tokens = sum(
            [len([1 for token in doc['tokens'] if '_' in token]) for doc in self._docs])

    def _output_tokens(self, name: str, step: int):
        output_tks_dir = os.path.join(self._output_dir, 'p' + str(step) + '_' + name)
        os.makedirs(output_tks_dir, exist_ok=True)
        for doc in self._docs:
            tokens_file_path = os.path.join(output_tks_dir, doc['file'] + '.txt')
            with open(tokens_file_path, 'w', encoding='utf-8') as tokens_file:
                output = '\n'.join(doc['tokens'])
                tokens_file.write(output)

        if self._cache_enabled:
            cache_tks_dir = os.path.join(self._cache_dir, os.path.basename(output_tks_dir))
            shutil.copytree(output_tks_dir, cache_tks_dir)

    def _output_tokens_from_cache(self, name: str, step: int):
        output_tks_dir = os.path.join(self._output_dir, 'p' + str(step) + '_' + name)

        cache_tks_dir = os.path.join(self._cache_dir, os.path.basename(output_tks_dir))
        for doc in self._docs:
            tokens_file_path = os.path.join(cache_tks_dir, doc['file'] + '.txt')
            with open(tokens_file_path, 'r', encoding='utf-8') as tokens_file:
                doc['tokens'] = tokens_file.read().split('\n')

        shutil.copytree(cache_tks_dir, output_tks_dir)

    def _output_wordcloud(self, name: str, step: int):
        wordcloud_file_path = os.path.join(self._output_dir, 'wc' + str(step) + '_' + name + '.png')
        wordcloud = WordCloud(width=1920,
                              height=1080,
                              background_color='white',
                              max_words=1000,
                              contour_width=3,
                              contour_color='steelblue',
                              stopwords=set(),
                              collocations=False,
                              normalize_plurals=False,
                              include_numbers=True)
        wordcloud.generate_from_text(' '.join([' '.join(doc['tokens']) for doc in self._docs]))
        wordcloud.to_image().save(wordcloud_file_path)

        if self._cache_enabled:
            cache_file_path = os.path.join(self._cache_dir, os.path.basename(wordcloud_file_path))
            shutil.copyfile(wordcloud_file_path, cache_file_path)

    def _output_wordcloud_from_cache(self, name: str, step: int):
        wordcloud_file_path = os.path.join(self._output_dir, 'wc' + str(step) + '_' + name + '.png')
        cache_file_path = os.path.join(self._cache_dir, os.path.basename(wordcloud_file_path))
        shutil.copyfile(cache_file_path, wordcloud_file_path)

    def _tokenize(self):
        print('Tokenizing')

    def _filter(self):
        print('Filtering')

    def _remove_stopwords(self):
        print('Removing stopwords')

    def _lemmatize(self):
        print('Lemmatizing')

    def _stem(self):
        print('Stemming')

    def _compute_ngrams(self):
        print('Computing n-grams')

    def _reduce_with_tfidf(self):
        print('Reducing with TF-IDF')

    def _build_dtm(self):
        return None
