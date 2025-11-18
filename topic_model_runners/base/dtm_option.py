from topic_model_runners.base.runner_option import RunnerOption


class DtmOption(RunnerOption):
    def __init__(self, option: dict = None):
        super().__init__(option)

        self._define('filter', True, 'flt')
        self._define('stop_words', True, 'stw')
        self._define('lemma', True, 'lem')
        self._define('stem', False, False)
        self._define('ngram', True, 'ngr')
        self._define('tfidf', True, 'tf')
        self._define('abbr_as_ngram', True, 'aan')
