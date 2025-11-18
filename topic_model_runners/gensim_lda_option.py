from topic_model_runners.base.lda_option import LdaOption


class GensimLdaOption(LdaOption):
    def __init__(self, option: dict = None):
        super().__init__(option)

        self._define('chunksize', 200, False)
        self._define('passes', 10, False)
        self._define('update_every', 1, False)
        self._define('iterations', 50, False)
