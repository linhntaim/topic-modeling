from topic_model_runners.base.lda_option import LdaOption


class MalletLdaOption(LdaOption):
    def __init__(self, option: dict = None):
        super().__init__(option)

        self._define('num_iterations', 1000, 'ni')
        self._define('optimize_interval', 0, 'oi')
        self._define('optimize_burn_in', 200, 'ob')