from topic_model_runners.base.topic_model_option import TopicModelOption


class LdaOption(TopicModelOption):
    def __init__(self, option: dict = None):
        super().__init__(option)

        self._define('alpha', 'autp', 'al')
        self._define('beta', 'auto', 'be')
        self._define('random_seed', 100, False)
