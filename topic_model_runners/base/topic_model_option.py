from topic_model_runners.base.runner_option import RunnerOption


class TopicModelOption(RunnerOption):
    def __init__(self, option: dict = None):
        super().__init__(option)

        self._define('num_topics', 3, 'nt')

        self._define('terms_per_topic', 10, False)
