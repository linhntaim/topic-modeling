from topic_model_runners.base.dtm_runner import DtmRunner
from topic_model_runners.base.topic_model_option import TopicModelOption


class TopicModelEvaluation:
    def __init__(self):
        return

    def evaluate(self, dtm_runner: DtmRunner, topic_model, option: TopicModelOption):
        return self

    def save(
            self,
            model_file_path,
            topics_file_path,
            topic_terms_file_path,
            document_topics_file_path
    ):
        return self

    def load(
            self,
            model_file_path,
            topics_file_path,
            topic_terms_file_path,
            document_topics_file_path
    ):
        return None
