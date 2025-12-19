from topic_model_runners.base.dtm_runner import DtmRunner
from topic_model_runners.base.topic_model_evaluation import TopicModelEvaluation
from topic_model_runners.base.topic_model_option import TopicModelOption


class BERTopicEvaluation(TopicModelEvaluation):
    def __init__(self):
        super().__init__()

        self.topic_matrix = []
        self.topic_term_matrix = []
        self.document_topic_matrix = []

    def evaluate(self, dtm_runner: DtmRunner, topic_model, option: TopicModelOption):
        docs = [' '.join(doc['tokens']) for doc in dtm_runner.get_docs()]
        topics, probabilities = topic_model.fit_transform(docs)
        # print(topics)
        # print(probabilities)
        topic_info = topic_model.get_topic_info()
        print(topic_info)
        # topic = topic_model.get_topic(0)
        # print(topic)
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