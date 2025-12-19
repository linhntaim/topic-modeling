from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

from topic_model_runners.base.default_dtm_runner import DefaultDtmRunner

from topic_model_runners.base.dtm_option import DtmOption
from topic_model_runners.base.topic_model_evaluation import TopicModelEvaluation
from topic_model_runners.base.topic_model_option import TopicModelOption
from topic_model_runners.base.topic_model_runner import TopicModelRunner
from topic_model_runners.bertopic_evaluation import BERTopicEvaluation


class BERTopicRunner(TopicModelRunner):
    def __init__(
            self,
            docs: list,
            dtm_option: DtmOption = None,
            topic_model_options: list[TopicModelOption] = None,
            cache_enabled: bool = True,
    ):
        super().__init__(
            DefaultDtmRunner(docs, dtm_option),
            topic_model_options,
            cache_enabled,
            'bertopic_'
        )

    def _build_topic_model(self):
        representation_model = KeyBERTInspired()
        topic_model = BERTopic(
            representation_model=representation_model,
            min_topic_size=self._option.num_topics,
            top_n_words=self._option.terms_per_topic,
        )

        return topic_model

    def _create_evaluation(self) -> TopicModelEvaluation:
        return BERTopicEvaluation()