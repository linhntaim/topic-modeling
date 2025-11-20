from topic_model_runners.base.dtm_runner import DtmRunner
from topic_model_runners.base.lda_option import LdaOption
from topic_model_runners.base.topic_model_evaluation import TopicModelEvaluation


class LdaEvaluation(TopicModelEvaluation):
    def evaluate(self, dtm_runner: DtmRunner, topic_model, option: LdaOption):
        self.perplexity = self._calculate_perplexity(dtm_runner, topic_model, option)

        self.coherence, self.topic_matrix = self._calculate_coherence(dtm_runner, topic_model, option)

        self.topic_term_matrix = topic_model.get_topic_term_matrix()
        self.document_topic_matrix = topic_model.get_document_topic_matrix(dtm_runner)

        return self

    def _calculate_perplexity(self, dtm_runner, topic_model, option):
        return topic_model.get_perplexity(
            dtm_runner,
            num_terms=option.terms_per_topic,
        )

    def _calculate_coherence(self, dtm_runner: DtmRunner, topic_model, option: LdaOption):
        return topic_model.get_coherence(
            dtm_runner,
            num_terms=option.terms_per_topic,
        )
