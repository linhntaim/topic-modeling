import pyLDAvis.gensim
from gensim.models import LdaModel

from helpers import number
from topic_model_runners.base.dtm_runner import DtmRunner
from topic_model_runners.base.models.lda_base import LdaBase


class LdaGensim(LdaBase, LdaModel):
    def get_topic_term_matrix(self):
        return [
            [number.probability(term_probability) for term_probability in terms] for terms in self.get_topics()
        ]

    def get_document_topic_matrix(self, dtm_runner: DtmRunner):
        _, corpus = dtm_runner.get_dtm()

        document_topic_matrix = []
        for bow in corpus:
            gamma, phis = self.inference([bow], collect_sstats=False)
            topic_dist = gamma[0] / sum(gamma[0])  # normalize distribution

            document_topic_matrix.append(
                [number.probability(topic_probability) for topic_probability in topic_dist]
            )

        return document_topic_matrix

    def get_perplexity(self, dtm_runner: DtmRunner, num_terms: int):
        id2word, corpus = dtm_runner.get_dtm()
        return self.log_perplexity(corpus)

    def _prepare_visualization(self, corpus, dictionary, **kwargs):
        return pyLDAvis.gensim.prepare(self, corpus, dictionary, mds='mmds')
