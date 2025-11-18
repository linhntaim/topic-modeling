import os

import funcy
import pyLDAvis
from gensim.models import CoherenceModel

from helpers.number import probability
from topic_model_runners.base.dtm_runner import DtmRunner


class LdaBase:
    def get_topic_term_matrix(self):
        return []

    def get_document_topic_matrix(self, dtm_runner: DtmRunner):
        return []

    def get_perplexity(self, dtm_runner: DtmRunner, num_terms: int):
        return 0.0

    def get_coherence(self, dtm_runner: DtmRunner, num_terms: int):
        coherence_model = CoherenceModel(
            model=self,
            texts=[doc['tokens'] for doc in dtm_runner.get_docs()],
            coherence='c_v',
            topn=num_terms
        )

        coherence_per_topic = coherence_model.get_coherence_per_topic()

        return (coherence_model.aggregate_measures(coherence_per_topic),
                [probability(coherence) for coherence in coherence_per_topic])

    def _extract_visualizing_data(self, corpus, dictionary):
        return {
            'topic_term_dists': [],
            'doc_topic_dists': [],
            'doc_lengths': [],
            'vocab': [],
            'term_frequency': [],
        }

    def _prepare_visualization(self, corpus, dictionary, **kwargs):
        opts = funcy.merge(self._extract_visualizing_data(corpus, dictionary), kwargs)
        return pyLDAvis.prepare(**opts)

    def create_visualization(self, corpus, dictionary, output_dir, output_name='visualized_topics.html', **kwargs):
        pyLDAvis.save_html(
            self._prepare_visualization(corpus, dictionary, **kwargs),
            os.path.join(output_dir, output_name)
        )
