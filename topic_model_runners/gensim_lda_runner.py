import os.path

from topic_model_runners.base.dtm_option import DtmOption
from topic_model_runners.base.lda_runner import LdaRunner
from topic_model_runners.base.models.lda_gensim import LdaGensim
from topic_model_runners.gensim_lda_option import GensimLdaOption


# @see https://phamdinhkhanh.github.io/2019/09/08/LDATopicModel.html
# @see https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#12buildingthetopicmodel
# @see https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
# @see https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py
# @see https://github.com/kapadias/medium-articles/blob/master/natural-language-processing/topic-modeling/Introduction%20to%20Topic%20Modeling.ipynb
class GensimLdaRunner(LdaRunner):
    def __init__(
            self,
            docs: list,
            dtm_option: DtmOption = None,
            topic_model_options: list[GensimLdaOption] = None,
            cache_enabled: bool = True,
    ):
        super().__init__(
            docs,
            dtm_option,
            topic_model_options,
            cache_enabled,
            'gensim_'
        )

    def _build_topic_model(self):
        id2word, corpus = self._dtm_runner.get_dtm()

        # Train LDA model
        topic_model = LdaGensim(
            corpus=corpus,
            id2word=id2word,

            num_topics=self._option.num_topics,

            alpha=self._option.alpha,
            eta=self._option.beta,
            random_state=self._option.random_seed,

            chunksize=self._option.chunksize,
            passes=self._option.passes,
            update_every=self._option.update_every,
            iterations=self._option.iterations,

            eval_every=None,  # Don't evaluate model perplexity, takes too much time.
            per_word_topics=True,
        )

        return topic_model

    def _topic_model_to_cache(self, topic_model):
        model_file_path = os.path.join(self._cache_dir, 'topic_model')
        topic_model.save(model_file_path)
        return

    def _topic_model_from_cache(self):
        model_file_path = os.path.join(self._cache_dir, 'topic_model')
        return LdaGensim.load(model_file_path)

    def _output_extra_result_option_names(self):
        return [
            'chunksize',
            'passes',
            'update_every',
            'iterations',
        ]
