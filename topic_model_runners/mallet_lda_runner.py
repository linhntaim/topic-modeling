import os

from topic_model_runners.base.dtm_option import DtmOption
from topic_model_runners.base.lda_runner import LdaRunner
from topic_model_runners.base.models.lda_mallet import LdaMallet
from topic_model_runners.mallet_lda_option import MalletLdaOption


class MalletLdaRunner(LdaRunner):
    def __init__(
            self,
            docs: list,
            dtm_option: DtmOption = None,
            topic_model_options: list[MalletLdaOption] = None,
            cache_enabled: bool = True,
    ):
        super().__init__(
            docs,
            dtm_option,
            topic_model_options,
            cache_enabled,
            'mallet_'
        )

        self._mallet_path = os.getenv('MALLET_PATH')
        if not os.path.isfile(self._mallet_path):
            raise FileNotFoundError(f"MALLET binary not found at: {self._mallet_path}")

    def _build_topic_model(self):
        id2word, corpus = self._dtm_runner.get_dtm()

        topic_model = LdaMallet(
            self._mallet_path,
            corpus=corpus,
            id2word=id2word,

            num_topics=self._option.num_topics,

            alpha=self._option.alpha,
            beta=self._option.beta,
            random_seed=self._option.random_seed,

            num_iterations=self._option.num_iterations,
            # @see https://mimno.github.io/Mallet/topics#hyperparameter-optimization
            optimize_interval=self._option.optimize_interval,
            optimize_burn_in=self._option.optimize_burn_in,
        )

        return topic_model

    def _topic_model_to_cache(self, topic_model):
        model_file_path = os.path.join(self._cache_dir, 'topic_model')
        topic_model.save(model_file_path)
        return

    def _topic_model_from_cache(self):
        model_file_path = os.path.join(self._cache_dir, 'topic_model')
        return LdaMallet.load(model_file_path)

    def _output_extra_result_option_names(self):
        return [
            'optimize_interval',
            'optimize_burn_in',
            'num_iterations',
        ]
