import os
import time

from topic_model_runners.base.dtm_runner import DtmRunner
from topic_model_runners.base.runner import Runner
from topic_model_runners.base.topic_model_evaluation import TopicModelEvaluation
from topic_model_runners.base.topic_model_option import TopicModelOption


class TopicModelRunner(Runner):
    def __init__(
            self,
            dtm_runner: DtmRunner,
            options: list[TopicModelOption] = None,
            cache_enabled: bool = True,
            naming_prefix: str = '',
    ):
        super().__init__(
            options if options is not None else [],
            cache_enabled,
            naming_prefix,
        )

        self._dtm_runner: DtmRunner = (dtm_runner
                                       .set_output_dir(self._output_dir)
                                       .enable_cache(self._cache_enabled))

    def generate_cache_key(self) -> str:
        return super().generate_cache_key() + '_' + self._dtm_runner.generate_cache_key()

    def _generate_cache_dir(self, cache_key: str) -> str:
        return '.\\.cache\\model\\' + cache_key

    def _before_execution(self):
        self._dtm_runner.run()

    def _make(self):
        if self._cached:
            return self._topic_model_from_cache()

        topic_model = self._build_topic_model()
        if self._cache_enabled:
            self._topic_model_to_cache(topic_model)

        return topic_model

    def _evaluate(self, topic_model):
        if self._cached:
            return self._evaluation_from_cache(topic_model)

        evaluation = self._build_evaluation(topic_model)
        if self._cache_enabled:
            self._evaluation_to_cache(evaluation)

        return evaluation

    def _build_topic_model(self):
        return None

    def _topic_model_to_cache(self, topic_model):
        return

    def _topic_model_from_cache(self):
        return self._build_topic_model()

    def _create_evaluation(self) -> TopicModelEvaluation:
        return TopicModelEvaluation()

    def _build_evaluation(self, topic_model):
        return self._create_evaluation().evaluate(self._dtm_runner, topic_model, self._option)

    def _evaluation_to_cache(self, evaluation: TopicModelEvaluation):
        evaluation.save(
            model_file_path=os.path.join(self._cache_dir, f'eval_nte{str(self._option.terms_per_topic)}.model'),
            topics_file_path=os.path.join(self._cache_dir, f'eval_nte{str(self._option.terms_per_topic)}.topics'),
            topic_terms_file_path=os.path.join(self._cache_dir,
                                               f'eval_nte{str(self._option.terms_per_topic)}.topic_terms'),
            document_topics_file_path=os.path.join(self._cache_dir,
                                                   f'eval_nte{str(self._option.terms_per_topic)}.document_topics'),
        )

    def _evaluation_from_cache(self, topic_model):
        evaluation = self._create_evaluation().load(
            model_file_path=os.path.join(self._cache_dir, f'eval_nte{str(self._option.terms_per_topic)}.model'),
            topics_file_path=os.path.join(self._cache_dir, f'eval_nte{str(self._option.terms_per_topic)}.topics'),
            topic_terms_file_path=os.path.join(self._cache_dir,
                                               f'eval_nte{str(self._option.terms_per_topic)}.topic_terms'),
            document_topics_file_path=os.path.join(self._cache_dir,
                                                   f'eval_nte{str(self._option.terms_per_topic)}.document_topics'),
        )

        if evaluation is None:
            evaluation = self._build_evaluation(topic_model)
            self._evaluation_to_cache(evaluation)

        return evaluation

    def _after_execution(self):
        self._output_results()
        self._output_umap_results()

        num_results = len(self._result)
        for i, result in enumerate(self._result):
            print('Result %d / %d' % (i + 1, num_results))
            start_at = time.time()

            _, _, option = result

            output_dir = os.path.join(self._output_dir, 'o_' + str(option))
            os.makedirs(output_dir, exist_ok=True)

            self._output_result(result, output_dir)

            duration = time.time() - start_at
            print(f'Result time: {duration:.0f} sec')

    def _output_results(self):
        return

    def _output_umap_results(self):
        return

    def _output_result(self, result, output_dir):
        self._output_topics(result, output_dir)
        self._output_topic_terms(result, output_dir)
        self._output_document_topics(result, output_dir)
        self._output_visualization(result, output_dir)
        self._output_umap_result(result, output_dir)
        return

    def _output_topics(self, result, output_dir):
        return

    def _output_topic_terms(self, result, output_dir):
        return

    def _output_document_topics(self, result, output_dir):
        return

    def _output_visualization(self, result, output_dir):
        return

    def _output_umap_result(self, result, output_dir):
        return
