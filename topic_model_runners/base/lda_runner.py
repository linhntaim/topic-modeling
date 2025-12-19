import csv
import json
import os

from gensim import matutils

from topic_model_runners.base.default_dtm_runner import DefaultDtmRunner
from topic_model_runners.base.dtm_option import DtmOption
from topic_model_runners.base.lda_evaluation import LdaEvaluation
from topic_model_runners.base.lda_option import LdaOption
from topic_model_runners.base.topic_model_evaluation import TopicModelEvaluation
from topic_model_runners.base.topic_model_runner import TopicModelRunner


class LdaRunner(TopicModelRunner):
    def __init__(
            self,
            docs: list,
            dtm_option: DtmOption = None,
            topic_model_options: list[LdaOption] = None,
            cache_enabled: bool = True,
            naming_prefix: str = '',
    ):
        super().__init__(
            DefaultDtmRunner(docs, dtm_option),
            topic_model_options,
            cache_enabled,
            naming_prefix,
        )

    def _create_evaluation(self) -> TopicModelEvaluation:
        return LdaEvaluation()

    def _output_extra_result_option_names(self):
        return []

    def _output_results(self):
        num_docs = self._dtm_runner.get_num_docs()
        dtm_option = self._dtm_runner.get_option()

        step_ngram = str(dtm_option.ngram)
        step_tfidf = str(dtm_option.tfidf)
        flag_abbr_as_ngram = str(dtm_option.abbr_as_ngram)

        count_tokens_before, count_tokens_after, count_ngram_tokens = self._dtm_runner.get_stats()

        with open(os.path.join(self._output_dir, 'results.csv'), 'w', newline='', encoding='utf-8') as results_file:
            results_writer = csv.writer(results_file)
            extra_option_names = self._output_extra_result_option_names()
            headers = [
                'num_topics',
                'num_terms',
                'perplexity',
                'coherence',
                'num_docs',
                'num_tokens_before',
                'num_tokens_after',
                'num_n-grams',
                'step_ngram',
                'step_tfidf',
                'flag_abbr_as_ngram',
                'alpha',
                'beta',
                'random_seed',
            ]
            headers.extend(extra_option_names)
            results_writer.writerow(headers)

            for result in self._result:
                topic_model, evaluation, option = result
                extra_option_values = [getattr(option, name) for name in extra_option_names]

                values = [
                    option.num_topics,
                    option.terms_per_topic,
                    evaluation.perplexity,
                    evaluation.coherence,
                    num_docs,
                    count_tokens_before,
                    count_tokens_after,
                    count_ngram_tokens,
                    step_ngram,
                    step_tfidf,
                    flag_abbr_as_ngram,
                    option.alpha,
                    option.beta,
                    option.random_seed,
                ]
                values.extend(extra_option_values)
                results_writer.writerow(values)

        print(f'Results stored in {self._output_dir}')

    def _output_umap_results(self):
        docs = self._dtm_runner.get_docs()
        id2word, corpus = self._dtm_runner.get_dtm()

        metadata = {
            'docs': [doc['title'] for doc in docs],
            'terms': [id2word[i] for i in range(len(id2word))],
            'doc_term_matrix': [{token_id: token_count for token_id, token_count in bow} for bow in corpus],
        }
        with open(os.path.join(self._output_dir, 'umap_metadata.json'), 'w', encoding='utf-8') as file:
            json.dump(metadata, file, indent=4)

    def _output_result(self, result, output_dir):
        id2word, _ = self._dtm_runner.get_dtm()
        _, evaluation, option = result

        topic_term_matrix = evaluation.topic_term_matrix

        self._store['topic_top_term_matrix'] = [
            [
                (term_id, id2word[term_id], topic_terms[term_id])
                for term_id in matutils.argsort(topic_terms, option.terms_per_topic, reverse=True)
            ]
            for topic_terms in topic_term_matrix
        ]

        super()._output_result(result, output_dir)

    def _output_topics(self, result, output_dir):
        id2word, _ = self._dtm_runner.get_dtm()
        _, evaluation, option = result

        topic_matrix = evaluation.topic_matrix
        topic_top_term_matrix = self._store['topic_top_term_matrix']

        with open(os.path.join(output_dir, 'topics.csv'), 'w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([
                'topic_id',
                'topic_coherence',
                'topic_keywords',
            ])
            for topic_id in range(option.num_topics):
                csv_writer.writerow([
                    topic_id,
                    topic_matrix[topic_id],
                    ', '.join([term_name for _, term_name, _ in topic_top_term_matrix[topic_id]]),
                ])

    def _output_topic_terms(self, result, output_dir):
        id2word, _ = self._dtm_runner.get_dtm()
        _, evaluation, option = result

        topic_top_term_matrix = self._store['topic_top_term_matrix']

        with open(os.path.join(output_dir, 'topic_terms.csv'), 'w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([
                'topic_id',
                'term_name',
                'term_probability',
            ])
            for topic_id in range(option.num_topics):
                for _, term_name, term_probability in topic_top_term_matrix[topic_id]:
                    csv_writer.writerow([
                        topic_id,
                        term_name,
                        term_probability,
                    ])

    def _output_document_topics(self, result, output_dir):
        docs = self._dtm_runner.get_docs()
        id2word, _ = self._dtm_runner.get_dtm()
        _, evaluation, option = result

        topic_top_term_matrix = self._store['topic_top_term_matrix']
        document_topic_matrix = evaluation.document_topic_matrix

        with open(os.path.join(output_dir, 'document_topics.csv'), 'w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([
                'doc_eid',
                'doc_title',
                'doc_file',
                'topic_id',
                'topic_probability',
                'topic_keywords',
            ])

            for doc_id in range(len(docs)):
                doc = docs[doc_id]
                for topic_id in range(option.num_topics):
                    csv_writer.writerow([
                        doc['eid'],
                        doc['title'],
                        doc['file'],
                        topic_id,
                        document_topic_matrix[doc_id][topic_id],
                        ', '.join([term_name for _, term_name, _ in topic_top_term_matrix[topic_id]]),
                    ])

        # document_dominant_topics = np.argmax(document_topic_matrix, axis=1)
        # with open(os.path.join(output_dir, 'document_dominant_topic.csv'), 'w', newline='', encoding='utf-8') as file:
        #     csv_writer = csv.writer(file)
        #     csv_writer.writerow([
        #         'doc_eid',
        #         'doc_title',
        #         'doc_file',
        #         'topic_id',
        #         'topic_probability',
        #         'topic_keywords',
        #     ])
        #
        #     for doc_id in range(len(docs)):
        #         doc = docs[doc_id]
        #         topic_id = document_dominant_topics[doc_id]
        #         csv_writer.writerow([
        #             doc['eid'],
        #             doc['title'],
        #             doc['file'],
        #             topic_id,
        #             document_topic_matrix[doc_id][topic_id],
        #             ', '.join([term_name for _, term_name, _ in topic_top_term_matrix[topic_id]]),
        #         ])

    def _output_visualization(self, result, output_dir):
        id2word, corpus = self._dtm_runner.get_dtm()

        topic_model, evaluation, _ = result

        topic_model.create_visualization(corpus, id2word, output_dir, mds='mmds')

    def _output_umap_result(self, result, output_dir):
        topic_model, evaluation, option = result

        topic_matrix = evaluation.topic_matrix
        topic_term_matrix = evaluation.topic_term_matrix
        document_topic_matrix = evaluation.document_topic_matrix

        with open(os.path.join(output_dir, 'umap_data.json'), 'w', encoding='utf-8') as file:
            umap_data = {
                'num_topics': option.num_topics,
                'terms_per_topic': option.terms_per_topic,
                'random_seed': option.random_seed,
                'topic_matrix': topic_matrix,
                'topic_term_matrix': topic_term_matrix,
                'document_topic_matrix': document_topic_matrix,
            }
            json.dump(umap_data, file, indent=4)
