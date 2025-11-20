import csv
import os

from topic_model_runners.base.dtm_runner import DtmRunner
from topic_model_runners.base.topic_model_option import TopicModelOption


class TopicModelEvaluation:
    def __init__(self):
        self.perplexity = 0.0
        self.coherence = 0.0

        self.topic_matrix = []
        self.topic_term_matrix = []
        self.document_topic_matrix = []

    def evaluate(self, dtm_runner: DtmRunner, topic_model, option: TopicModelOption):
        return self

    def save(
            self,
            model_file_path,
            topics_file_path,
            topic_terms_file_path,
            document_topics_file_path
    ):
        with open(model_file_path, 'w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([self.perplexity, self.coherence])

        with open(topics_file_path, 'w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            for topic_id, topic_coherence in enumerate(self.topic_matrix):
                csv_writer.writerow([topic_id, topic_coherence])

        with open(topic_terms_file_path, 'w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            for topic_id, topic_terms in enumerate(self.topic_term_matrix):
                for term_id, term_probability in enumerate(topic_terms):
                    csv_writer.writerow([topic_id, term_id, term_probability])

        with open(document_topics_file_path, 'w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            for doc_id, document_topics in enumerate(self.document_topic_matrix):
                for topic_id, topic_probability in enumerate(document_topics):
                    csv_writer.writerow([doc_id, topic_id, topic_probability])

        return self

    def load(
            self,
            model_file_path,
            topics_file_path,
            topic_terms_file_path,
            document_topics_file_path
    ):
        if (not os.path.isfile(model_file_path)
                or not os.path.isfile(topics_file_path)
                or not os.path.isfile(topic_terms_file_path)
                or not os.path.isfile(document_topics_file_path)):
            return None

        with open(model_file_path, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.perplexity = float(row[0])
                self.coherence = float(row[1])
                break

        with open(topics_file_path, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                topic_coherence = float(row[1])
                self.topic_matrix.append(topic_coherence)

        with open(topic_terms_file_path, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)

            prev_topic_id = -1
            topic_terms = []
            for row in csv_reader:
                topic_id = row[0]

                if prev_topic_id != -1:
                    if topic_id != prev_topic_id:
                        self.topic_term_matrix.append(topic_terms)

                        topic_terms = []

                term_probability = float(row[2])
                topic_terms.append(term_probability)

                prev_topic_id = topic_id
            self.topic_term_matrix.append(topic_terms)

        with open(document_topics_file_path, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)

            prev_doc_id = -1
            document_topics = []
            for row in csv_reader:
                doc_id = row[0]

                if prev_doc_id != -1:
                    if doc_id != prev_doc_id:
                        self.document_topic_matrix.append(document_topics)

                        document_topics = []

                topic_probability = float(row[2])
                document_topics.append(topic_probability)

                prev_doc_id = doc_id
            self.document_topic_matrix.append(document_topics)

        return self
