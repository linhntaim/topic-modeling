import argparse
import base64
import csv
import glob
import json
import os.path

import hdbscan
import numpy as np
import pandas

from dash import Dash, html, dcc, Output, Input, State
from dotenv import load_dotenv
from gensim import matutils
from plotly import express as px, graph_objs as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from base.ab_fixed_umap import ABFixedUMAP

# Load environment variables from .env file
load_dotenv()


# region Common Handlers
def calculate_umap_embedding(
        data,
        n_components,
        n_neighbors,
        min_dist,
        metric,
        random_state,
):
    return ABFixedUMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    ).fit_transform(data)


def create_figure_from_umap_embbeding(
        n_components,
        embedding,
        figure_data,
        figure_hover_data,
        figure_labels,
        figure_color,
        figure_title,
):
    global discrete_colors

    if n_components == 3:
        return px.scatter_3d(
            pandas.DataFrame({
                                 'x': embedding[:, 0],
                                 'y': embedding[:, 1],
                                 'z': embedding[:, 2],
                             } | figure_data),
            x='x',
            y='y',
            z='z',
            hover_data={
                           'x': False,
                           'y': False,
                           'z': False,
                       } | figure_hover_data,
            labels={
                       'x': 'UMAP Dimension 1',
                       'y': 'UMAP Dimension 2',
                       'z': 'UMAP Dimension 3',
                   } | figure_labels,
            color=figure_color,
            color_discrete_sequence=discrete_colors,
            title=figure_title,
            height=980,
        ).update_traces(
            marker_size=10,
            hovertemplate=None,
            hoverinfo='none',
        ).update_layout(
            hoverdistance=1,
        )

    if n_components == 2:
        return px.scatter(
            pandas.DataFrame({
                                 'x': embedding[:, 0],
                                 'y': embedding[:, 1],
                             } | figure_data),
            x='x',
            y='y',
            hover_data={
                           'x': False,
                           'y': False,
                       } | figure_hover_data,
            labels={
                       'x': 'UMAP Dimension 1',
                       'y': 'UMAP Dimension 2',
                   } | figure_labels,
            color=figure_color,
            color_discrete_sequence=discrete_colors,
            title=figure_title,
            height=980,
        ).update_traces(
            marker_size=12,
            hovertemplate=None,
            hoverinfo='none',
        ).update_layout(
            hoverdistance=1,
        )

    if n_components == 1:
        return px.scatter(
            pandas.DataFrame({
                                 'x': embedding.flatten(),
                                 'y': [0] * len(embedding),
                             } | figure_data),
            x='x',
            y='y',
            hover_data={
                           'x': False,
                           'y': False,
                       } | figure_hover_data,
            labels={
                       'x': 'UMAP Dimension 1',
                       'y': 'UMAP Dimension 2',
                   } | figure_labels,
            color=figure_color,
            color_discrete_sequence=discrete_colors,
            title=figure_title,
            height=980,
        ).update_traces(
            marker_size=12,
            hovertemplate=None,
            hoverinfo='none',
        ).update_layout(
            hoverdistance=1,
        )

    return None


def update_graph_hover(hover_data, transform_customdata_func):
    if hover_data is None:
        return '', {'display': 'none'}

    return (
        transform_customdata_func(hover_data['points'][0]['customdata']),
        {
            'display': 'block',
            'position': 'fixed',
            'bottom': '10px',
            'left': '10px',
            'padding': '10px',
            'background': 'black',
            'border': '1px solid black',
            'color': 'white',
            'z-index': '9999',
        }
    )


def naming(name, names):
    return names[name] if name in names else name


def calculate_tote_umap_graph(stored_metadata, stored_data, stored_umap_params, embedding=None, named_topics=None):
    if named_topics is None:
        named_topics = {}

    if stored_metadata is None:
        return (
            None,
            html.P(children=f'Metadata not found!', style={'color': 'red'}),
            None,
        )
    if stored_data is None:
        return (
            None,
            html.P(children=f'Data not found!', style={'color': 'red'}),
            None,
        )

    meta_docs, meta_terms = stored_metadata
    num_docs = len(meta_docs)
    num_terms = len(meta_terms)

    (
        num_topics,
        terms_per_topic,
        random_seed,
        _,
        topic_term_matrix,
        document_topic_matrix,
    ) = stored_data
    topic_keywords = [[meta_terms[term_id] for term_id in matutils.argsort(topic_terms, terms_per_topic, reverse=True)]
                      for topic_terms in topic_term_matrix]

    (
        n_components_value,
        n_neighbors_value,
        min_dist_value,
        metric_value,
    ) = stored_umap_params

    if embedding is None:
        embedding = calculate_umap_embedding(
            topic_term_matrix,
            n_components=n_components_value,
            n_neighbors=n_neighbors_value,
            min_dist=min_dist_value,
            metric=metric_value,
            random_state=random_seed,
        )

    fig = create_figure_from_umap_embbeding(
        n_components_value,
        embedding,
        {
            'topic': [naming(f'Topic {topic_id}', named_topics) for topic_id in range(num_topics)],
            'topic_keywords': [', '.join(topic_keywords[topic_id]) for topic_id in range(num_topics)],
        },
        {
            'topic': True,
            'topic_keywords': True,
        },
        {
            'topic': 'Topic',
            'topic_keywords': 'Keywords',
        },
        'topic',
        'UMAP Embedding of Topic-Term Matrix',
    )

    return (
        embedding,
        [
            f'Documents: {num_docs} / Terms: {num_terms} / Topics: {num_topics}',
            html.Br(),
            html.A(
                'UMAP parameters',
                href='https://umap-learn.readthedocs.io/en/latest/parameters.html',
                target='_blank',
            ),
            ':',
            html.Br(),
            f'- n_components: {n_components_value}',
            html.Br(),
            f'- n_neighbor: {n_neighbors_value}',
            html.Br(),
            f'- min_dist: {min_dist_value}',
            html.Br(),
            f'- metric: {metric_value}',
        ],
        fig,
    )


def calculate_doto_umap_graph(stored_metadata, stored_data, stored_umap_params, embedding=None, named_topics=None):
    if named_topics is None:
        named_topics = {}

    if stored_metadata is None:
        return (
            None,
            html.P(children=f'Metadata not found!', style={'color': 'red'}),
            None,
        )
    if stored_data is None:
        return (
            None,
            html.P(children=f'Data not found!', style={'color': 'red'}),
            None,
        )

    meta_docs, meta_terms = stored_metadata
    num_docs = len(meta_docs)
    num_terms = len(meta_terms)

    (
        num_topics,
        terms_per_topic,
        random_seed,
        _,
        topic_term_matrix,
        document_topic_matrix,
    ) = stored_data
    topic_keywords = [[meta_terms[term_id] for term_id in matutils.argsort(topic_terms, terms_per_topic, reverse=True)]
                      for topic_terms in topic_term_matrix]
    document_dominant_topics = np.argmax(document_topic_matrix, axis=1)

    (
        n_components_value,
        n_neighbors_value,
        min_dist_value,
        metric_value,
    ) = stored_umap_params

    if embedding is None:
        embedding = calculate_umap_embedding(
            document_topic_matrix,
            n_components=n_components_value,
            n_neighbors=n_neighbors_value,
            min_dist=min_dist_value,
            metric=metric_value,
            random_state=random_seed,
        )

    fig = create_figure_from_umap_embbeding(
        n_components_value,
        embedding,
        {
            'document': [meta_docs[doc_id] for doc_id in range(num_docs)],
            'document_dominant_topic': [naming(f'Topic {topic_id}', named_topics)
                                        for topic_id in document_dominant_topics],
            'document_dominant_keywords': [', '.join(topic_keywords[topic_id])
                                           for topic_id in document_dominant_topics],
        },
        {
            'document': True,
            'document_dominant_topic': True,
            'document_dominant_keywords': True,
        },
        {
            'document': 'Document',
            'document_dominant_topic': 'Dominant Topic',
            'document_dominant_keywords': 'Dominant Keywords',
        },
        'document_dominant_topic',
        'UMAP Embedding of Document-Topic Matrix',
    )

    return (
        embedding,
        [
            f'Documents: {num_docs} / Terms: {num_terms} / Topics: {num_topics}',
            html.Br(),
            html.A(
                'UMAP parameters',
                href='https://umap-learn.readthedocs.io/en/latest/parameters.html',
                target='_blank',
            ),
            ':',
            html.Br(),
            f'- n_components: {n_components_value}',
            html.Br(),
            f'- n_neighbor: {n_neighbors_value}',
            html.Br(),
            f'- min_dist: {min_dist_value}',
            html.Br(),
            f'- metric: {metric_value}',
        ],
        fig,
    )


def calculate_tote_clustering_graph(
        stored_metadata,
        stored_data,
        stored_tote_umap_embedding,
        stored_doto_umap_embedding,
        stored_tote_umap_params,
        stored_doto_umap_params,
        clustering_method_params,
        clustering_method_name,
        clustering_method,
        named_topics=None,
        named_labels=None,
):
    if named_topics is None:
        named_topics = {}
    if named_labels is None:
        named_labels = {}

    if stored_metadata is None:
        return (
            html.P(children=f'Metadata not found!', style={'color': 'red'}),
            None,
            None,
            None,
            None,
        )
    if stored_data is None:
        return (
            html.P(children=f'Data not found!', style={'color': 'red'}),
            None,
            None,
            None,
            None,
        )
    if stored_tote_umap_embedding is None:
        return (
            html.P(children=f'UMAP Embedding of Topic-Term Matrix not found!', style={'color': 'red'}),
            None,
            None,
            None,
            None,
        )
    if stored_doto_umap_embedding is None:
        return (
            html.P(children=f'UMAP Embedding of Document-Topic Matrix not found!', style={'color': 'red'}),
            None,
            None,
            None,
            None,
        )

    meta_docs, meta_terms = stored_metadata
    num_docs = len(meta_docs)
    num_terms = len(meta_terms)

    (
        num_topics,
        terms_per_topic,
        random_seed,
        _,
        topic_term_matrix,
        document_topic_matrix,
    ) = stored_data
    topic_keywords = [[meta_terms[term_id] for term_id in matutils.argsort(topic_terms, terms_per_topic, reverse=True)]
                      for topic_terms in topic_term_matrix]
    document_dominant_topics = np.argmax(document_topic_matrix, axis=1)

    tote_embedding = np.array(stored_tote_umap_embedding)

    doto_embedding = np.array(stored_doto_umap_embedding)

    (
        tote_n_components_value,
        tote_n_neighbors_value,
        tote_min_dist_value,
        tote_metric_value,
    ) = stored_tote_umap_params

    (
        doto_n_components_value,
        doto_n_neighbors_value,
        doto_min_dist_value,
        doto_metric_value,
    ) = stored_doto_umap_params

    labels, labels_output, labels_config = clustering_method(tote_embedding, clustering_method_params, random_seed)

    tote_data = {
        'topic': [naming(f'Topic {topic_id}', named_topics) for topic_id in range(num_topics)],
        'topic_keywords': [', '.join(topic_keywords[topic_id]) for topic_id in range(num_topics)],
        'topic_label': [naming(f'Label {labels[topic_id]}', named_labels) for topic_id in range(num_topics)],
    }

    tote_fig = create_figure_from_umap_embbeding(
        tote_n_components_value,
        tote_embedding,
        tote_data,
        {
            'topic': True,
            'topic_keywords': True,
            'topic_label': True,
        },
        {
            'topic': 'Topic',
            'topic_keywords': 'Keywords',
            'topic_label': 'Group',
        },
        'topic_label',
        f'Clusters using {clustering_method_name} of UMAP Embedding of Topic-Term Matrix',
    )

    tote_download = dcc.send_data_frame(
        pandas.DataFrame({
            'Topic': tote_data['topic'],
            'Keywords': tote_data['topic_keywords'],
            'Label': tote_data['topic_label'],
        }).to_csv,
        f'topic_labels_using_{clustering_method_name}.csv',
    )

    tote_doto_data = {
        'document': [meta_docs[doc_id] for doc_id in range(num_docs)],
        'document_dominant_topic': [naming(f'Topic {topic_id}', named_topics) for topic_id in document_dominant_topics],
        'document_dominant_keywords': [', '.join(topic_keywords[topic_id]) for topic_id in document_dominant_topics],
        'document_dominant_label': [naming(f'Label {labels[topic_id]}', named_labels)
                                    for topic_id in document_dominant_topics],
    }

    tote_doto_fig = create_figure_from_umap_embbeding(
        doto_n_components_value,
        doto_embedding,
        tote_doto_data,
        {
            'document': True,
            'document_dominant_topic': True,
            'document_dominant_keywords': True,
            'document_dominant_label': True,
        },
        {
            'document': 'Document',
            'document_dominant_topic': 'Dominant Topic',
            'document_dominant_keywords': 'Dominant Keywords',
            'document_dominant_label': 'Dominant Group',
        },
        'document_dominant_label',
        f'Clusters by Dominant Topic using {clustering_method_name} of UMAP Embedding of Document-Topic Matrix',
    )

    tote_doto_download = dcc.send_data_frame(
        pandas.DataFrame({
            'Document': tote_doto_data['document'],
            'Dominant Topic': tote_doto_data['document_dominant_topic'],
            'Dominant Keywords': tote_doto_data['document_dominant_keywords'],
            'Dominant Label': tote_doto_data['document_dominant_label'],
        }).to_csv,
        f'document_labels_using_{clustering_method_name}.csv',
    )

    tote_config_download = dict(
        content=json.dumps({
            "umap_embeddings": {
                "topic_term_matrix": {
                    "n_components": tote_n_components_value,
                    "n_neighbors": tote_n_neighbors_value,
                    "min_dist": tote_min_dist_value,
                    "metric": tote_metric_value
                },
                "document_topic_matrix": {
                    "n_components": doto_n_components_value,
                    "n_neighbors": doto_n_neighbors_value,
                    "min_dist": doto_min_dist_value,
                    "metric": doto_metric_value
                }
            },
            "topic_clustering": labels_config
        }, indent=4),
        filename=f'config_{clustering_method_name}.json'
    )

    output = [
        f'Documents: {num_docs} / Terms: {num_terms} / Topics: {num_topics}',
        html.Br(),
    ]
    output.extend(labels_output)
    return (
        output,
        tote_fig,
        tote_download,
        tote_doto_fig,
        tote_doto_download,
        tote_config_download,
    )


def calculate_kmeans_clusters(tote_embedding, method_params, random_state):
    n_clusters_value = method_params

    kmeans = KMeans(
        n_clusters=n_clusters_value,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(tote_embedding)
    return (
        labels,
        [
            'KMeans parameters:',
            html.Br(),
            f'- n_clusters: {n_clusters_value}',
        ],
        {
            'kmeans': {
                'n_clusters': n_clusters_value
            }
        }
    )


def calculate_hdbscan_clusters(tote_embedding, method_params, random_state):
    min_cluster_size_value, min_samples_value, metric_value = method_params

    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size_value,
        min_samples=min_samples_value,
        metric=metric_value,
    ).fit_predict(tote_embedding)

    return (
        labels,
        [
            html.A(
                'HDBSCAN parameters',
                href='https://hdbscan.readthedocs.io/en/latest/parameter_selection.html',
                target='_blank',
            ),
            ':',
            html.Br(),
            f'- min_cluster_size: {min_cluster_size_value}',
            html.Br(),
            f'- min_samples: {min_samples_value}',
            html.Br(),
            f'- metric: {metric_value}',
        ],
        {
            'hdbscan': {
                'min_cluster_size': min_cluster_size_value,
                'min_samples': min_samples_value,
                'metric': metric_value,
            }
        }
    )


def update_tote_clustering_hover(hover_data):
    return update_graph_hover(hover_data, lambda customdata: [
        f'Topic: {customdata[0]}',
        html.Br(),
        f'Keywords: {customdata[1]}',
        html.Br(),
        f'Group: {customdata[2]}',
    ])


def update_tote_doto_clustering_hover(hover_data):
    return update_graph_hover(hover_data, lambda customdata: [
        f'Document: {customdata[0]}',
        html.Br(),
        f'Dominant Topic: {customdata[1]}',
        html.Br(),
        f'Dominant Keywords: {customdata[2]}',
        html.Br(),
        f'Dominant Group: {customdata[3]}',
    ])


# endregion

# region Argv
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port',
                    help='Set port',
                    default='8050')
parser.add_argument('--debug',
                    action='store_true',
                    help='Enable debug')
parser.add_argument('-l', '--load',
                    help='Load data from a directory')
parser.add_argument('-f', '--fixed',
                    action='store_true',
                    help='Enable load from fixed location got from `UMAP_LOAD_DIR`')
parser.add_argument('-tnc', '--topic-name-col',
                    help='Set topic shortname column',
                    default='topic_shortname')
parser.add_argument('-gnc', '--group-name-col',
                    help='Set group shortname column',
                    default='group_shortname')
args = parser.parse_args()

port = int(args.port)
debug = args.debug
loading_dir = args.load
if args.fixed:
    loading_dir = os.getenv('UMAP_LOAD_DIR')
topic_name_col = args.topic_name_col
group_name_col = args.group_name_col
# endregion

# region Global vars
discrete_colors = [
    *px.colors.qualitative.Light24,
    *px.colors.qualitative.Dark24,
    *px.colors.qualitative.Set3,
]

readonly = False
clustering_method_kmeans = 'KMeans'
clustering_method_hdbscan = 'HDBSCAN'

# region UMAP params
umap_n_components_min = 1
umap_n_components_max = 3
umap_n_components_step = 1

umap_n_neighbors_min = 2
umap_n_neighbors_max = 200
umap_n_neighbors_step = 1

umap_min_dist_min = 0.00
umap_min_dist_max = 1.00
umap_min_dist_step = 0.01

umap_metrics = [
    'euclidean',
    'manhattan',
    'chebyshev',
    'minkowski',

    'canberra',
    'braycurtis',
    'haversine',

    'mahalanobis',
    'wminkowski',
    'seuclidean',

    'cosine',
    'correlation',

    'hamming',
    'jaccard',
    'dice',
    'russellrao',
    'kulsinski',
    'rogerstanimoto',
    'sokalmichener',
    'sokalsneath',
    'yule',

    'hellinger',
]

tote_umap_n_components = 2
tote_umap_n_neighbors = 5
tote_umap_min_dist = 0.1
tote_umap_metric = 'hellinger'

tote_umap_output = []
tote_umap_fig = None

doto_umap_n_components = 2
doto_umap_n_neighbors = 5
doto_umap_min_dist = 0.1
doto_umap_metric = 'hellinger'

doto_umap_output = []
doto_umap_fig = None
# endregion

# region KMeans params
kmeans_n_clusters_min = 2
kmeans_n_clusters_max = 20
kmeans_n_clusters_step = 1

scoring_kmeans_n_clusters = 2
kmeans_n_clusters = 2
kmeans_enabled = True

tote_kmeans_output = []
tote_kmeans_fig = None
tote_doto_kmeans_fig = None
# endregion

# region HDBSCAN params
hdbscan_min_cluster_size_min = 2
hdbscan_min_cluster_size_max = 20
hdbscan_min_cluster_size_step = 1

hdbscan_min_samples_min = 1
hdbscan_min_samples_max = 20
hdbscan_min_samples_step = 1

hdbscan_metrics = [
    'braycurtis',
    'canberra',
    'chebyshev',
    'cityblock',
    'dice',
    'euclidean',
    'hamming',
    'haversine',
    'infinity',
    'jaccard',
    'kulsinski',
    'l1',
    'l2',
    'mahalanobis',
    'manhattan',
    'matching',
    'minkowski',
    'p',
    'pyfunc',
    'rogerstanimoto',
    'russellrao',
    'seuclidean',
    'sokalmichener',
    'sokalsneath',
    'wminkowski',
]

hdbscan_enabled = True
hdbscan_min_cluster_size = 2
hdbscan_min_samples = 1
hdbscan_metric = 'euclidean'

tote_hdbscan_output = []
tote_hdbscan_fig = None
tote_doto_hdbscan_fig = None
# endregion
# endregion

# region Load
if loading_dir is not None and loading_dir != '' and os.path.isdir(loading_dir):
    umap_metadata_file_path = glob.glob(os.path.join(loading_dir, '*umap_metadata*.json'))
    if len(umap_metadata_file_path) > 0:
        umap_metadata_file_path = umap_metadata_file_path[0]
        umap_data_file_path = glob.glob(os.path.join(loading_dir, '*umap_data*.json'))
        if len(umap_data_file_path) > 0:
            umap_data_file_path = umap_data_file_path[0]
            config_file_path = glob.glob(os.path.join(loading_dir, '*config*.json'))
            if len(config_file_path) > 0:
                config_file_path = config_file_path[0]

                with open(umap_metadata_file_path, 'r', encoding='utf-8') as f:
                    loaded_metadata = json.loads(f.read())
                with open(umap_data_file_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.loads(f.read())
                with open(config_file_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.loads(f.read())

                tote_umap_config = loaded_config['umap_embeddings']['topic_term_matrix']
                tote_umap_n_components = tote_umap_config['n_components']
                if (tote_umap_n_components > umap_n_components_max
                        or tote_umap_n_components < umap_n_components_min
                        or tote_umap_n_components % umap_n_components_step != 0):
                    raise 'Invalid tote_umap_n_components'
                tote_umap_n_neighbors = tote_umap_config['n_neighbors']
                if (tote_umap_n_neighbors > umap_n_neighbors_max
                        or tote_umap_n_neighbors < umap_n_neighbors_min
                        or tote_umap_n_neighbors % umap_n_neighbors_step != 0):
                    raise 'Invalid tote_umap_n_neighbors'
                tote_umap_min_dist = tote_umap_config['min_dist']
                if (tote_umap_min_dist > umap_min_dist_max
                        or tote_umap_min_dist < umap_min_dist_min
                        or tote_umap_min_dist % umap_min_dist_step != 0):
                    raise 'Invalid tote_umap_min_dist'
                tote_umap_metric = tote_umap_config['metric']
                if tote_umap_metric not in umap_metrics:
                    raise 'Invalid tote_umap_metric'

                doto_umap_config = loaded_config['umap_embeddings']['document_topic_matrix']
                doto_umap_n_components = doto_umap_config['n_components']
                if (doto_umap_n_components > umap_n_components_max
                        or doto_umap_n_components < umap_n_components_min
                        or doto_umap_n_components % umap_n_components_step != 0):
                    raise 'Invalid doto_umap_n_components'
                doto_umap_n_neighbors = doto_umap_config['n_neighbors']
                if (doto_umap_n_neighbors > umap_n_neighbors_max
                        or doto_umap_n_neighbors < umap_n_neighbors_min
                        or doto_umap_n_neighbors % umap_n_neighbors_step != 0):
                    raise 'Invalid doto_umap_n_neighbors'
                doto_umap_min_dist = doto_umap_config['min_dist']
                if (doto_umap_min_dist > umap_min_dist_max
                        or doto_umap_min_dist < umap_min_dist_min
                        or doto_umap_min_dist % umap_min_dist_step != 0):
                    raise 'Invalid doto_umap_min_dist'
                doto_umap_metric = doto_umap_config['metric']
                if doto_umap_metric not in umap_metrics:
                    raise 'Invalid doto_umap_metric'

                tote_umap_embedding = calculate_umap_embedding(
                    loaded_data['topic_term_matrix'],
                    n_components=tote_umap_n_components,
                    n_neighbors=tote_umap_n_neighbors,
                    min_dist=tote_umap_min_dist,
                    metric=tote_umap_metric,
                    random_state=loaded_data['random_seed'],
                )

                doto_umap_embedding = calculate_umap_embedding(
                    loaded_data['document_topic_matrix'],
                    n_components=doto_umap_n_components,
                    n_neighbors=doto_umap_n_neighbors,
                    min_dist=doto_umap_min_dist,
                    metric=doto_umap_metric,
                    random_state=loaded_data['random_seed'],
                )

                loaded_named_topics = {}
                loaded_named_labels = {}
                if 'kmeans' in loaded_config['topic_clustering']:
                    topic_labels_file_path = glob.glob(
                        os.path.join(loading_dir, f'*topic_labels_using_{clustering_method_hdbscan}*.csv'))
                    if len(topic_labels_file_path) > 0:
                        topic_labels_file_path = topic_labels_file_path[0]
                        with open(topic_labels_file_path, 'r', newline='', encoding='utf-8') as f:
                            csv_reader = csv.DictReader(f)
                            for row in csv_reader:
                                loaded_named_topics[row['Topic']] = row[topic_name_col]
                                loaded_named_labels[row['Label']] = row[group_name_col]

                    parameters = loaded_config['topic_clustering']['kmeans']

                    kmeans_n_clusters_max = loaded_data['num_topics'] - 1

                    kmeans_n_clusters = parameters['n_clusters']
                    if (kmeans_n_clusters > kmeans_n_clusters_max
                            or kmeans_n_clusters < kmeans_n_clusters_min
                            or kmeans_n_clusters % kmeans_n_clusters_step != 0):
                        raise 'Invalid kmeans_n_clusters'

                    tote_kmeans_output, tote_kmeans_fig, _, tote_doto_kmeans_fig, _, _ = calculate_tote_clustering_graph(
                        (
                            loaded_metadata['docs'],
                            loaded_metadata['terms'],
                        ),
                        (
                            loaded_data['num_topics'],
                            loaded_data['terms_per_topic'],
                            loaded_data['random_seed'],
                            loaded_data['topic_matrix'],
                            loaded_data['topic_term_matrix'],
                            loaded_data['document_topic_matrix'],
                        ),
                        tote_umap_embedding,
                        doto_umap_embedding,
                        (
                            tote_umap_n_components,
                            tote_umap_n_neighbors,
                            tote_umap_min_dist,
                            tote_umap_metric,
                        ),
                        (
                            doto_umap_n_components,
                            doto_umap_n_neighbors,
                            doto_umap_min_dist,
                            doto_umap_metric,
                        ),
                        kmeans_n_clusters,
                        clustering_method_kmeans,
                        calculate_kmeans_clusters,
                        loaded_named_topics,
                        loaded_named_labels,
                    )
                else:
                    kmeans_enabled = False

                    if 'hdbscan' in loaded_config['topic_clustering']:
                        topic_labels_file_path = glob.glob(
                            os.path.join(loading_dir, f'*topic_labels_using_{clustering_method_hdbscan}*.csv'))
                        if len(topic_labels_file_path) > 0:
                            topic_labels_file_path = topic_labels_file_path[0]
                            with open(topic_labels_file_path, 'r', newline='', encoding='utf-8') as f:
                                csv_reader = csv.DictReader(f)
                                for row in csv_reader:
                                    loaded_named_topics[row['Topic']] = row[topic_name_col]
                                    loaded_named_labels[row['Label']] = row[group_name_col]

                        parameters = loaded_config['topic_clustering']['hdbscan']

                        hdbscan_min_cluster_size_max = loaded_data['num_topics']
                        hdbscan_min_samples_max = loaded_data['num_topics']

                        hdbscan_min_cluster_size = parameters['min_cluster_size']
                        if (hdbscan_min_cluster_size > hdbscan_min_cluster_size_max
                                or hdbscan_min_cluster_size < hdbscan_min_cluster_size_min
                                or hdbscan_min_cluster_size % hdbscan_min_cluster_size_step != 0):
                            raise 'Invalid hdbscan_min_cluster_size'

                        hdbscan_min_samples = parameters['min_samples']
                        if (hdbscan_min_samples > hdbscan_min_samples_max
                                or hdbscan_min_samples < hdbscan_min_samples_min
                                or hdbscan_min_samples % hdbscan_min_samples_step != 0):
                            raise 'Invalid hdbscan_min_samples'

                        hdbscan_metric = parameters['metric']
                        if hdbscan_metric not in hdbscan_metrics:
                            raise 'Invalid hdbscan_metric'

                        tote_hdbscan_output, tote_hdbscan_fig, _, tote_doto_hdbscan_fig, _, _ = calculate_tote_clustering_graph(
                            (
                                loaded_metadata['docs'],
                                loaded_metadata['terms'],
                            ),
                            (
                                loaded_data['num_topics'],
                                loaded_data['terms_per_topic'],
                                loaded_data['random_seed'],
                                loaded_data['topic_matrix'],
                                loaded_data['topic_term_matrix'],
                                loaded_data['document_topic_matrix'],
                            ),
                            tote_umap_embedding,
                            doto_umap_embedding,
                            (
                                tote_umap_n_components,
                                tote_umap_n_neighbors,
                                tote_umap_min_dist,
                                tote_umap_metric,
                            ),
                            (
                                doto_umap_n_components,
                                doto_umap_n_neighbors,
                                doto_umap_min_dist,
                                doto_umap_metric,
                            ),
                            (
                                hdbscan_min_cluster_size,
                                hdbscan_min_samples,
                                hdbscan_metric
                            ),
                            clustering_method_hdbscan,
                            calculate_hdbscan_clusters,
                            loaded_named_topics,
                            loaded_named_labels,
                        )
                    else:
                        hdbscan_enabled = False

                _, tote_umap_output, tote_umap_fig = calculate_tote_umap_graph(
                    (
                        loaded_metadata['docs'],
                        loaded_metadata['terms'],
                    ),
                    (
                        loaded_data['num_topics'],
                        loaded_data['terms_per_topic'],
                        loaded_data['random_seed'],
                        loaded_data['topic_matrix'],
                        loaded_data['topic_term_matrix'],
                        loaded_data['document_topic_matrix'],
                    ),
                    (
                        tote_umap_n_components,
                        tote_umap_n_neighbors,
                        tote_umap_min_dist,
                        tote_umap_metric,
                    ),
                    tote_umap_embedding,
                    loaded_named_topics,
                )

                _, doto_umap_output, doto_umap_fig = calculate_doto_umap_graph(
                    (
                        loaded_metadata['docs'],
                        loaded_metadata['terms'],
                    ),
                    (
                        loaded_data['num_topics'],
                        loaded_data['terms_per_topic'],
                        loaded_data['random_seed'],
                        loaded_data['topic_matrix'],
                        loaded_data['topic_term_matrix'],
                        loaded_data['document_topic_matrix'],
                    ),
                    (
                        doto_umap_n_components,
                        doto_umap_n_neighbors,
                        doto_umap_min_dist,
                        doto_umap_metric,
                    ),
                    doto_umap_embedding,
                    loaded_named_topics,
                )

                readonly = True
# endregion

# region Main Layout
app = Dash(
    __name__,
    title='UMAP Embeddings of Topic Modeling',
)
app.layout = html.Div([
    html.Div(
        children=[
            html.H1(children='Chapter 0. Import Topic Modeling Result'),

            # region Main Layout of Import Data
            dcc.Store(id='metadata-store', storage_type='memory'),
            dcc.Upload(
                id='metadata-uploader',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select "umap_metadata.json" Files')
                ]),
                accept='application/json, text/plain',
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                },
            ),
            html.Div(
                id='metadata-output',
                style={
                    'margin': '25px 0',
                }
            ),

            dcc.Store(id='data-store', storage_type='memory'),
            dcc.Upload(
                id='data-uploader',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select "umap_data.json" Files')
                ]),
                accept='application/json, text/plain',
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                },
            ),
            html.Div(
                id='data-output',
                style={
                    'margin': '25px 0',
                }
            ),
            # endregion
        ],
        style={
            'display': 'none' if readonly else 'block',
        }
    ),

    html.H1(children='Chapter 1. UMAP Embeddings'),

    html.H2(children='Chapter 1.1. UMAP Embedding of Topic-Term Matrix'),

    # region Main Layout of UMAP Embedding of Topic-Term Matrix
    html.Div(
        children=[
            dcc.Store(id='tote-umap-params-store', storage_type='memory'),
            dcc.Store(id='tote-umap-embedding-store', storage_type='memory'),
            html.Div(style={
                'width': '100%',
                'height': '25px',
            }),
            dcc.Slider(
                umap_n_components_min, umap_n_components_max, umap_n_components_step,
                value=tote_umap_n_components,
                id='tote-umap-n-components-slider',
                marks=None,
                tooltip={
                    'always_visible': True,
                    'template': 'n_components = {value}'
                },
            ),
            html.Div(style={
                'width': '100%',
                'height': '25px',
            }),
            dcc.Slider(
                umap_n_neighbors_min, umap_n_neighbors_max, umap_n_neighbors_step,
                value=tote_umap_n_neighbors,
                id='tote-umap-n-neighbors-slider',
                marks=None,
                tooltip={
                    'always_visible': True,
                    'template': 'n_neighbors = {value}'
                },
            ),
            html.Div(style={
                'width': '100%',
                'height': '25px',
            }),
            dcc.Slider(
                umap_min_dist_min, umap_min_dist_max, umap_min_dist_step,
                value=tote_umap_min_dist,
                id='tote-umap-min-dist-slider',
                marks=None,
                tooltip={
                    'always_visible': True,
                    'template': 'min_dist = {value}'
                },
            ),
            html.Div(
                dcc.Dropdown(
                    [{'label': f'metric = "{metric}"', 'value': metric} for metric in umap_metrics],
                    value=tote_umap_metric,
                    id='tote-umap-metric-selector',
                    clearable=False,
                ),
                style={
                    'margin': '0px 25px 25px 25px',
                },
            ),
            html.Div(
                html.Button(
                    'Calculate and plot UMAP Embedding',
                    id='tote-umap-button',
                    style={
                        'padding': '6px 12px',
                        'background': 'rgb(13, 110, 253)',
                        'border': '1px solid rgb(13, 110, 253)',
                        'color': 'rgb(255, 255, 255)',
                        'cursor': 'pointer',
                    },
                ),
                style={
                    'margin': '0px 25px 25px 25px',
                },
            ),
        ],
        style={
            'display': 'none' if readonly else 'block',
        }
    ),

    html.Div(
        id='tote-umap-output',
        children=tote_umap_output,
        style={
            'margin': '25px 0',
            'padding': '0 25px',
        }
    ),

    dcc.Graph(
        id='tote-umap-plotter',
        responsive='auto',
        clear_on_unhover=True,
        figure=tote_umap_fig,
        config={
            'toImageButtonOptions': {
                'filename': 'UMAP_Embedding_of_Topic_Term_Matrix'
            }
        },
    ),
    html.Div(
        id='tote-umap-hover',
        style={
            'display': 'none',
        },
    ),
    # endregion

    html.H2(children='Chapter 1.2. UMAP Embedding of Document-Topic Matrix'),

    # region Main Layout of UMAP Embedding of Document-Topic Matrix
    html.Div(
        children=[
            dcc.Store(id='doto-umap-params-store', storage_type='memory'),
            dcc.Store(id='doto-umap-embedding-store', storage_type='memory'),
            html.Div(style={
                'width': '100%',
                'height': '25px',
            }),
            dcc.Slider(
                umap_n_components_min, umap_n_components_max, umap_n_components_step,
                value=doto_umap_n_components,
                id='doto-umap-n-components-slider',
                marks=None,
                tooltip={
                    'always_visible': True,
                    'template': 'n_components = {value}'
                },
            ),
            html.Div(style={
                'width': '100%',
                'height': '25px',
            }),
            dcc.Slider(
                umap_n_neighbors_min, umap_n_neighbors_max, umap_n_neighbors_step,
                value=doto_umap_n_neighbors,
                id='doto-umap-n-neighbors-slider',
                marks=None,
                tooltip={
                    'always_visible': True,
                    'template': 'n_neighbors = {value}'
                },
            ),
            html.Div(style={
                'width': '100%',
                'height': '25px',
            }),
            dcc.Slider(
                umap_min_dist_min, umap_min_dist_max, umap_min_dist_step,
                value=doto_umap_min_dist,
                id='doto-umap-min-dist-slider',
                marks=None,
                tooltip={
                    'always_visible': True,
                    'template': 'min_dist = {value}'
                },
            ),
            html.Div(
                dcc.Dropdown(
                    [{'label': f'metric = "{metric}"', 'value': metric} for metric in umap_metrics],
                    value=doto_umap_metric,
                    id='doto-umap-metric-selector',
                    clearable=False,
                ),
                style={
                    'margin': '0px 25px 25px 25px',
                },
            ),
            html.Div(
                html.Button(
                    'Calculate and plot UMAP Embedding',
                    id='doto-umap-button',
                    style={
                        'padding': '6px 12px',
                        'background': 'rgb(13, 110, 253)',
                        'border': '1px solid rgb(13, 110, 253)',
                        'color': 'rgb(255, 255, 255)',
                        'cursor': 'pointer',
                    },
                ),
                style={
                    'margin': '0px 25px 25px 25px',
                },
            ),
        ],
        style={
            'display': 'none' if readonly else 'block',
        }
    ),

    html.Div(
        id='doto-umap-output',
        children=doto_umap_output,
        style={
            'margin': '25px 0',
            'padding': '0 25px',
        }
    ),

    dcc.Graph(
        id='doto-umap-plotter',
        responsive='auto',
        clear_on_unhover=True,
        figure=doto_umap_fig,
        config={
            'toImageButtonOptions': {
                'filename': 'UMAP_Embedding_of_Document_Topic_Matrix'
            }
        },
    ),
    html.Div(
        id='doto-umap-hover',
        style={
            'display': 'none',
        },
    ),
    # endregion

    html.H1(children='Chapter 2. Topic Clustering'),

    html.Div(
        children=[
            html.H2(children='Chapter 2.1. Using KMeans'),

            # region Main Layout of Topic Clustering using KMeans
            html.Div(
                children=[
                    html.H3(children='Chapter 2.1.1. KMeans Scoring'),

                    # region Main Layout of Scoring of Topic Clustering using KMeans
                    dcc.Store(id='tote-kmeans-scoring-params-store', storage_type='memory'),
                    html.Div(style={
                        'width': '100%',
                        'height': '25px',
                    }),
                    dcc.Slider(
                        kmeans_n_clusters_min, kmeans_n_clusters_max, kmeans_n_clusters_step,
                        value=scoring_kmeans_n_clusters,
                        id='tote-kmeans-scoring-n-clusters-slider',
                        marks=None,
                        tooltip={
                            'always_visible': True,
                            'template': 'n_clusters = {value}'
                        },
                    ),
                    html.Div(
                        html.Button(
                            'Calculate and plot KMeans Scoring',
                            id='tote-kmeans-scoring-button',
                            style={
                                'padding': '6px 12px',
                                'background': 'rgb(13, 110, 253)',
                                'border': '1px solid rgb(13, 110, 253)',
                                'color': 'rgb(255, 255, 255)',
                                'cursor': 'pointer',
                            },
                        ),
                        style={
                            'margin': '0px 25px 25px 25px',
                        },
                    ),
                    html.Div(
                        id='tote-kmeans-scoring-output',
                        style={
                            'margin': '25px 0',
                            'padding': '0 25px',
                        }
                    ),
                    dcc.Graph(
                        id='tote-kmeans-scoring-plotter',
                        responsive='auto',
                        clear_on_unhover=True,
                        config={
                            'toImageButtonOptions': {
                                'filename': 'Scoring_of_Clusters_using_KMeans_of_UMAP_Embedding_of_Topic_Term_Matrix'
                            }
                        },
                    ),
                    # endregion

                    html.H3(children='Chapter 2.1.2. KMeans Clustering'),

                    dcc.Store(id='tote-kmeans-params-store', storage_type='memory'),
                    html.Div(style={
                        'width': '100%',
                        'height': '25px',
                    }),
                    dcc.Slider(
                        kmeans_n_clusters_min, kmeans_n_clusters_max, kmeans_n_clusters_step,
                        value=kmeans_n_clusters,
                        id='tote-kmeans-n-clusters-slider',
                        marks=None,
                        tooltip={
                            'always_visible': True,
                            'template': 'n_clusters = {value}'
                        },
                    ),
                    html.Div(
                        html.Button(
                            'Calculate and plot Clusters by KMeans',
                            id='tote-kmeans-button',
                            style={
                                'padding': '6px 12px',
                                'background': 'rgb(13, 110, 253)',
                                'border': '1px solid rgb(13, 110, 253)',
                                'color': 'rgb(255, 255, 255)',
                                'cursor': 'pointer',
                            },
                        ),
                        style={
                            'margin': '0px 25px 25px 25px',
                        },
                    ),
                ],
                style={
                    'display': 'none' if readonly else 'block',
                }
            ),

            html.Div(
                id='tote-kmeans-output',
                children=tote_kmeans_output,
                style={
                    'margin': '25px 0',
                    'padding': '0 25px',
                }
            ),

            dcc.Graph(
                id='tote-kmeans-plotter',
                responsive='auto',
                clear_on_unhover=True,
                figure=tote_kmeans_fig,
                config={
                    'toImageButtonOptions': {
                        'filename': 'Clusters_using_KMeans_of_UMAP_Embedding_of_Topic_Term_Matrix'
                    }
                },
            ),
            html.Div(
                id='tote-kmeans-hover',
                style={
                    'display': 'none',
                },
            ),
            dcc.Download(id='tote-kmeans-download'),

            dcc.Graph(
                id='tote-doto-kmeans-plotter',
                responsive='auto',
                clear_on_unhover=True,
                figure=tote_doto_kmeans_fig,
                config={
                    'toImageButtonOptions': {
                        'filename': 'Clusters_using_KMeans_of_UMAP_Embedding_of_Document_Topic_Matrix'
                    }
                },
            ),
            html.Div(
                id='tote-doto-kmeans-hover',
                style={
                    'display': 'none',
                },
            ),
            dcc.Download(id='tote-doto-kmeans-download'),
            dcc.Download(id='tote-config-kmeans-download'),
            # endregion
        ],
        style={
            'display': 'block' if kmeans_enabled else 'none',
        }
    ),

    html.Div(
        children=[
            html.H2(children='Chapter 2.2. Using HDBSCAN' if kmeans_enabled else 'Chapter 2.1. Using HDBSCAN'),

            # region Main Layout of Topic Clustering using HDBSCAN
            html.Div(
                children=[
                    dcc.Store(id='tote-hdbscan-params-store', storage_type='memory'),
                    html.Div(style={
                        'width': '100%',
                        'height': '25px',
                    }),
                    dcc.Slider(
                        hdbscan_min_cluster_size_min, hdbscan_min_cluster_size_max, hdbscan_min_cluster_size_step,
                        value=hdbscan_min_cluster_size,
                        id='tote-hdbscan-min-cluster-size-slider',
                        marks=None,
                        tooltip={
                            'always_visible': True,
                            'template': 'min_cluster_size = {value}'
                        },
                    ),
                    html.Div(style={
                        'width': '100%',
                        'height': '25px',
                    }),
                    dcc.Slider(
                        hdbscan_min_samples_min, hdbscan_min_samples_max, hdbscan_min_samples_step,
                        value=hdbscan_min_samples,
                        id='tote-hdbscan-min-samples-slider',
                        marks=None,
                        tooltip={
                            'always_visible': True,
                            'template': 'min_samples = {value}'
                        },
                    ),
                    html.Div(
                        dcc.Dropdown(
                            [{'label': f'metric = "{metric}"', 'value': metric} for metric in hdbscan_metrics],
                            value=hdbscan_metric,
                            id='tote-hdbscan-metric-selector',
                            clearable=False,
                        ),
                        style={
                            'margin': '0px 25px 25px 25px',
                        },
                    ),
                    html.Div(
                        html.Button(
                            'Calculate and plot Clusters by HDBSCAN',
                            id='tote-hdbscan-button',
                            style={
                                'padding': '6px 12px',
                                'background': 'rgb(13, 110, 253)',
                                'border': '1px solid rgb(13, 110, 253)',
                                'color': 'rgb(255, 255, 255)',
                                'cursor': 'pointer',
                            },
                        ),
                        style={
                            'margin': '0px 25px 25px 25px',
                        },
                    ),
                ],
                style={
                    'display': 'none' if readonly else 'block',
                }
            ),

            html.Div(
                id='tote-hdbscan-output',
                children=tote_hdbscan_output,
                style={
                    'margin': '25px 0',
                    'padding': '0 25px',
                }
            ),

            dcc.Graph(
                id='tote-hdbscan-plotter',
                responsive='auto',
                clear_on_unhover=True,
                figure=tote_hdbscan_fig,
                config={
                    'toImageButtonOptions': {
                        'filename': 'Clusters_using_HDBSCAN_of_UMAP_Embedding_of_Topic_Term_Matrix'
                    }
                },
            ),
            html.Div(
                id='tote-hdbscan-hover',
                style={
                    'display': 'none',
                },
            ),
            dcc.Download(id='tote-hdbscan-download'),

            dcc.Graph(
                id='tote-doto-hdbscan-plotter',
                responsive='auto',
                clear_on_unhover=True,
                figure=tote_doto_hdbscan_fig,
                config={
                    'toImageButtonOptions': {
                        'filename': 'Clusters_using_HDBSCAN_of_UMAP_Embedding_of_Document_Topic_Matrix'
                    }
                },
            ),
            html.Div(
                id='tote-doto-hdbscan-hover',
                style={
                    'display': 'none',
                },
            ),
            dcc.Download(id='tote-doto-hdbscan-download'),
            dcc.Download(id='tote-config-hdbscan-download'),
            # endregion
        ],
        style={
            'display': 'block' if hdbscan_enabled else 'none',
        }
    ),
])


# endregion

# region Handlers of Import Data
@app.callback(
    Output('metadata-store', 'data'),
    Output('metadata-output', 'children'),
    Input('metadata-uploader', 'contents'),
    State('metadata-uploader', 'filename'),
    State('metadata-uploader', 'last_modified'),
    prevent_initial_call=True,
)
def update_metadata(file_content, file_name, file_last_modified):
    try:
        _, base64_content = file_content.split(',')
        metadata = json.loads(base64.b64decode(base64_content).decode('utf-8'))
        return (
            (
                metadata['docs'],
                metadata['terms'],
            ),
            html.P(children=[f'Metadata: ', html.Span('OK', style={'color': 'green'})])
        )
    except:
        if file_name is None:
            return (
                None,
                html.P(children=f'Must upload "umap_metadata.json" first!', style={'color': 'red'})
            )
        return (
            None,
            html.P(children=f'Cannot get metadata from {file_name}', style={'color': 'red'})
        )


@app.callback(
    Output('data-store', 'data'),
    Output('data-output', 'children'),
    Output('tote-kmeans-scoring-n-clusters-slider', 'max'),
    Output('tote-kmeans-scoring-n-clusters-slider', 'value'),
    Output('tote-kmeans-n-clusters-slider', 'max'),
    Output('tote-hdbscan-min-cluster-size-slider', 'max'),
    Output('tote-hdbscan-min-samples-slider', 'max'),
    Input('data-uploader', 'contents'),
    State('data-uploader', 'filename'),
    State('data-uploader', 'last_modified'),
    prevent_initial_call=True,
)
def update_data(file_content, file_name, file_last_modified):
    global kmeans_n_clusters_max, hdbscan_min_cluster_size_max, hdbscan_min_samples_max

    try:
        _, base64_content = file_content.split(',')
        data = json.loads(base64.b64decode(base64_content).decode('utf-8'))
        return (
            (
                data['num_topics'],
                data['terms_per_topic'],
                data['random_seed'],
                data['topic_matrix'],
                data['topic_term_matrix'],
                data['document_topic_matrix'],
            ),
            html.P(children=[f'Data: ', html.Span('OK', style={'color': 'green'})]),
            data['num_topics'] - 1,
            data['num_topics'] - 1,
            data['num_topics'] - 1,
            data['num_topics'],
            data['num_topics'],
        )
    except:
        if file_name is None:
            return (
                None,
                html.P(children=f'Must upload "umap_data.json" first!', style={'color': 'red'}),
                kmeans_n_clusters_max,
                kmeans_n_clusters_max,
                hdbscan_min_cluster_size_max,
                hdbscan_min_samples_max,
            )
        return (
            None,
            html.P(children=f'Cannot get data from {file_name}', style={'color': 'red'}),
            kmeans_n_clusters_max,
            kmeans_n_clusters_max,
            hdbscan_min_cluster_size_max,
            hdbscan_min_samples_max,
        )


# endregion

# region Handlers of UMAP Embedding of Topic-Term Matrix
@app.callback(
    Output('tote-umap-params-store', 'data'),
    Input('tote-umap-n-components-slider', 'value'),
    Input('tote-umap-n-neighbors-slider', 'value'),
    Input('tote-umap-min-dist-slider', 'value'),
    Input('tote-umap-metric-selector', 'value'),
)
def update_tote_umap_params(n_components_value, n_neighbors_value, min_dist_value, metric_value):
    return n_components_value, n_neighbors_value, min_dist_value, metric_value


@app.callback(
    Output('tote-umap-embedding-store', 'data'),
    Output('tote-umap-output', 'children'),
    Output('tote-umap-plotter', 'figure'),
    Input('tote-umap-button', 'n_clicks'),
    State('metadata-store', 'data'),
    State('data-store', 'data'),
    State('tote-umap-params-store', 'data'),
    prevent_initial_call=True,
)
def update_tote_umap_graph(n_clicks, stored_metadata, stored_data, stored_umap_params):
    return calculate_tote_umap_graph(stored_metadata, stored_data, stored_umap_params)


@app.callback(
    Output('tote-umap-hover', 'children'),
    Output('tote-umap-hover', 'style'),
    Input('tote-umap-plotter', 'hoverData'),
)
def update_tote_umap_hover(hover_data):
    return update_graph_hover(hover_data, lambda customdata: [
        f'Topic: {customdata[0]}',
        html.Br(),
        f'Keywords: {customdata[1]}',
    ])


# endregion

# region Handlers of UMAP Embedding of Document-Topic Matrix
@app.callback(
    Output('doto-umap-params-store', 'data'),
    Input('doto-umap-n-components-slider', 'value'),
    Input('doto-umap-n-neighbors-slider', 'value'),
    Input('doto-umap-min-dist-slider', 'value'),
    Input('doto-umap-metric-selector', 'value'),
)
def update_doto_umap_params(n_components_value, n_neighbors_value, min_dist_value, metric_value):
    return n_components_value, n_neighbors_value, min_dist_value, metric_value


@app.callback(
    Output('doto-umap-embedding-store', 'data'),
    Output('doto-umap-output', 'children'),
    Output('doto-umap-plotter', 'figure'),
    Input('doto-umap-button', 'n_clicks'),
    State('metadata-store', 'data'),
    State('data-store', 'data'),
    State('doto-umap-params-store', 'data'),
    prevent_initial_call=True,
)
def update_doto_umap_graph(n_clicks, stored_metadata, stored_data, stored_umap_params):
    return calculate_doto_umap_graph(stored_metadata, stored_data, stored_umap_params)


@app.callback(
    Output('doto-umap-hover', 'children'),
    Output('doto-umap-hover', 'style'),
    Input('doto-umap-plotter', 'hoverData'),
)
def update_doto_umap_hover(hover_data):
    return update_graph_hover(hover_data, lambda customdata: [
        f'Document: {customdata[0]}',
        html.Br(),
        f'Dominant Topic: {customdata[1]}',
        html.Br(),
        f'Dominant Keywords: {customdata[2]}',
    ])


# endregion

# region Handlers of Scoring of Topic Clustering using KMeans
@app.callback(
    Output('tote-kmeans-scoring-params-store', 'data'),
    Input('tote-kmeans-scoring-n-clusters-slider', 'value'),
)
def update_tote_kmeans_scoring_params(n_clusters_value):
    return n_clusters_value


@app.callback(
    Output('tote-kmeans-scoring-output', 'children'),
    Output('tote-kmeans-scoring-plotter', 'figure'),
    Input('tote-kmeans-scoring-button', 'n_clicks'),
    State('data-store', 'data'),
    State('tote-umap-embedding-store', 'data'),
    State('tote-kmeans-scoring-params-store', 'data'),
    prevent_initial_call=True,
)
def update_tote_kmeans_scoring_graph(
        n_clicks,
        stored_data,
        stored_umap_embedding,
        stored_kmeans_scoring_params
):
    global kmeans_n_clusters_min

    if stored_data is None:
        return (
            html.P(children=f'Data not found!', style={'color': 'red'}),
            None,
        )
    if stored_umap_embedding is None:
        return (
            html.P(children=f'UMAP Embedding of Topic-Term Matrix not found!', style={'color': 'red'}),
            None,
        )

    (
        _,
        _,
        random_seed,
        _,
        _,
        _,
    ) = stored_data

    embedding = np.array(stored_umap_embedding)

    n_clusters_value = stored_kmeans_scoring_params

    list_of_num_clusters = list(range(kmeans_n_clusters_min, n_clusters_value + 1))
    inertia_values = []
    silhouette_avg_values = []
    for num_clusters in list_of_num_clusters:
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=random_seed,
        )
        kmeans.fit(embedding)
        inertia_values.append(kmeans.inertia_)
        silhouette_avg_values.append(silhouette_score(embedding, kmeans.labels_))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('Elbow Method using Inertia', 'Silhouette Avg. Scores'),
    )
    fig.add_trace(
        go.Scatter(
            x=list_of_num_clusters,
            y=inertia_values,
            mode='lines+markers',
            marker=dict(size=8, color='blue'),
            line=dict(width=2),
            name='Inertia',
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list_of_num_clusters,
            y=silhouette_avg_values,
            mode='lines+markers',
            marker=dict(size=8, color='green'),
            line=dict(width=2),
            name='Silhouette Avg. Score',
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(row=1, col=1, title_text='n_clusters')
    fig.update_xaxes(row=1, col=2, title_text='n_clusters')
    fig.update_yaxes(row=1, col=1, title_text='Inertia')
    fig.update_yaxes(row=1, col=2, title_text='Silhouette Avg. Score')

    return (
        [
            f'- n-clusters: {kmeans_n_clusters_min} -> {n_clusters_value}',
        ],
        fig,
    )


# endregion

# region Handlers of Topic Clustering using KMeans
@app.callback(
    Output('tote-kmeans-params-store', 'data'),
    Input('tote-kmeans-n-clusters-slider', 'value'),
)
def update_tote_kmeans_params(n_clusters_value):
    return n_clusters_value


@app.callback(
    Output('tote-kmeans-output', 'children'),
    Output('tote-kmeans-plotter', 'figure'),
    Output('tote-kmeans-download', 'data'),
    Output('tote-doto-kmeans-plotter', 'figure'),
    Output('tote-doto-kmeans-download', 'data'),
    Output('tote-config-kmeans-download', 'data'),
    Input('tote-kmeans-button', 'n_clicks'),
    State('metadata-store', 'data'),
    State('data-store', 'data'),
    State('tote-umap-embedding-store', 'data'),
    State('doto-umap-embedding-store', 'data'),
    State('tote-umap-params-store', 'data'),
    State('doto-umap-params-store', 'data'),
    State('tote-kmeans-params-store', 'data'),
    prevent_initial_call=True,
)
def update_tote_kmeans_graph(
        n_clicks,
        stored_metadata,
        stored_data,
        stored_tote_umap_embedding,
        stored_doto_umap_embedding,
        stored_tote_umap_params,
        stored_doto_umap_params,
        stored_kmeans_params,
):
    global clustering_method_kmeans

    return calculate_tote_clustering_graph(
        stored_metadata,
        stored_data,
        stored_tote_umap_embedding,
        stored_doto_umap_embedding,
        stored_tote_umap_params,
        stored_doto_umap_params,
        stored_kmeans_params,
        clustering_method_kmeans,
        calculate_kmeans_clusters,
    )


@app.callback(
    Output('tote-kmeans-hover', 'children'),
    Output('tote-kmeans-hover', 'style'),
    Input('tote-kmeans-plotter', 'hoverData'),
)
def update_tote_kmeans_hover(hover_data):
    return update_tote_clustering_hover(hover_data)


@app.callback(
    Output('tote-doto-kmeans-hover', 'children'),
    Output('tote-doto-kmeans-hover', 'style'),
    Input('tote-doto-kmeans-plotter', 'hoverData'),
)
def update_tote_doto_kmeans_hover(hover_data):
    return update_tote_doto_clustering_hover(hover_data)


# endregion

# region Handlers of Topic Clustering using HDBSCAN
@app.callback(
    Output('tote-hdbscan-params-store', 'data'),
    Input('tote-hdbscan-min-cluster-size-slider', 'value'),
    Input('tote-hdbscan-min-samples-slider', 'value'),
    Input('tote-hdbscan-metric-selector', 'value'),
)
def update_tote_hdbscan_params(min_cluster_size_value, min_samples_value, metric_value):
    return min_cluster_size_value, min_samples_value, metric_value


@app.callback(
    Output('tote-hdbscan-output', 'children'),
    Output('tote-hdbscan-plotter', 'figure'),
    Output('tote-hdbscan-download', 'data'),
    Output('tote-doto-hdbscan-plotter', 'figure'),
    Output('tote-doto-hdbscan-download', 'data'),
    Output('tote-config-hdbscan-download', 'data'),
    Input('tote-hdbscan-button', 'n_clicks'),
    State('metadata-store', 'data'),
    State('data-store', 'data'),
    State('tote-umap-embedding-store', 'data'),
    State('doto-umap-embedding-store', 'data'),
    State('tote-umap-params-store', 'data'),
    State('doto-umap-params-store', 'data'),
    State('tote-hdbscan-params-store', 'data'),
    prevent_initial_call=True,
)
def update_tote_hdbscan_graph(
        n_clicks,
        stored_metadata,
        stored_data,
        stored_tote_umap_embedding,
        stored_doto_umap_embedding,
        stored_tote_umap_params,
        stored_doto_umap_params,
        stored_hdbscan_params,
):
    global clustering_method_hdbscan

    return calculate_tote_clustering_graph(
        stored_metadata,
        stored_data,
        stored_tote_umap_embedding,
        stored_doto_umap_embedding,
        stored_tote_umap_params,
        stored_doto_umap_params,
        stored_hdbscan_params,
        clustering_method_hdbscan,
        calculate_hdbscan_clusters,
    )


@app.callback(
    Output('tote-hdbscan-hover', 'children'),
    Output('tote-hdbscan-hover', 'style'),
    Input('tote-hdbscan-plotter', 'hoverData'),
)
def update_tote_hdbscan_hover(hover_data):
    return update_tote_clustering_hover(hover_data)


@app.callback(
    Output('tote-doto-hdbscan-hover', 'children'),
    Output('tote-doto-hdbscan-hover', 'style'),
    Input('tote-doto-hdbscan-plotter', 'hoverData'),
)
def update_tote_doto_hdbscan_hover(hover_data):
    return update_tote_doto_clustering_hover(hover_data)


# endregion

if __name__ == '__main__':
    app.run(
        debug=debug,
        port=port,
    )
