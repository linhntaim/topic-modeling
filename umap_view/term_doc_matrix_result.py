import argparse
import base64
import json

import hdbscan
import numpy as np
import pandas

from dash import Dash, html, dcc, Output, Input, State
from gensim import matutils
from gensim.models import TfidfModel
from plotly import express as px, graph_objs as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from base.ab_fixed_umap import ABFixedUMAP

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', help='Set port', default='8050')
parser.add_argument('--debug', action='store_true', help='Enable debug')
args = parser.parse_args()

port = int(args.port)
debug = args.debug

discrete_colors = [
    *px.colors.qualitative.Light24,
    *px.colors.qualitative.Dark24,
    *px.colors.qualitative.Set3,
]

# region UMAP params
umap_n_components_min = 1
umap_n_components_max = 3
umap_n_components_step = 1
umap_n_components = 2

umap_n_neighbors_min = 2
umap_n_neighbors_max = 200
umap_n_neighbors_step = 1
umap_n_neighbors = 5

umap_min_dist_min = 0.00
umap_min_dist_max = 1.00
umap_min_dist_step = 0.01
umap_min_dist = 0.1

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
umap_metric = 'hellinger'
# endregion

# region KMeans params
kmeans_n_clusters_min = 2
kmeans_n_clusters_max = 20
kmeans_n_clusters_step = 1
kmeans_n_clusters = 2
# endregion

# region HDBSCAN params
hdbscan_min_cluster_size_min = 2
hdbscan_min_cluster_size_max = 20
hdbscan_min_cluster_size_step = 1
hdbscan_min_cluster_size = 2

hdbscan_min_samples_min = 1
hdbscan_min_samples_max = 20
hdbscan_min_samples_step = 1
hdbscan_min_samples = 1

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
hdbscan_metric = 'euclidean'
# endregion

app = Dash(
    __name__,
    title='UMAP Embedding of Term-Document Matrix',
)
app.layout = html.Div([

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

    html.H2(children='Chapter 1. UMAP Embedding of Term-Document Matrix weighted by TF-IDF'),

    # region Main Layout of UMAP Embedding of Term-Document Matrix weighted by TF-IDF
    dcc.Store(id='tfidf-tedo-umap-params-store', storage_type='memory'),
    dcc.Store(id='tfidf-tedo-umap-embedding-store', storage_type='memory'),
    html.Div(style={
        'width': '100%',
        'height': '25px',
    }),
    dcc.Slider(
        umap_n_components_min, umap_n_components_max, umap_n_components_step,
        value=umap_n_components,
        id='tfidf-tedo-umap-n-components-slider',
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
        value=umap_n_neighbors,
        id='tfidf-tedo-umap-n-neighbors-slider',
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
        value=umap_min_dist,
        id='tfidf-tedo-umap-min-dist-slider',
        marks=None,
        tooltip={
            'always_visible': True,
            'template': 'min_dist = {value}'
        },
    ),
    html.Div(
        dcc.Dropdown(
            [{'label': f'metric = "{metric}"', 'value': metric} for metric in umap_metrics],
            value=umap_metric,
            id='tfidf-tedo-umap-metric-selector',
            clearable=False,
        ),
        style={
            'margin': '0px 25px 25px 25px',
        },
    ),
    html.Div(
        html.Button(
            'Calculate and plot UMAP Embedding',
            id='tfidf-tedo-umap-button',
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
        id='tfidf-tedo-umap-output',
        style={
            'margin': '25px 0',
            'padding': '0 25px',
        }
    ),
    dcc.Graph(
        id='tfidf-tedo-umap-plotter',
        responsive='auto',
        clear_on_unhover=True,
    ),
    html.Div(
        id='tfidf-tedo-umap-hover',
        style={
            'display': 'none',
        },
    ),
    # endregion

    html.H1(children='Chapter 2. Topic Clustering'),

    html.H2(children='Chapter 2.1. Using KMeans'),

    html.H3(children='Chapter 2.1.1. KMeans Scoring'),

    # region Main Layout of Scoring of Topic Clustering using KMeans
    dcc.Store(id='tfidf-tedo-kmeans-scoring-params-store', storage_type='memory'),
    html.Div(style={
        'width': '100%',
        'height': '25px',
    }),
    dcc.Slider(
        kmeans_n_clusters_min, kmeans_n_clusters_max, kmeans_n_clusters_step,
        value=kmeans_n_clusters,
        id='tfidf-tedo-kmeans-scoring-n-clusters-slider',
        marks=None,
        tooltip={
            'always_visible': True,
            'template': 'n_clusters = {value}'
        },
    ),
    html.Div(
        html.Button(
            'Calculate and plot KMeans Scoring',
            id='tfidf-tedo-kmeans-scoring-button',
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
        id='tfidf-tedo-kmeans-scoring-output',
        style={
            'margin': '25px 0',
            'padding': '0 25px',
        }
    ),
    dcc.Graph(
        id='tfidf-tedo-kmeans-scoring-plotter',
        responsive='auto',
        clear_on_unhover=True,
    ),
    # endregion

    html.H3(children='Chapter 2.1.2. KMeans Clustering'),

    # region Main Layout of Topic Clustering using KMeans
    dcc.Store(id='tfidf-tedo-kmeans-params-store', storage_type='memory'),
    html.Div(style={
        'width': '100%',
        'height': '25px',
    }),
    dcc.Slider(
        kmeans_n_clusters_min, kmeans_n_clusters_max, kmeans_n_clusters_step,
        value=kmeans_n_clusters,
        id='tfidf-tedo-kmeans-n-clusters-slider',
        marks=None,
        tooltip={
            'always_visible': True,
            'template': 'n_clusters = {value}'
        },
    ),
    html.Div(
        html.Button(
            'Calculate and plot Clusters by KMeans',
            id='tfidf-tedo-kmeans-button',
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
        id='tfidf-tedo-kmeans-output',
        style={
            'margin': '25px 0',
            'padding': '0 25px',
        }
    ),

    dcc.Graph(
        id='tfidf-tedo-kmeans-plotter',
        responsive='auto',
        clear_on_unhover=True,
    ),
    dcc.Download(id='tfidf-tedo-kmeans-download'),
    # html.Div(
    #     id='tfidf-tedo-kmeans-table',
    #     style={
    #         'margin': '25px 0',
    #         'padding': '0 25px',
    #     }
    # ),
    html.Div(
        id='tfidf-tedo-kmeans-hover',
        style={
            'display': 'none',
        },
    ),

    dcc.Graph(
        id='tfidf-tedo-doto-kmeans-plotter',
        responsive='auto',
        clear_on_unhover=True,
    ),
    dcc.Download(id='tfidf-tedo-doto-kmeans-download'),
    # html.Div(
    #     id='tfidf-tedo-doto-kmeans-table',
    #     style={
    #         'margin': '25px 0',
    #         'padding': '0 25px',
    #     }
    # ),
    html.Div(
        id='tfidf-tedo-doto-kmeans-hover',
        style={
            'display': 'none',
        },
    ),
    # endregion

    html.H2(children='Chapter 2.2. Using HDBSCAN'),

    # region Main Layout of Topic Clustering using HDBSCAN
    dcc.Store(id='tfidf-tedo-hdbscan-params-store', storage_type='memory'),
    html.Div(style={
        'width': '100%',
        'height': '25px',
    }),
    dcc.Slider(
        hdbscan_min_cluster_size_min, hdbscan_min_cluster_size_max, hdbscan_min_cluster_size_step,
        value=hdbscan_min_cluster_size,
        id='tfidf-tedo-hdbscan-min-cluster-size-slider',
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
        id='tfidf-tedo-hdbscan-min-samples-slider',
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
            id='tfidf-tedo-hdbscan-metric-selector',
            clearable=False,
        ),
        style={
            'margin': '0px 25px 25px 25px',
        },
    ),
    html.Div(
        html.Button(
            'Calculate and plot Clusters by HDBSCAN',
            id='tfidf-tedo-hdbscan-button',
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
        id='tfidf-tedo-hdbscan-output',
        style={
            'margin': '25px 0',
            'padding': '0 25px',
        }
    ),

    dcc.Graph(
        id='tfidf-tedo-hdbscan-plotter',
        responsive='auto',
        clear_on_unhover=True,
    ),
    dcc.Download(id='tfidf-tedo-hdbscan-download'),
    # html.Div(
    #     id='tfidf-tedo-hdbscan-table',
    #     style={
    #         'margin': '25px 0',
    #         'padding': '0 25px',
    #     }
    # ),
    html.Div(
        id='tfidf-tedo-hdbscan-hover',
        style={
            'display': 'none',
        },
    ),

    dcc.Graph(
        id='tfidf-tedo-doto-hdbscan-plotter',
        responsive='auto',
        clear_on_unhover=True,
    ),
    dcc.Download(id='tfidf-tedo-doto-hdbscan-download'),
    # html.Div(
    #     id='tfidf-tedo-doto-hdbscan-table',
    #     style={
    #         'margin': '25px 0',
    #         'padding': '0 25px',
    #     }
    # ),
    html.Div(
        id='tfidf-tedo-doto-hdbscan-hover',
        style={
            'display': 'none',
        },
    ),
    # endregion

])


# region Common Handlers
def convert_to_tfidf_doc_term_matrix(meta_doc_term_matrix, num_docs, num_terms):
    corpus = [[(int(term_id), term_count) for term_id, term_count in doc.items()] for doc in meta_doc_term_matrix]
    tfidf_model = TfidfModel(corpus)
    tfidf_corpus = [{term_id: term_freq for term_id, term_freq in tfidf_doc} for tfidf_doc in tfidf_model[corpus]]
    tfidf_doc_term_matrix = []
    for doc_id in range(num_docs):
        doc_terms = []
        tfidf_doc = tfidf_corpus[doc_id]
        for term_id in range(num_terms):
            doc_terms.append(tfidf_doc[term_id] if term_id in tfidf_doc else 0.0)
        tfidf_doc_term_matrix.append(doc_terms)
    return tfidf_doc_term_matrix


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

    if figure_data is None:
        figure_data = {}
        figure_hover_data = {}
        figure_labels = {}

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


def update_tfidf_tedo_clustering_graph(
        stored_metadata,
        stored_data,
        stored_umap_embedding,
        stored_umap_params,
        clustering_method_params,
        clustering_method_name,
        clustering_method,
):
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
    if stored_umap_embedding is None:
        return (
            html.P(children=f'UMAP Embedding of Topic-Term Matrix not found!', style={'color': 'red'}),
            None,
            None,
            None,
            None,
        )

    meta_docs, meta_terms, meta_tfidf_doc_term_matrix = stored_metadata
    num_docs = len(meta_docs)
    num_terms = len(meta_terms)
    tfidf_term_doc_matrix = np.array(meta_tfidf_doc_term_matrix).T.tolist()

    (
        _,
        _,
        random_seed,
        _,
        _,
        _,
    ) = stored_data

    embedding = np.array(stored_umap_embedding)

    (
        n_components_value,
        _,
        _,
        _,
    ) = stored_umap_params

    labels, labels_output = clustering_method(embedding, clustering_method_params, random_seed)

    fig = create_figure_from_umap_embbeding(
        n_components_value,
        embedding,
        {
            'term': meta_terms,
            'term_label': [f'Label {labels[term_id]}' for term_id in range(num_terms)],
        },
        {
            'term': True,
            'term_label': True,
        },
        {
            'term': 'Term',
            'term_label': 'Label',
        },
        'term_label',
        f'Clusters using {clustering_method_name} of UMAP Embedding of Term-Document Matrix weighted by TF-IDF',
    )

    download = dcc.send_data_frame(
        pandas.DataFrame({
                             'Term': meta_terms,
                             'Label': labels,
                         } | {meta_docs[doc_id]: doc for doc_id, doc in enumerate(meta_tfidf_doc_term_matrix)}).to_csv,
        f'term_labels_using_{clustering_method_name}.csv',
    )

    output = [
        f'Documents: {num_docs} / Terms: {num_terms}',
        html.Br(),
    ]
    output.extend(labels_output)
    return (
        output,
        fig,
        download,
    )


def update_tfidf_tedo_clustering_hover(hover_data):
    return update_graph_hover(hover_data, lambda customdata: [
        f'Term: {customdata[0]}',
    ])


# endregion

# region Handlers of Import Data
@app.callback(
    Output('metadata-store', 'data'),
    Output('metadata-output', 'children'),
    Output('tfidf-tedo-kmeans-scoring-n-clusters-slider', 'max'),
    Output('tfidf-tedo-kmeans-scoring-n-clusters-slider', 'value'),
    Output('tfidf-tedo-kmeans-n-clusters-slider', 'max'),
    Output('tfidf-tedo-hdbscan-min-cluster-size-slider', 'max'),
    Output('tfidf-tedo-hdbscan-min-samples-slider', 'max'),
    Input('metadata-uploader', 'contents'),
    State('metadata-uploader', 'filename'),
    State('metadata-uploader', 'last_modified'),
    prevent_initial_call=True,
)
def update_metadata(file_content, file_name, file_last_modified):
    try:
        _, base64_content = file_content.split(',')
        metadata = json.loads(base64.b64decode(base64_content).decode('utf-8'))
        num_docs = len(metadata['docs'])
        num_terms = len(metadata['terms'])
        return (
            (
                metadata['docs'],
                metadata['terms'],
                convert_to_tfidf_doc_term_matrix(metadata['doc_term_matrix'], num_docs, num_terms),
            ),
            html.P(children=[f'Metadata: ', html.Span('OK', style={'color': 'green'})]),
            num_terms - 1,
            num_terms - 1,
            num_terms - 1,
            num_terms,
            num_terms,
        )
    except:
        if file_name is None:
            return (
                None,
                html.P(children=f'Must upload "umap_metadata.json" first!', style={'color': 'red'}),
                kmeans_n_clusters_max,
                kmeans_n_clusters_max,
                hdbscan_min_cluster_size_max,
                hdbscan_min_samples_max,
            )
        return (
            None,
            html.P(children=f'Cannot get metadata from {file_name}', style={'color': 'red'}),
            kmeans_n_clusters_max,
            kmeans_n_clusters_max,
            hdbscan_min_cluster_size_max,
            hdbscan_min_samples_max,
        )


@app.callback(
    Output('data-store', 'data'),
    Output('data-output', 'children'),
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
        )
    except:
        if file_name is None:
            return (
                None,
                html.P(children=f'Must upload "umap_data.json" first!', style={'color': 'red'}),
            )
        return (
            None,
            html.P(children=f'Cannot get data from {file_name}', style={'color': 'red'}),
        )


# endregion

# region Handlers of UMAP Embedding of Term-Document Matrix weighted by TF-IDF
@app.callback(
    Output('tfidf-tedo-umap-params-store', 'data'),
    Input('tfidf-tedo-umap-n-components-slider', 'value'),
    Input('tfidf-tedo-umap-n-neighbors-slider', 'value'),
    Input('tfidf-tedo-umap-min-dist-slider', 'value'),
    Input('tfidf-tedo-umap-metric-selector', 'value'),
)
def update_tfidf_tedo_umap_params(n_components_value, n_neighbors_value, min_dist_value, metric_value):
    return n_components_value, n_neighbors_value, min_dist_value, metric_value


@app.callback(
    Output('tfidf-tedo-umap-embedding-store', 'data'),
    Output('tfidf-tedo-umap-output', 'children'),
    Output('tfidf-tedo-umap-plotter', 'figure'),
    Input('tfidf-tedo-umap-button', 'n_clicks'),
    State('metadata-store', 'data'),
    State('data-store', 'data'),
    State('tfidf-tedo-umap-params-store', 'data'),
    prevent_initial_call=True,
)
def update_tfidf_tedo_umap_graph(n_clicks, stored_metadata, stored_data, stored_umap_params):
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

    meta_docs, meta_terms, meta_tfidf_doc_term_matrix = stored_metadata
    num_docs = len(meta_docs)
    num_terms = len(meta_terms)
    tfidf_term_doc_matrix = np.array(meta_tfidf_doc_term_matrix).T.tolist()

    (
        _,
        _,
        random_seed,
        _,
        _,
        _,
    ) = stored_data

    (
        n_components_value,
        n_neighbors_value,
        min_dist_value,
        metric_value,
    ) = stored_umap_params

    embedding = calculate_umap_embedding(
        tfidf_term_doc_matrix,
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
            'term': meta_terms,
        },
        {
            'term': True,
        },
        {
            'term': 'Term',
        },
        None,
        'UMAP Embedding of Term-Document Matrix weighted by TF-IDF',
    )

    return (
        embedding,
        [
            f'Documents: {num_docs} / Terms: {num_terms}',
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


@app.callback(
    Output('tfidf-tedo-umap-hover', 'children'),
    Output('tfidf-tedo-umap-hover', 'style'),
    Input('tfidf-tedo-umap-plotter', 'hoverData'),
)
def update_tfidf_tedo_umap_hover(hover_data):
    return update_graph_hover(hover_data, lambda customdata: [
        f'Term: {customdata[0]}',
    ])


# endregion

# region Handlers of Scoring of Topic Clustering using KMeans
@app.callback(
    Output('tfidf-tedo-kmeans-scoring-params-store', 'data'),
    Input('tfidf-tedo-kmeans-scoring-n-clusters-slider', 'value'),
)
def update_tfidf_tedo_kmeans_scoring_params(n_clusters_value):
    return n_clusters_value


@app.callback(
    Output('tfidf-tedo-kmeans-scoring-output', 'children'),
    Output('tfidf-tedo-kmeans-scoring-plotter', 'figure'),
    Input('tfidf-tedo-kmeans-scoring-button', 'n_clicks'),
    State('data-store', 'data'),
    State('tfidf-tedo-umap-embedding-store', 'data'),
    State('tfidf-tedo-kmeans-scoring-params-store', 'data'),
    prevent_initial_call=True,
)
def update_tfidf_tedo_kmeans_scoring_graph(
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
    Output('tfidf-tedo-kmeans-params-store', 'data'),
    Input('tfidf-tedo-kmeans-n-clusters-slider', 'value'),
)
def update_tfidf_tedo_kmeans_params(n_clusters_value):
    return n_clusters_value


@app.callback(
    Output('tfidf-tedo-kmeans-output', 'children'),
    Output('tfidf-tedo-kmeans-plotter', 'figure'),
    Output('tfidf-tedo-kmeans-download', 'data'),
    Input('tfidf-tedo-kmeans-button', 'n_clicks'),
    State('metadata-store', 'data'),
    State('data-store', 'data'),
    State('tfidf-tedo-umap-embedding-store', 'data'),
    State('tfidf-tedo-umap-params-store', 'data'),
    State('tfidf-tedo-kmeans-params-store', 'data'),
    prevent_initial_call=True,
)
def update_tfidf_tedo_kmeans_graph(
        n_clicks,
        stored_metadata,
        stored_data,
        stored_umap_embedding,
        stored_umap_params,
        stored_kmeans_params,
):
    def calculate_kmeans_clusters(embedding, method_params, random_state):
        n_clusters_value = method_params

        kmeans = KMeans(
            n_clusters=n_clusters_value,
            random_state=random_state,
        )
        labels = kmeans.fit_predict(embedding)
        return (
            labels,
            [
                'KMeans parameters:',
                html.Br(),
                f'- n_clusters: {n_clusters_value}',
            ],
        )

    return update_tfidf_tedo_clustering_graph(
        stored_metadata,
        stored_data,
        stored_umap_embedding,
        stored_umap_params,
        stored_kmeans_params,
        'KMeans',
        calculate_kmeans_clusters,
    )


@app.callback(
    Output('tfidf-tedo-kmeans-hover', 'children'),
    Output('tfidf-tedo-kmeans-hover', 'style'),
    Input('tfidf-tedo-kmeans-plotter', 'hoverData'),
)
def update_tfidf_tedo_kmeans_hover(hover_data):
    return update_tfidf_tedo_clustering_hover(hover_data)


# endregion

# region Handlers of Topic Clustering using HDBSCAN
@app.callback(
    Output('tfidf-tedo-hdbscan-params-store', 'data'),
    Input('tfidf-tedo-hdbscan-min-cluster-size-slider', 'value'),
    Input('tfidf-tedo-hdbscan-min-samples-slider', 'value'),
    Input('tfidf-tedo-hdbscan-metric-selector', 'value'),
)
def update_tfidf_tedo_hdbscan_params(min_cluster_size_value, min_samples_value, metric_value):
    return min_cluster_size_value, min_samples_value, metric_value


@app.callback(
    Output('tfidf-tedo-hdbscan-output', 'children'),
    Output('tfidf-tedo-hdbscan-plotter', 'figure'),
    Output('tfidf-tedo-hdbscan-download', 'data'),
    Input('tfidf-tedo-hdbscan-button', 'n_clicks'),
    State('metadata-store', 'data'),
    State('data-store', 'data'),
    State('tfidf-tedo-umap-embedding-store', 'data'),
    State('tfidf-tedo-umap-params-store', 'data'),
    State('tfidf-tedo-hdbscan-params-store', 'data'),
    prevent_initial_call=True,
)
def update_tfidf_tedo_hdbscan_graph(
        n_clicks,
        stored_metadata,
        stored_data,
        stored_umap_embedding,
        stored_umap_params,
        stored_hdbscan_params,
):
    def calculate_hdbscan_clusters(embedding, method_params, random_state):
        min_cluster_size_value, min_samples_value, metric_value = method_params

        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size_value,
            min_samples=min_samples_value,
            metric=metric_value,
        ).fit_predict(embedding)

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
        )

    return update_tfidf_tedo_clustering_graph(
        stored_metadata,
        stored_data,
        stored_umap_embedding,
        stored_umap_params,
        stored_hdbscan_params,
        'HDBSCAN',
        calculate_hdbscan_clusters,
    )


@app.callback(
    Output('tfidf-tedo-hdbscan-hover', 'children'),
    Output('tfidf-tedo-hdbscan-hover', 'style'),
    Input('tfidf-tedo-hdbscan-plotter', 'hoverData'),
)
def update_tfidf_tedo_hdbscan_hover(hover_data):
    return update_tfidf_tedo_clustering_hover(hover_data)


# endregion

if __name__ == '__main__':
    app.run(
        debug=debug,
        port=port,
    )
