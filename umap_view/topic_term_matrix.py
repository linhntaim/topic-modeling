import argparse
import base64
import json

import hdbscan
import numpy as np
import pandas

from dash import Dash, html, dcc, Output, Input, State
from gensim import matutils
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

# UMAP params
n_components = 2
n_components_min = 1
n_components_max = 3
n_components_step = 1

n_neighbors = 5  # 15
n_neighbors_min = 2
n_neighbors_max = 200
n_neighbors_step = 1

min_dist = 0.3  # 0.1
min_dist_min = 0.00
min_dist_max = 1.00
min_dist_step = 0.01

metric = 'hellinger'
metrics = [
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

# KMeans params
n_clusters = 2
n_clusters_min = 2
n_clusters_max = 20
n_clusters_step = 1

# HDBSCAN params
min_cluster_size = 2
min_cluster_size_min = 2
min_cluster_size_max = 20
min_cluster_size_step = 1

min_samples = 1
min_samples_min = 1
min_samples_max = 20
min_samples_step = 1

hdbscan_metric = 'euclidean'
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

# Dash App
title = 'UMAP of LDA Topic-Term Matrix'

app = Dash(
    __name__,
    title=title,
)

app.layout = html.Div([
    dcc.Store(id='metadata-store', storage_type='memory'),
    dcc.Store(id='data-store', storage_type='memory'),
    dcc.Store(id='umap-params-store', storage_type='memory'),
    dcc.Store(id='umap-embedding-store', storage_type='memory'),
    dcc.Store(id='kmeans-scoring-params-store', storage_type='memory'),
    dcc.Store(id='kmeans-params-store', storage_type='memory'),
    dcc.Store(id='hdbscan-params-store', storage_type='memory'),
    html.H1(children='Chapter 0. Import LDA Result'),
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
    html.H1(children='Chapter 1. UMAP'),
    html.Div(style={
        'width': '100%',
        'height': '25px',
    }),
    dcc.Slider(
        n_components_min, n_components_max, n_components_step,
        value=n_components,
        id='n-components-slider',
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
        n_neighbors_min, n_neighbors_max, n_neighbors_step,
        value=n_neighbors,
        id='n-neighbors-slider',
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
        min_dist_min, min_dist_max, min_dist_step,
        value=min_dist,
        id='min-dist-slider',
        marks=None,
        tooltip={
            'always_visible': True,
            'template': 'min_dist = {value}'
        },
    ),
    html.Div(
        dcc.Dropdown(
            [{'label': f'metric = "{metric}"', 'value': metric} for metric in metrics],
            value=metric,
            id='metric-selector',
            clearable=False,
        ),
        style={
            'margin': '0px 25px 25px 25px',
        },
    ),
    html.Div(
        html.Button(
            'Calculate and plot UMAP Embedding',
            id='umap-button',
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
        id='umap-output',
        style={
            'margin': '25px 0',
            'padding': '0 25px',
        }
    ),
    dcc.Graph(
        id='umap-plotter',
        responsive='auto',
        clear_on_unhover=True,
    ),
    html.Div(
        id='umap-hover-output',
        style={
            'display': 'none',
        },
    ),
    html.H1(children='Chapter 2. Clustering'),
    html.H2(children='Chapter 2.1. Using KMeans'),
    html.H3(children='Chapter 2.1.1. KMeans Scoring'),
    html.Div(style={
        'width': '100%',
        'height': '25px',
    }),
    dcc.Slider(
        n_clusters_min, n_clusters_max, n_clusters_step,
        value=n_clusters,
        id='n-clusters-slider1',
        marks=None,
        tooltip={
            'always_visible': True,
            'template': 'n_clusters = {value}'
        },
    ),
    html.Div(
        html.Button(
            'Calculate and plot KMeans Scoring',
            id='kmeans-scoring-button',
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
        id='kmeans-scoring-output',
        style={
            'margin': '25px 0',
            'padding': '0 25px',
        }
    ),
    dcc.Graph(
        id='kmeans-scoring-plotter',
        responsive='auto',
        clear_on_unhover=True,
    ),
    html.H3(children='Chapter 2.1.2. Chosen KMeans'),
    html.Div(style={
        'width': '100%',
        'height': '25px',
    }),
    dcc.Slider(
        n_clusters_min, n_clusters_max, n_clusters_step,
        value=n_clusters,
        id='n-clusters-slider',
        marks=None,
        tooltip={
            'always_visible': True,
            'template': 'n_clusters = {value}'
        },
    ),
    html.Div(
        html.Button(
            'Calculate and plot Clusters by KMeans',
            id='kmeans-button',
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
        id='kmeans-output',
        style={
            'margin': '25px 0',
            'padding': '0 25px',
        }
    ),
    dcc.Graph(
        id='kmeans-plotter',
        responsive='auto',
        clear_on_unhover=True,
    ),
    html.Div(
        id='kmeans-table',
        style={
            'margin': '25px 0',
            'padding': '0 25px',
        }
    ),
    html.Div(
        id='kmeans-hover-output',
        style={
            'display': 'none',
        },
    ),
    html.H2(children='Chapter 2.2. Using HDBSCAN'),
    html.Div(style={
        'width': '100%',
        'height': '25px',
    }),
    dcc.Slider(
        min_cluster_size_min, min_cluster_size_max, min_cluster_size_step,
        value=min_cluster_size,
        id='min-cluster-size-slider',
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
        min_samples_min, min_samples_max, min_samples_step,
        value=min_samples,
        id='min-samples-slider',
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
            id='hdbscan-metric-selector',
            clearable=False,
        ),
        style={
            'margin': '0px 25px 25px 25px',
        },
    ),
    html.Div(
        html.Button(
            'Calculate and plot Clusters by HDBSCAN',
            id='hdbscan-button',
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
        id='hdbscan-output',
        style={
            'margin': '25px 0',
            'padding': '0 25px',
        }
    ),
    dcc.Graph(
        id='hdbscan-plotter',
        responsive='auto',
        clear_on_unhover=True,
    ),
    html.Div(
        id='hdbscan-table',
        style={
            'margin': '25px 0',
            'padding': '0 25px',
        }
    ),
    html.Div(
        id='hdbscan-hover-output',
        style={
            'display': 'none',
        },
    ),
])


@app.callback(
    Output('metadata-store', 'data'),
    Output('metadata-output', 'children'),
    Input('metadata-uploader', 'contents'),
    State('metadata-uploader', 'filename'),
    State('metadata-uploader', 'last_modified'),
    prevent_initial_call=True
)
def update_metadata(file_content, file_name, file_last_modified):
    try:
        _, base64_content = file_content.split(',')
        metadata = json.loads(base64.b64decode(base64_content).decode('utf-8'))
        return (
            metadata['docs'],
            metadata['terms'],
        ), html.P(children=[f'Metadata: ', html.Span('OK', style={'color': 'green'})])
    except:
        if file_name is None:
            return None, html.P(children=f'Must upload "umap_metadata.json" first!', style={'color': 'red'})
        return None, html.P(children=f'Cannot get metadata from {file_name}', style={'color': 'red'})


@app.callback(
    Output('data-store', 'data'),
    Output('data-output', 'children'),
    Output('n-clusters-slider1', 'max'),
    Output('n-clusters-slider', 'max'),
    Output('min-cluster-size-slider', 'max'),
    Output('min-samples-slider', 'max'),
    Input('data-uploader', 'contents'),
    State('data-uploader', 'filename'),
    State('data-uploader', 'last_modified'),
    prevent_initial_call=True
)
def update_data(file_content, file_name, file_last_modified):
    global n_clusters_max, min_cluster_size_max, min_samples_max

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
            data['num_topics'],
            data['num_topics'],
            data['num_topics'],
            data['num_topics'],
        )
    except:
        if file_name is None:
            return (
                None,
                html.P(children=f'Must upload "umap_data.json" first!', style={'color': 'red'}),
                n_clusters_max,
                n_clusters_max,
                min_cluster_size_max,
                min_samples_max,
            )
        return (
            None,
            html.P(children=f'Cannot get data from {file_name}', style={'color': 'red'}),
            n_clusters_max,
            n_clusters_max,
            min_cluster_size_max,
            min_samples_max,
        )


@app.callback(
    Output('umap-params-store', 'data'),
    Input('n-components-slider', 'value'),
    Input('n-neighbors-slider', 'value'),
    Input('min-dist-slider', 'value'),
    Input('metric-selector', 'value'),
)
def update_umap_params(n_components_value, n_neighbors_value, min_dist_value, metric_value):
    return n_components_value, n_neighbors_value, min_dist_value, metric_value


@app.callback(
    Output('umap-output', 'children'),
    Output('umap-embedding-store', 'data'),
    Output('umap-plotter', 'figure'),
    Input('umap-button', 'n_clicks'),
    State('data-store', 'data'),
    State('metadata-store', 'data'),
    State('umap-params-store', 'data'),
    prevent_initial_call=True
)
def update_umap_graph(n_clicks, stored_data, stored_metadata, stored_umap_params):
    global discrete_colors, title

    if stored_metadata is None:
        return (
            html.P(children=f'Metadata not found!', style={'color': 'red'}),
            None,
            None,
        )
    if stored_data is None:
        return (
            html.P(children=f'Data not found!', style={'color': 'red'}),
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
        topic_matrix,
        topic_term_matrix,
        document_topic_matrix,
    ) = stored_data

    topic_top_terms = [[meta_terms[term_id] for term_id in matutils.argsort(topic_terms, terms_per_topic, reverse=True)]
                       for topic_terms in topic_term_matrix]

    (
        n_components_value,
        n_neighbors_value,
        min_dist_value,
        metric_value,
    ) = stored_umap_params
    umap_model = ABFixedUMAP(
        n_components=n_components_value,
        n_neighbors=n_neighbors_value,
        min_dist=min_dist_value,
        metric=metric_value,
        random_state=random_seed,
    )

    embedding = umap_model.fit_transform(topic_term_matrix)
    if n_components_value == 3:
        fig = px.scatter_3d(
            pandas.DataFrame({
                'x': embedding[:, 0],
                'y': embedding[:, 1],
                'z': embedding[:, 2],
                'topic': [f'Topic {topic_id}' for topic_id in range(num_topics)],
                'topic_top_terms': [', '.join(topic_top_terms[topic_id]) for topic_id in range(num_topics)],
            }),
            x='x',
            y='y',
            z='z',
            color='topic',
            color_discrete_sequence=discrete_colors,
            title=title,
            hover_data={
                'x': False,
                'y': False,
                'z': False,
                'topic': True,
                'topic_top_terms': True,
            },
            labels={
                'x': 'UMAP Dimension 1',
                'y': 'UMAP Dimension 2',
                'z': 'UMAP Dimension 3',
                'topic': 'Topic',
                'topic_top_terms': 'Top Terms',
            },
            height=980,
        )
        fig.update_traces(
            marker_size=12,
            hovertemplate=None,
            hoverinfo='none',
        ).update_layout(
            hoverdistance=1,
        )
    elif n_components_value == 2:
        fig = px.scatter(
            pandas.DataFrame({
                'x': embedding[:, 0],
                'y': embedding[:, 1],
                'topic': [f'Topic {topic_id}' for topic_id in range(num_topics)],
                'topic_top_terms': [', '.join(topic_top_terms[topic_id]) for topic_id in range(num_topics)],
            }),
            x='x',
            y='y',
            color='topic',
            color_discrete_sequence=discrete_colors,
            title=title,
            hover_data={
                'x': False,
                'y': False,
                'topic': True,
                'topic_top_terms': True,
            },
            labels={
                'x': 'UMAP Dimension 1',
                'y': 'UMAP Dimension 2',
                'topic': 'Topic',
                'topic_top_terms': 'Top Terms',
            },
            height=980,
        )
        fig.update_traces(
            marker_size=12,
            hovertemplate=None,
            hoverinfo='none',
        ).update_layout(
            hoverdistance=1,
        )
    elif n_components_value == 1:
        fig = px.scatter(
            pandas.DataFrame({
                'x': embedding.flatten(),
                'y': [0] * len(embedding),
                'topic': [f'Topic {topic_id}' for topic_id in range(num_topics)],
                'topic_top_terms': [', '.join(topic_top_terms[topic_id]) for topic_id in range(num_topics)],
            }),
            x='x',
            y='y',
            color='topic',
            color_discrete_sequence=discrete_colors,
            title=title,
            hover_data={
                'x': False,
                'y': False,
                'topic': True,
                'topic_top_terms': True,
            },
            labels={
                'x': 'UMAP Dimension 1',
                'y': 'UMAP Dimension 2',
                'topic': 'Topic',
                'topic_top_terms': 'Top Terms',
            },
            height=980,
        )
        fig.update_traces(
            marker_size=12,
            hovertemplate=None,
            hoverinfo='none',
        ).update_layout(
            hoverdistance=1,
        )
    else:
        fig = None

    return (
        [
            f'Documents: {num_docs} / Terms: {num_terms} / Topics: {num_topics}',
            html.Br(),
            html.A('UMAP parameters', href='https://umap-learn.readthedocs.io/en/latest/parameters.html'),
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
        embedding,
        fig,
    )


@app.callback(
    Output('umap-hover-output', 'children'),
    Output('umap-hover-output', 'style'),
    Input('umap-plotter', 'hoverData')
)
def display_umap_hover(hover_data):
    if hover_data is None:
        return '', {'display': 'none'}

    hover_data = hover_data['points'][0]['customdata']
    return (
        [
            f'Topic: {hover_data[0]}',
            html.Br(),
            f'Top Terms: {hover_data[1]}',
        ],
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


@app.callback(
    Output('kmeans-scoring-params-store', 'data'),
    Input('n-clusters-slider1', 'value'),
)
def update_kmeans_scoring_params(n_clusters_value):
    return n_clusters_value


@app.callback(
    Output('kmeans-scoring-output', 'children'),
    Output('kmeans-scoring-plotter', 'figure'),
    Input('kmeans-scoring-button', 'n_clicks'),
    State('data-store', 'data'),
    State('umap-embedding-store', 'data'),
    State('kmeans-scoring-params-store', 'data'),
    prevent_initial_call=True
)
def update_kmeans_scoring_graph(
        n_clicks,
        stored_data,
        stored_umap_embedding,
        stored_kmeans_scoring_params
):
    global n_clusters_min

    if stored_data is None:
        return (
            html.P(children=f'Data not found!', style={'color': 'red'}),
            None,
        )
    if stored_umap_embedding is None:
        return (
            html.P(children=f'UMAP Embedding not found!', style={'color': 'red'}),
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

    list_of_num_clusters = list(range(n_clusters_min, n_clusters_value + 1))
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
        subplot_titles=('Inertia', 'Silhouette Avg. Scores'),
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
    fig.update_xaxes(row=1, col=1, title_text='K-clusters')
    fig.update_xaxes(row=1, col=2, title_text='K-clusters')
    fig.update_yaxes(row=1, col=1, title_text='Inertia')
    fig.update_yaxes(row=1, col=2, title_text='Silhouette Avg. Score')

    return (
        f'- n-clusters: {n_clusters_min} -> {n_clusters_value}',
        fig,
    )


@app.callback(
    Output('kmeans-params-store', 'data'),
    Input('n-clusters-slider', 'value'),
)
def update_kmeans_params(n_clusters_value):
    return n_clusters_value


@app.callback(
    Output('kmeans-output', 'children'),
    Output('kmeans-plotter', 'figure'),
    Output('kmeans-table', 'children'),
    Input('kmeans-button', 'n_clicks'),
    State('data-store', 'data'),
    State('metadata-store', 'data'),
    State('umap-params-store', 'data'),
    State('umap-embedding-store', 'data'),
    State('kmeans-params-store', 'data'),
    prevent_initial_call=True
)
def update_kmeans_graph(
        n_clicks,
        stored_data,
        stored_metadata,
        stored_umap_params,
        stored_umap_embedding,
        stored_kmeans_params,
):
    if stored_data is None:
        return (
            html.P(children=f'Data not found!', style={'color': 'red'}),
            None,
            None,
        )
    if stored_umap_embedding is None:
        return (
            html.P(children=f'UMAP Embedding not found!', style={'color': 'red'}),
            None,
            None,
        )

    _, meta_terms = stored_metadata

    (
        num_topics,
        terms_per_topic,
        random_seed,
        _,
        topic_term_matrix,
        _,
    ) = stored_data

    topic_top_terms = [[meta_terms[term_id] for term_id in matutils.argsort(topic_terms, terms_per_topic, reverse=True)]
                       for topic_terms in topic_term_matrix]

    (
        n_components_value,
        _,
        _,
        _,
    ) = stored_umap_params

    embedding = np.array(stored_umap_embedding)

    n_clusters_value = stored_kmeans_params

    kmeans = KMeans(
        n_clusters=n_clusters_value,
        random_state=random_seed,
    )
    labels = kmeans.fit_predict(embedding)

    table = html.Table(children=[
        html.Thead(children=[
            html.Tr(children=[
                html.Th(children='Topics'),
                html.Th(children='Top Terms'),
                html.Th(children='Labels'),
            ]),
        ]),
        html.Tbody(children=[
            html.Tr(children=[
                html.Td(children=f'Topic {topic_id}'),
                html.Td(children=', '.join(topic_top_terms[topic_id])),
                html.Td(children=f'Label {labels[topic_id]}'),
            ]) for topic_id in range(num_topics)
        ]),
    ])

    if n_components_value == 3:
        fig = px.scatter_3d(
            pandas.DataFrame({
                'x': embedding[:, 0],
                'y': embedding[:, 1],
                'z': embedding[:, 2],
                'topic': [f'Topic {topic_id}' for topic_id in range(num_topics)],
                'topic_top_terms': [', '.join(topic_top_terms[topic_id]) for topic_id in range(num_topics)],
                'topic_labels': [f'Label {labels[topic_id]}' for topic_id in range(num_topics)],
            }),
            x='x',
            y='y',
            z='z',
            color='topic_labels',
            color_discrete_sequence=discrete_colors,
            title='Clusters by KMeans of ' + title,
            hover_data={
                'x': False,
                'y': False,
                'z': False,
                'topic': True,
                'topic_top_terms': True,
                'topic_labels': True,
            },
            labels={
                'x': 'UMAP Dimension 1',
                'y': 'UMAP Dimension 2',
                'z': 'UMAP Dimension 3',
                'topic': 'Topic',
                'topic_top_terms': 'Top Terms',
                'topic_labels': 'Label',
            },
            height=980,
        )
        fig.update_traces(
            marker_size=12,
            hovertemplate=None,
            hoverinfo='none',
        ).update_layout(
            hoverdistance=1,
        )
    elif n_components_value == 2:
        fig = px.scatter(
            pandas.DataFrame({
                'x': embedding[:, 0],
                'y': embedding[:, 1],
                'topic': [f'Topic {topic_id}' for topic_id in range(num_topics)],
                'topic_top_terms': [', '.join(topic_top_terms[topic_id]) for topic_id in range(num_topics)],
                'topic_labels': [f'Label {labels[topic_id]}' for topic_id in range(num_topics)],
            }),
            x='x',
            y='y',
            color='topic_labels',
            color_discrete_sequence=discrete_colors,
            title='Clusters by KMeans of ' + title,
            hover_data={
                'x': False,
                'y': False,
                'topic': True,
                'topic_top_terms': True,
                'topic_labels': True,
            },
            labels={
                'x': 'UMAP Dimension 1',
                'y': 'UMAP Dimension 2',
                'topic': 'Topic',
                'topic_top_terms': 'Top Terms',
                'topic_labels': 'Label',
            },
            height=980,
        )
        fig.update_traces(
            marker_size=12,
            hovertemplate=None,
            hoverinfo='none',
        ).update_layout(
            hoverdistance=1,
        )
    elif n_components_value == 1:
        fig = px.scatter(
            pandas.DataFrame({
                'x': embedding.flatten(),
                'y': [0] * len(embedding),
                'topic': [f'Topic {topic_id}' for topic_id in range(num_topics)],
                'topic_top_terms': [', '.join(topic_top_terms[topic_id]) for topic_id in range(num_topics)],
                'topic_labels': [f'Label {labels[topic_id]}' for topic_id in range(num_topics)],
            }),
            x='x',
            y='y',
            color='topic_labels',
            color_discrete_sequence=discrete_colors,
            title='Clusters by KMeans of ' + title,
            hover_data={
                'x': False,
                'y': False,
                'topic': True,
                'topic_top_terms': True,
                'topic_labels': True,
            },
            labels={
                'x': 'UMAP Dimension 1',
                'y': 'UMAP Dimension 2',
                'topic': 'Topic',
                'topic_top_terms': 'Top Terms',
                'topic_labels': 'Label',
            },
            height=980,
        )
        fig.update_traces(
            marker_size=12,
            hovertemplate=None,
            hoverinfo='none',
        ).update_layout(
            hoverdistance=1,
        )
    else:
        fig = None

    return (
        [
            html.A('KMeans parameters', href='https://umap-learn.readthedocs.io/en/latest/parameters.html'),
            ':',
            html.Br(),
            f'- n_clusters: {n_clusters_value}',
        ],
        fig,
        table,
    )


@app.callback(
    Output('kmeans-hover-output', 'children'),
    Output('kmeans-hover-output', 'style'),
    Input('kmeans-plotter', 'hoverData')
)
def display_kmeans_hover(hover_data):
    if hover_data is None:
        return '', {'display': 'none'}

    hover_data = hover_data['points'][0]['customdata']
    return (
        [
            f'Topic: {hover_data[0]}',
            html.Br(),
            f'Top Terms: {hover_data[1]}',
            html.Br(),
            f'Label: {hover_data[2]}',
        ],
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


@app.callback(
    Output('hdbscan-params-store', 'data'),
    Input('min-cluster-size-slider', 'value'),
    Input('min-samples-slider', 'value'),
    Input('hdbscan-metric-selector', 'value'),
)
def update_hdbscan_params(min_cluster_size_value, min_samples_value, metric_value):
    return min_cluster_size_value, min_samples_value, metric_value


@app.callback(
    Output('hdbscan-output', 'children'),
    Output('hdbscan-plotter', 'figure'),
    Output('hdbscan-table', 'children'),
    Input('hdbscan-button', 'n_clicks'),
    State('data-store', 'data'),
    State('metadata-store', 'data'),
    State('umap-params-store', 'data'),
    State('umap-embedding-store', 'data'),
    State('hdbscan-params-store', 'data'),
    prevent_initial_call=True
)
def update_hdbscan_graph(
        n_clicks,
        stored_data,
        stored_metadata,
        stored_umap_params,
        stored_umap_embedding,
        stored_hdbscan_params,
):
    if stored_data is None:
        return (
            html.P(children=f'Data not found!', style={'color': 'red'}),
            None,
            None,
        )
    if stored_umap_embedding is None:
        return (
            html.P(children=f'UMAP Embedding not found!', style={'color': 'red'}),
            None,
            None,
        )

    _, meta_terms = stored_metadata

    (
        num_topics,
        terms_per_topic,
        random_seed,
        _,
        topic_term_matrix,
        _,
    ) = stored_data

    topic_top_terms = [[meta_terms[term_id] for term_id in matutils.argsort(topic_terms, terms_per_topic, reverse=True)]
                       for topic_terms in topic_term_matrix]

    (
        n_components_value,
        _,
        _,
        _,
    ) = stored_umap_params

    embedding = np.array(stored_umap_embedding)

    min_cluster_size_value, min_samples_value, metric_value = stored_hdbscan_params

    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size_value,
        min_samples=min_samples_value,
        metric=metric_value,
    ).fit_predict(embedding)

    table = html.Table(children=[
        html.Thead(children=[
            html.Tr(children=[
                html.Th(children='Topics'),
                html.Th(children='Top Terms'),
                html.Th(children='Labels'),
            ]),
        ]),
        html.Tbody(children=[
            html.Tr(children=[
                html.Td(children=f'Topic {topic_id}'),
                html.Td(children=', '.join(topic_top_terms[topic_id])),
                html.Td(children=f'Label {labels[topic_id]}'),
            ]) for topic_id in range(num_topics)
        ]),
    ])

    if n_components_value == 3:
        fig = px.scatter_3d(
            pandas.DataFrame({
                'x': embedding[:, 0],
                'y': embedding[:, 1],
                'z': embedding[:, 2],
                'topic': [f'Topic {topic_id}' for topic_id in range(num_topics)],
                'topic_top_terms': [', '.join(topic_top_terms[topic_id]) for topic_id in range(num_topics)],
                'topic_labels': [f'Label {labels[topic_id]}' for topic_id in range(num_topics)],
            }),
            x='x',
            y='y',
            z='z',
            color='topic_labels',
            color_discrete_sequence=discrete_colors,
            title='Clusters by HDBSCAN of ' + title,
            hover_data={
                'x': False,
                'y': False,
                'z': False,
                'topic': True,
                'topic_top_terms': True,
                'topic_labels': True,
            },
            labels={
                'x': 'UMAP Dimension 1',
                'y': 'UMAP Dimension 2',
                'z': 'UMAP Dimension 3',
                'topic': 'Topic',
                'topic_top_terms': 'Top Terms',
                'topic_labels': 'Label',
            },
            height=980,
        )
        fig.update_traces(
            marker_size=12,
            hovertemplate=None,
            hoverinfo='none',
        ).update_layout(
            hoverdistance=1,
        )
    elif n_components_value == 2:
        fig = px.scatter(
            pandas.DataFrame({
                'x': embedding[:, 0],
                'y': embedding[:, 1],
                'topic': [f'Topic {topic_id}' for topic_id in range(num_topics)],
                'topic_top_terms': [', '.join(topic_top_terms[topic_id]) for topic_id in range(num_topics)],
                'topic_labels': [f'Label {labels[topic_id]}' for topic_id in range(num_topics)],
            }),
            x='x',
            y='y',
            color='topic_labels',
            color_discrete_sequence=discrete_colors,
            title='Clusters by HDBSCAN of ' + title,
            hover_data={
                'x': False,
                'y': False,
                'topic': True,
                'topic_top_terms': True,
                'topic_labels': True,
            },
            labels={
                'x': 'UMAP Dimension 1',
                'y': 'UMAP Dimension 2',
                'topic': 'Topic',
                'topic_top_terms': 'Top Terms',
                'topic_labels': 'Label',
            },
            height=980,
        )
        fig.update_traces(
            marker_size=12,
            hovertemplate=None,
            hoverinfo='none',
        ).update_layout(
            hoverdistance=1,
        )
    elif n_components_value == 1:
        fig = px.scatter(
            pandas.DataFrame({
                'x': embedding.flatten(),
                'y': [0] * len(embedding),
                'topic': [f'Topic {topic_id}' for topic_id in range(num_topics)],
                'topic_top_terms': [', '.join(topic_top_terms[topic_id]) for topic_id in range(num_topics)],
                'topic_labels': [f'Label {labels[topic_id]}' for topic_id in range(num_topics)],
            }),
            x='x',
            y='y',
            color='topic_labels',
            color_discrete_sequence=discrete_colors,
            title='Clusters by HDBSCAN of ' + title,
            hover_data={
                'x': False,
                'y': False,
                'topic': True,
                'topic_top_terms': True,
                'topic_labels': True,
            },
            labels={
                'x': 'UMAP Dimension 1',
                'y': 'UMAP Dimension 2',
                'topic': 'Topic',
                'topic_top_terms': 'Top Terms',
                'topic_labels': 'Label',
            },
            height=980,
        )
        fig.update_traces(
            marker_size=12,
            hovertemplate=None,
            hoverinfo='none',
        ).update_layout(
            hoverdistance=1,
        )
    else:
        fig = None

    return (
        [
            html.A('HDBSCAN parameters', href='https://hdbscan.readthedocs.io/en/latest/parameter_selection.html'),
            ':',
            html.Br(),
            f'- min_cluster_size: {min_cluster_size_value}',
            html.Br(),
            f'- min_samples: {min_samples_value}',
            html.Br(),
            f'- metric: {metric_value}',
        ],
        fig,
        table,
    )


@app.callback(
    Output('hdbscan-hover-output', 'children'),
    Output('hdbscan-hover-output', 'style'),
    Input('hdbscan-plotter', 'hoverData')
)
def display_hdbscan_hover(hover_data):
    if hover_data is None:
        return '', {'display': 'none'}

    hover_data = hover_data['points'][0]['customdata']
    return (
        [
            f'Topic: {hover_data[0]}',
            html.Br(),
            f'Top Terms: {hover_data[1]}',
            html.Br(),
            f'Label: {hover_data[2]}',
        ],
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


if __name__ == '__main__':
    app.run(
        debug=debug,
        port=port,
    )
