import argparse
import base64
import json

import numpy as np
import pandas

from dash import Dash, html, dcc, Output, Input, State
from gensim import matutils
from plotly import express as px

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

metric = 'cosine'
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

title = 'UMAP of LDA Topic-Term Matrix'

app = Dash(
    __name__,
    title=title,
)

app.layout = html.Div([
    dcc.Store(id='metadata-store', storage_type='memory'),
    dcc.Store(id='data-store', storage_type='memory'),
    dcc.Store(id='umap-store', storage_type='memory'),
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
            'Calculate and plot UMAP',
            id='plot-button',
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
        id='plot-output',
        style={
            'margin': '25px 0',
            'padding': '0 25px',
        }
    ),
    dcc.Graph(
        id='plotter',
        responsive='auto',
        clear_on_unhover=True,
    ),
    html.Div(
        id='hover-output',
        style={
            'display': 'none',
        },
    ),
])


@app.callback(
    Output('hover-output', 'children'),
    Output('hover-output', 'style'),
    Input('plotter', 'hoverData')
)
def display_hover(hover_data):
    if hover_data is None:
        return '', {'display': 'none'}

    hover_data = hover_data['points'][0]['customdata']
    return (
        [
            f'Document: {hover_data[2]}',
            html.Br(),
            f'Dominant Topic: {hover_data[0]}',
            html.Br(),
            f'Dominant Keywords: {hover_data[1]}',
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
        }
    )


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
    Input('data-uploader', 'contents'),
    State('data-uploader', 'filename'),
    State('data-uploader', 'last_modified'),
    prevent_initial_call=True
)
def update_data(file_content, file_name, file_last_modified):
    try:
        _, base64_content = file_content.split(',')
        data = json.loads(base64.b64decode(base64_content).decode('utf-8'))
        return (
            data['num_topics'],
            data['terms_per_topic'],
            data['random_seed'],
            data['topic_matrix'],
            data['topic_term_matrix'],
            data['document_topic_matrix'],
        ), html.P(children=[f'Data: ', html.Span('OK', style={'color': 'green'})])
    except:
        if file_name is None:
            return None, html.P(children=f'Must upload "umap_data.json" first!', style={'color': 'red'})
        return None, html.P(children=f'Cannot get data from {file_name}', style={'color': 'red'})


@app.callback(
    Output('umap-store', 'data'),
    Input('n-components-slider', 'value'),
    Input('n-neighbors-slider', 'value'),
    Input('min-dist-slider', 'value'),
    Input('metric-selector', 'value'),
)
def update_umap(n_components_value, n_neighbors_value, min_dist_value, metric_value):
    return n_components_value, n_neighbors_value, min_dist_value, metric_value


@app.callback(
    Output('plot-output', 'children'),
    Output('plotter', 'figure'),
    Input('plot-button', 'n_clicks'),
    State('data-store', 'data'),
    State('metadata-store', 'data'),
    State('umap-store', 'data'),
    prevent_initial_call=True
)
def update_graph(n_clicks, stored_data, stored_metadata, stored_umap):
    global discrete_colors, title

    if stored_metadata is None:
        return html.P(children=f'Metadata not found!', style={'color': 'red'}), None
    if stored_data is None:
        return html.P(children=f'Data not found!', style={'color': 'red'}), None

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

    term_topic_matrix = np.array(topic_term_matrix).T.tolist()
    term_dominant_topics = np.argmax(term_topic_matrix, axis=1)

    n_components_value, n_neighbors_value, min_dist_value, metric_value = stored_umap
    umap_model = ABFixedUMAP(
        n_components=n_components_value,
        n_neighbors=n_neighbors_value,
        min_dist=min_dist_value,
        metric=metric_value,
        random_state=random_seed,
    )

    embedding = umap_model.fit_transform(term_topic_matrix)
    if n_components_value == 3:
        fig = px.scatter_3d(
            pandas.DataFrame({
                'x': embedding[:, 0],
                'y': embedding[:, 1],
                'z': embedding[:, 2],
                'term_dominant_topic': [f'Topic {topic_id}' for topic_id in term_dominant_topics],
                'term_dominant_keywords': [', '.join(topic_top_terms[topic_id])
                                           for topic_id in term_dominant_topics],
                'term': meta_terms,
            }),
            x='x',
            y='y',
            z='z',
            color='term_dominant_topic',
            color_discrete_sequence=discrete_colors,
            title=title,
            hover_data={
                'x': False,
                'y': False,
                'z': False,
                'term_dominant_topic': True,
                'term_dominant_keywords': True,
                'term': True,
            },
            labels={
                'x': 'UMAP Dimension 1',
                'y': 'UMAP Dimension 2',
                'z': 'UMAP Dimension 3',
                'document_dominant_topic': 'Dominant Topic',
                'document_dominant_keywords': 'Dominant Keywords',
                'document': 'Document',
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
                'term_dominant_topic': [f'Topic {topic_id}' for topic_id in term_dominant_topics],
                'term_dominant_keywords': [', '.join(topic_top_terms[topic_id])
                                           for topic_id in term_dominant_topics],
                'term': meta_terms,
            }),
            x='x',
            y='y',
            color='term_dominant_topic',
            color_discrete_sequence=discrete_colors,
            title=title,
            hover_data={
                'x': False,
                'y': False,
                'term_dominant_topic': True,
                'term_dominant_keywords': True,
                'term': True,
            },
            labels={
                'x': 'UMAP Dimension 1',
                'y': 'UMAP Dimension 2',
                'document_dominant_topic': 'Dominant Topic',
                'document_dominant_keywords': 'Dominant Keywords',
                'document': 'Document',
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
                'term_dominant_topic': [f'Topic {topic_id}' for topic_id in term_dominant_topics],
                'term_dominant_keywords': [', '.join(topic_top_terms[topic_id])
                                           for topic_id in term_dominant_topics],
                'term': meta_terms,
            }),
            x='x',
            y='y',
            color='term_dominant_topic',
            color_discrete_sequence=discrete_colors,
            title=title,
            hover_data={
                'x': False,
                'y': False,
                'term_dominant_topic': True,
                'term_dominant_keywords': True,
                'term': True,
            },
            labels={
                'x': 'UMAP Dimension 1',
                'y': 'UMAP Dimension 2',
                'document_dominant_topic': 'Dominant Topic',
                'document_dominant_keywords': 'Dominant Keywords',
                'document': 'Document',
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
        fig,
    )


if __name__ == '__main__':
    app.run(
        debug=debug,
        port=port,
    )
