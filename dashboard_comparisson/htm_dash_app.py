"""
This script contains the code for the dashboard that allows to compare hierarchical topic models.

Author: Lorena Calvo-Bartolomé
Date: 17/06/2023
"""

import configparser
import json
import os
import warnings

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from dash import Dash, Input, Output, dcc, html
import sys

sys.path.append('..')
from src.utils.misc import unpickler

warnings.simplefilter(action='ignore', category=FutureWarning)

# ======================================================
# Initialize the app - incorporate css
# ======================================================
app = Dash(
    __name__,
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1"}],
)
app.title = "Hierarchical Topic Modeling Comparisson Dashboard"

# ======================================================
# DATA
# ======================================================
# Get config with path to data
cf = configparser.ConfigParser()
current_path = \
    os.path.dirname(os.path.dirname(os.getcwd()))
if current_path.endswith("UserInLoopHTM"):
    cf.read(os.path.join(current_path,
                         'dashboard_comparisson',
                         'config',
                         'config.cf'))
else:
    cf.read(os.path.join(current_path,
                         'UserInLoopHTM',
                         'dashboard_comparisson',
                         'config',
                         'config.cf'))
    
# Unpickle model info dataframe
# We assume dataframe with the following format:
# # MODEL | MODEL_TYPE | FATHER_MODEL | CORPUS | ALPHAS | COHRS | KEYS
df = unpickler(cf.get("paths", "path_df_info"))

# Unpickle model sims and wmds dataframe
# We assume dataframe with the following format:
#  # MODEL_1 | MODEL_2 | VS_SIMS | WMD_1 | WMD_2
# # Note that there will ba as many rows as combinations of second level submodels belonging to the same first level model
df_sims_wmds = unpickler(cf.get("paths", "path_df_sims"))

# Read reference topics
with open(cf.get("paths", "path_ref_topics"), "r") as f:
    data = json.load(f)
df_ref = pd.DataFrame.from_records(data)
ref_titles = df_ref.title.values.tolist()


# ======================================================
# AUX VARS
# ======================================================
n_clicks = 0


# ======================================================
# AUX FUNCTIONS
# ======================================================
def description_card():
    """Returns a Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            # html.H5("UserInLoopHTM"),
            html.H3("HTMs Comparisson Dashboard"),
            html.Div(
                id="intro",
                children="Select a corpus for comparing topic models. Next, choose one primary topic model and two secondary-level topic models to conduct a comparative analysis.",
            ),
        ],
    )


def generate_control_card():
    """Returns a Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            # CORPUS
            html.P("Select Corpus"),
            dcc.Dropdown(
                id='corpus-input',
                options=df.corpus.unique(),
                value=df.corpus.unique()[0],
                style={
                    'color': '#212121',
                    'marginBottom': '5%'
                }
            ),
            html.Br(),

            # FIRST LEVEL TOPIC MODEL
            html.P("Select 1st level topic model"),
            dcc.Dropdown(
                id='first-level-tm-input',
            ),
            html.Br(),
            dcc.Graph(
                id='root_cohr_graph',
            ),
            html.Br(),

            # SECOND LEVEL TOPIC MODELS
            html.P("Select two 2nd level topic models to compare"),
            html.Div(
                children=[
                     dbc.Checklist(id="radio-items",
                          labelStyle=dict(display='inline')),
                    ],
                style={"maxHeight": "120px", "overflow": "scroll"}),
            html.Br(),
            dbc.Button(
                "Compare",
                id="compare-button",
                outline=True,
                color="primary",
                className="me-2",
                n_clicks=0
            ),
        ]
    )


def get_buttons_metrics():
    """Returns a Div containing buttons for metrics.
    """
    return html.Div(
        [
            dbc.RadioItems(
                id="radios-metric",
                options=[
                    {"label": "Topic Alignment",
                     "value":  "Topic Alignment"},
                    {"label": "WMD", "value": "WMD"}],
                value='Topic Alignment',
                inline=True,
                labelStyle={"display": "inline-block", "margin-right": "10px"}),
        ],
    )


def Table(dataframe):
    """Out of a one-row dataframe, it returns a table where each column in the dataframe is a row in the table.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe with one row and as many columns as desired.

    Returns
    -------
    html.Div
        Table with as many rows as columns in the dataframe.
    """
    rows_divs = []
    for i in range(len(dataframe)):
        # Create one row for each column in the dataframe
        for idx, col in enumerate(dataframe.columns):
            if col == "keywords":
                style = {"height": "100px"}
            else:
                style = {"height": "25px"}
            child_title = html.Div(
                id=col + "_title",
                style={"display": "table",
                       "height": "100%",
                       "vertical-align": "middle"},
                className="five columns",
                children=[html.B(col)]
            )
            child_value = html.Div(
                id=col + "_value",
                style={"display": "table",
                       "height": "100%",
                       "vertical-align": "middle"},
                className="five columns",
                children=[dataframe.iloc[i][col]]
            )
            if idx % 2 == 0:
                style["background-color"] = "#FFFFFF"
            row_div = html.Div(
                id=col + "_row",
                className="table-row",
                style=style,
                children=[child_title, child_value]
            )
            rows_divs.append(row_div)
    return html.Div(rows_divs)


# ======================================================
# LAYOUT
# ======================================================
app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            # children=[html.Img(src=app.get_asset_url("plotly_logo.png"))],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[
                description_card(),
                generate_control_card()]
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                # HEATMAPS
                html.Div(
                    id="heatmaps_card",
                    children=[
                        get_buttons_metrics(),
                        # html.B(id="tile_heatmap_card"),
                        html.Hr(),
                        dcc.Graph(id="heatmaps"),
                    ],
                ),
                # TOPICS INFO
                html.Div(
                    id="topics_info_card",
                    children=[
                        html.B("Topic Information"),
                        html.Hr(),
                        html.Div(id="topics_info",
                                 children="Click on any position on the heatmaps avobe to see the topic information of the selected topic for each submodel."),
                        html.Br(),
                        html.Div(id="table_keys"),
                    ],
                ),
            ],
        ),
    ],
)


# ======================================================
# CALLBACKS
# ======================================================
@app.callback(
    Output('first-level-tm-input', 'options'),
    Input('corpus-input', 'value')
)
def update_first_level_tm_input(selected_value):
    """Updates the options of the first-level topic model dropdown menu based on the selected corpus.

    Parameters
    ----------
    selected_value : str
        Selected corpus.
    """
    df_corpus = df[df.corpus == selected_value]
    options = df_corpus[df_corpus.model_type == "first"].model.unique()
    return options


@app.callback(
    Output('root_cohr_graph', 'figure'),
    Input('first-level-tm-input', 'value'))
def root_cohr_figure(selected_model):
    """Updates the root coherences graph based on the selected first-level topic model.

    Parameters
    ----------
    selected_model : str
        Selected first-level topic model.
    """
    if selected_model is None:
        # Return an empty layout with desired background color
        empty_layout = go.Layout(
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=50, r=50, t=40, b=10),
        )
        return go.Figure(layout=empty_layout)

    alphas = df[df.model == selected_model]['alphas'].values.tolist()[0]
    alphas = [float(value) for value in alphas.split(',')]
    cohrs = df[df.model == selected_model]['cohrs_cv'].values.tolist()[0]
    cohrs = [float(value) for value in cohrs.split(',')]
    # Create the bar trace
    bar_trace = go.Bar(
        x=[el for el in range(len(alphas))],
        y=cohrs,
        name="Coherence",
    )

    # Create the line trace
    line_trace = go.Scatter(
        x=[el for el in range(len(alphas))],
        y=alphas,
        mode='lines',
        line=dict(
            width=4,
            color="#A9294F"
        ),
        name='Size',
        yaxis='y2'
    )

    # Create the layout
    layout = go.Layout(
        barmode='stack',
        title=dict(text='Cohrence vs Size per Topic', font=dict(size=12)),
        xaxis=dict(title='Topic ID', titlefont=dict(size=10)),
        yaxis=dict(title='Cohrence', titlefont=dict(size=10)),
        yaxis2=dict(title='Size', overlaying='y',
                    side='right', titlefont=dict(size=10)),
        legend=dict(
            orientation='h',
            font=dict(size=8),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.5)',
            borderwidth=1
        ),
        showlegend=True,
        margin=dict(l=50, r=50, t=40, b=10),  # Adjust the margin values
    )
    # Combine the traces into a data list
    fig = dict(data=[bar_trace, line_trace], layout=layout)

    return fig


@app.callback(
    Output('radio-items', 'options'),
    [Input('first-level-tm-input', 'value'),
     Input('corpus-input', 'value')]
)
def update_radio_options(selected_model, selected_corpus):
    """Updates the options of the radio items based on the selected first-level topic model and corpus.

    Parameters
    ----------
    selected_model : str
        Selected first-level topic model.
    selected_corpus : str
        Selected corpus.
    """
    if selected_corpus is None or selected_model is None:
        return []

    df_corpus = df[df.corpus == selected_corpus]
    options = \
        df_corpus[(df_corpus.father_model == selected_model)].model.unique()
    options = [{'label': value, 'value': value} for value in options]

    return options


# @app.callback(
#     Output('tile_heatmap_card', 'children'),
#     Input("radios-metric", "value")
# )
# def update_tile_heatmap_card(radio_metrics_value):
#     return radio_metrics_value


@app.callback(
    Output('heatmaps', 'figure'),
    [
        Input("radio-items", "value"),
        Input("radios-metric", "value"),
        Input("compare-button", "n_clicks")
    ]
)
def update_heatmap(radio_items_value, radio_metrics_value, n):
    """Updates the heatmaps based on the selected submodels and metric. If the metric selected is "Topic Alignment", one heatmap representing the alignment between the two selected submodels is returned. If the metric selected is "Word Mover's Distance", two heatmaps are returned, each of the representing the WMD between each submodel's topics and a list of reference topics.

    Parameters
    ----------
    radio_items_value : list
        List of selected submodels.
    radio_metrics_value : int
        Selected metric.
    n : int
        Number of times the compare button has been clicked.
    """

    empty_layout = go.Layout(
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    if n is None:
        return go.Figure(layout=empty_layout)

    if n > 0 and n > n_clicks and radio_metrics_value != 1:

        if len(radio_items_value) < 2:
            return go.Figure(layout=empty_layout)

        row = df_sims_wmds[((df_sims_wmds.model_1 == radio_items_value[0]) & (df_sims_wmds.model_2 == radio_items_value[1])) | (
            (df_sims_wmds.model_1 == radio_items_value[1]) & (df_sims_wmds.model_2 == radio_items_value[0]))]

        if radio_metrics_value == "Topic Alignment":
            vals = row.vs_sims.values[0]

            x = ["Topic " + str(i) for i in range(vals.shape[1])]
            y = ["Topic " + str(i) for i in range(vals.shape[0])]
            z = vals
            # Only show rounded value (full value on hover)
            z_text = np.around(z, decimals=2)

            # Heatmap
            data = [
                dict(
                    x=x,
                    y=y,
                    z=z,
                    type="heatmap",
                    name="",
                    hovertemplate="<b> %{y} vs %{x} <br><br> %{z}",
                    texttemplate="%{text}",
                    text=z_text,
                    textfont=dict(family="sans-serif"),
                    showscale=False,
                    colorscale=[[0, "#caf3ff"], [1, "#2c82ff"]],
                )
            ]

            layout = dict(
                margin=dict(l=70, b=50, t=50, r=50),
                modebar={"orientation": "v"},
                font=dict(family="Open Sans"),
                xaxis=dict(
                    side="top",
                    ticks="",
                    ticklen=2,
                    tickfont=dict(family="sans-serif"),
                    tickcolor="#ffffff",
                ),
                yaxis=dict(
                    side="left", ticks="", tickfont=dict(family="sans-serif"), ticksuffix=" "
                ),
                hovermode="closest",
                showlegend=False,
            )
            return {"data": data, "layout": layout}

        elif radio_metrics_value == "WMD":
            vals1 = row.wmd_1.values[0]
            vals2 = row.wmd_2.values[0]

            x = ["Topic " + str(i) for i in range(vals1.shape[1])]
            y = ref_titles

            z_text1 = np.around(vals1, decimals=2)
            z_text2 = np.around(vals2, decimals=2)
            # Create two subplots
            fig = sp.make_subplots(rows=1, cols=2, shared_yaxes=True,
                                   column_titles=(
                                       row.model_1.values[0],
                                       row.model_2.values[0]),
                                   horizontal_spacing=0.02)

            # Add the first subplot
            fig.add_trace(
                go.Heatmap(
                    z=vals1,
                    x=x,
                    y=y,
                    colorscale=[[0, "#caf3ff"], [1, "#2c82ff"]],
                    texttemplate="%{text}",
                    text=z_text1,
                    hovertemplate="<b> '%{y}' vs %{x} <br><br> %{z}",
                    textfont=dict(family="sans-serif"),
                    showscale=False,
                ),
                row=1,
                col=1,
            )

            # Add the second subplot
            fig.add_trace(
                go.Heatmap(
                    z=vals2,
                    x=x,
                    y=y,
                    colorscale=[[0, "#caf3ff"], [1, "#2c82ff"]],
                    texttemplate="%{text}",
                    text=z_text2,
                    hovertemplate="<b> '%{y}' vs %{x} <br><br> %{z}",
                    textfont=dict(family="sans-serif"),
                    showscale=False,
                ),
                row=1,
                col=2,
            )

            fig.update_xaxes(
                title='Topic ID in submodel', titlefont=dict(family="sans-serif", size=14),
                tickangle=-15,
                row=1, col=1)
            fig.update_xaxes(
                title='Topic ID in submodel', titlefont=dict(family="sans-serif", size=14),
                tickangle=-15,
                row=1, col=2)
            fig.update_yaxes(tickmode='linear',
                             title='Reference topic', titlefont=dict(family="sans-serif", size=14))
            fig.update_layout(
                margin=dict(l=70, b=50, t=50, r=50),
                modebar={"orientation": "v"},
                font=dict(family="Open Sans"),
                xaxis=dict(
                    ticks="",
                    ticklen=2,
                    tickfont=dict(family="sans-serif"),
                    tickcolor="#ffffff",
                ),
                xaxis2=dict(
                    ticks="",
                    ticklen=2,
                    tickfont=dict(family="sans-serif"),
                    tickcolor="#ffffff",
                ),
                yaxis=dict(
                    side="left", ticks="", tickfont=dict(family="sans-serif"), ticksuffix=" "
                ),
                hovermode="closest",
                showlegend=False,
            )
            fig.update_annotations(font=dict(family="sans-serif", size=16))

            return fig

    else:
        pass
        empty_layout = go.Layout(
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return go.Figure(layout=empty_layout)


@app.callback(
    Output('table_keys', 'children'),
    [Input('heatmaps', 'clickData'),
     Input("radio-items", "value"),
     Input("radios-metric", "value")])
def generate_topics_tables(click_data, radio_items_value, radio_metrics_value):
    """Returns a table with the topic information for the topic selected on the top right heatmap for each submodel under comparison.

    Parameters
    ----------
    click_data : dict
        The clickData property of the heatmap.
    radio_items_value : list
        The list of the two submodels under comparison.
    radio_metrics_value : str
        The metric selected in the radio buttons.
    """
    if click_data is None or len(radio_items_value) < 2:
        return html.Div()

    # Get the clicked label information
    label = click_data['points'][0]['x']
    value = click_data['points'][0]['y']

    # Get the dataframes
    df1 = df[df.model == radio_items_value[0]]
    df2 = df[df.model == radio_items_value[1]]

    def get_df_topic(df, topic):
        topic = int(topic.split(" ")[1])
        df = pd.DataFrame(
            {
                'id': topic,
                'keywords': ", ".join(df.keywords.values[0][topic]),
                'alpha': [float(idx) for idx in df.alphas.values[0].split(', ')][topic],
                'coherence': [float(idx) for idx in df.cohrs_cv.values[0].split(', ')][topic],
            }, index=[0]
        )

        return df

    if radio_metrics_value == "Topic Alignment":
        df1 = get_df_topic(df1, label)
        df2 = get_df_topic(df2, value)

    elif radio_metrics_value == "WMD":
        df1 = get_df_topic(df1, label)
        df2 = get_df_topic(df2, label)

    return html.Div(
        id="right-column-tables",
        children=[
            html.Div(
                id="table_1",
                className="six columns",
                children=[
                    html.Br(),
                    html.B(radio_items_value[0]),
                    html.Hr(),
                    Table(df1),
                ]
            ),
            html.Div(
                id="table_2",
                className="six columns",
                children=[
                    html.Br(),
                    html.B(radio_items_value[1]),
                    html.Hr(),
                    Table(df2),
                ]
            ),
        ],
    ),


if __name__ == '__main__':
    app.run_server(debug=True, port=8055)
