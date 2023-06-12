# Import packages
import warnings

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from src.utils.misc import unpickler

warnings.simplefilter(action='ignore', category=FutureWarning)

# Initialize the app - incorporate css
app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])

# ======================================================
# DATA
# ======================================================
# Unpickle model info dataframe
# We assume dataframe with the following format:
# # MODEL | MODEL_TYPE | FATHER_MODEL | CORPUS | ALPHAS | COHRS | KEYS
df = unpickler(
    '/Users/lbartolome/Documents/GitHub/UserInLoopHTM/data/info.pkl')
# Unpickle model sims and wmds dataframe
# We assume dataframe with the following format:
#  # MODEL_1 | MODEL_2 | VS_SIMS | WMD_1 | WMD_2
# # Note that there will ba as many rows as combinations of second level submodels belonging to the same first level model
df_sims_wmds = unpickler(
    '/Users/lbartolome/Documents/GitHub/UserInLoopHTM/data/sims_wmds.pkl')

# ======================================================
# LAYOUT
# ======================================================
app.layout = dbc.Container([
    dbc.Row([
        ########################################################################
        # Left column #
        ########################################################################
        dbc.Col(
            html.Div(
                children=[
                    html.H4(
                        children='Hierarchical Topic Models comparisson',
                        style={
                            'color': '#213555',
                            'marginBottom': '1%'
                        }
                    ),

                    html.Div(
                        children='Choose a corpus to compare the topic models. Then, select one first topic model and two second-level topic models to compare.',
                        style={
                            'color': '#212A3E',
                            'marginBottom': '5%'
                        },
                    ),

                    html.Label(
                        children='Corpus',
                        style={
                            'font-weight': 'bold',
                            'color': '#4F709C',
                            'marginBottom': '1%'
                        }
                    ),
                    html.Div(
                        children=[
                            dcc.Dropdown(
                                id='corpus-input',
                                options=df.corpus.unique(),
                                value=df.corpus.unique()[0],
                                style={
                                    'color': '#212121',
                                    'marginBottom': '5%'
                                }
                            ),
                        ],
                        style={"width": "65%"},
                    ),
                    html.Label(
                        children='1st level topic model',
                        style={
                            'font-weight': 'bold',
                            'color': '#4F709C',
                            'marginBottom': '1%'
                        }
                    ),
                    html.Div(
                        children=[
                            dcc.Dropdown(
                                id='first-level-tm-input',
                                style={
                                    'color': '#212121',
                                    'marginBottom': '5%'
                                }
                            ),
                        ],
                        style={"width": "65%"},
                    ),

                    dcc.Graph(
                        id='root_cohr_graph',
                        style={
                            'marginBottom': '5%',
                        },
                        responsive=True
                    ),

                    html.Label(
                        children='2nd level topic models',
                        style={
                            'font-weight': 'bold',
                            'color': '#4F709C',
                            'marginBottom': '1%'
                        }
                    ),

                    html.Div(
                        children=[
                            dbc.Label("Choose two models to compare:"),
                            dbc.Checklist(
                                id="radio-items",
                            ),
                        ]
                    ),

                    html.Div(
                        children=[
                            dbc.Button(
                                "Click me",
                                id="example-button",
                                outline=True,
                                color="primary",
                                className="me-2",
                                n_clicks=0
                            ),
                            html.Span(id="example-output",
                                      style={"verticalAlign": "middle"}),
                        ]
                    )
                ],
                style={'padding': 10,
                       'margin': 10,
                       'backgroundColor': '#F4F4F4'}
            ), width=4),

        ########################################################################
        # Rigth column #
        ########################################################################
        dbc.Col(
            html.Div(
                children=[
                    dbc.Row([
                        html.Div(
                            children=[
                                html.Label(
                                    children='Metric to be used for comparisson:',
                                    style={
                                        'font-weight': 'bold',
                                        'color': '#4F709C',
                                        'margin': '0 0 0 10px',
                                        'display': 'flex',
                                        'align-items': 'center'
                                    }
                                ),
                                html.Div(
                                    [
                                        dbc.RadioItems(
                                            id="radios",
                                            className="btn-group",
                                            inputClassName="btn-check",
                                            labelClassName="btn btn-outline-primary",
                                            labelCheckedClassName="active",
                                            options=[
                                                {"label": "Topic Alignment",
                                                    "value":  1},
                                                {"label": "WMD", "value": 2},
                                            ],
                                            value=1,
                                        ),
                                        html.Div(id="output"),
                                    ],
                                    className="radio-group",
                                )
                            ],
                            style={
                                'display': 'flex',
                                'align-items': 'center',
                                'marginBottom': '2%'}),
                    ]),
                    dbc.Row([
                        dcc.Graph(
                            id='submodel-vs-graph',
                            responsive=True
                        ),
                    ], style={'marginBottom': '2%'}),
                    dbc.Row(
                        dash_table.DataTable(
                            id='table_keys',
                            style_header={
                                # 'border': '1px solid black',
                                'font-family': 'sans-serif',
                                'backgroundColor': 'white',
                                'fontWeight': 'bold'},
                            style_cell={'border': '1px solid grey',
                                        'font-family': 'sans-serif',
                                        'padding': '5px'},
                            style_cell_conditional=[
                                {
                                    'if': {'column_id': c},
                                    'textAlign': 'left'
                                } for c in ['topic', 'keywords']
                            ],
                            style_as_list_view=True,
                            fill_width=False
                        ),
                    ),
                ],
                style={'padding': 10,
                       'margin': 10,
                       'backgroundColor': '#F5EFE7'}
            ),
            width=8),

    ],),
], style={"height": "90vh",
          "font-size": "12px"},
    fluid=True)


@app.callback(
    Output('first-level-tm-input', 'options'),
    Input('corpus-input', 'value')
)
def update_first_level_tm_input(selected_value):
    df_corpus = df[df.corpus == selected_value]
    options = df_corpus[df_corpus.model_type == "first"].model.unique()
    return options


@app.callback(
    Output('submodel-vs-graph', 'figure'),
    Input('first-level-tm-input', 'value'))
def update_figure_submodel_vs(selected_model):
    if selected_model is None:
        # Return an empty layout with desired background color
        empty_layout = go.Layout(
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return go.Figure(layout=empty_layout)
    vals = df[df.model == selected_model]['sims'].values[0]
    x = ["Topic " + str(i) for i in range(vals.shape[1])]
    y = ["Topic " + str(i) for i in range(vals.shape[0])]
    z = vals
    # Only show rounded value (full value on hover)
    z_text = np.around(z, decimals=2)

    fig = px.imshow(z, x=x, y=y, color_continuous_scale='Blues', aspect="auto")
    fig.update_traces(text=z_text, texttemplate="%{text}")
    fig.update_xaxes(side="top")
    # Adjust the margin values to reduce the size of the figure
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    return fig


@app.callback(
    Output('table_keys', 'data'),
    [Input('submodel-vs-graph', 'clickData'),
     Input('first-level-tm-input', 'value')])
def update_selected_label_text(click_data, selected_model):
    if click_data is None or selected_model is None:
        return []

    # Get the clicked label information
    label = click_data['points'][0]['x']
    value = click_data['points'][0]['y']

    # TODO: Update this to get info of submodels
    desc = df[df.model == selected_model]['keywords'].values.tolist()[0]
    keys = [el for el in range(len(desc))]
    df2 = pd.DataFrame({"topic": keys, "keywords": desc})
    df2["keywords"] = df2["keywords"].apply(lambda x: ", ".join(x))

    table_data = df2.to_dict('rows')

    return table_data


@app.callback(
    Output('radio-items', 'options'),
    Input('corpus-input', 'value')
)
def update_radio_options(selected_corpus):
    if selected_corpus is None:
        return []

    df_corpus = df[df.corpus == selected_corpus]
    # TODO: Change dataframe so father model is contemplated
    options = df_corpus[(df_corpus.model_type == "WS") | (
        df_corpus.model_type == "DS")].model.unique()
    options = [{'label': value, 'value': value} for value in options]

    return options

# Callback to check with checklists have been selected
@app.callback(
    Output("radioitems-checklist-output", "children"),
    [
        Input("radioitems-input", "value"),
        Input("checklist-input", "value"),
        Input("switches-input", "value"),
    ],
)
def on_form_change(radio_items_value, checklist_value, switches_value):
    template = "Radio button {}, {} checklist item{} and {} switch{} selected."

    n_checkboxes = len(checklist_value)
    n_switches = len(switches_value)

    output_string = template.format(
        radio_items_value,
        n_checkboxes,
        "s" if n_checkboxes != 1 else "",
        n_switches,
        "es" if n_switches != 1 else "",
    )
    return output_string



@app.callback(
    Output('root_cohr_graph', 'figure'),
    Input('first-level-tm-input', 'value'))
def root_cohr_figure(selected_model):
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
    cohrs = df[df.model == selected_model]['cohrs'].values.tolist()[0]
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
    Output("example-output", "children"), [Input("example-button", "n_clicks")]
)
def on_button_click(n):
    if n is None:
        return "Not clicked."
    else:
        return f"Clicked {n} times."

# @app.callback(Output("output", "children"), [Input("radios", "value")])
# def display_value(value):
#    return f"Selected value: {value}"


if __name__ == '__main__':
    app.run_server(debug=True, port=8055)
