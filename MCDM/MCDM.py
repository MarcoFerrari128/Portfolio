import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import json

all_seasons = pd.read_csv('all_seasons.csv')

def TOPSIS(df, features, weights=1):
    """TOPSIS MCDM method. Steps:

        1. Normalization;
        2. Ideal positive and negative;
        3. Distance S_plus, S_minus;
        4. Distance indicator C;
        5. Order based on C (descending).

        Input:
            df: dataframe of alternatives. Each row is an alternative, each columns is a feature.
            features: list of features to use during evaluation.
            weights: weights of each feature. If none, all features are assumed to be equally important.

        Output:
            ordered_df: original dataframe ordered from best to worst, with added 'score' feature.
    """
    A = df[features].values
    Z = A / np.max(A, axis=0)
    try:
        X = Z * np.array(weights)
    except:
        print('Wrong weights dimension.')
    A_plus = np.max(X, axis=0)
    A_minus = np.min(X, axis=0)

    S_plus = np.linalg.norm(X-A_plus, axis=1)
    S_minus = np.linalg.norm(X-A_minus, axis=1)

    C = S_minus / (S_plus+S_minus)
    index = np.argsort(C)[::-1]

    df['score'] = C
    return df.loc[index]


def Pareto(df, features):
    """Find Pareto frontier alternatives.
    """
    is_pareto = np.ones(df.shape[0], dtype=bool)
    for i in range(df.shape[0]):
        is_pareto *= np.logical_not(np.all(df.loc[i, features].values > df.loc[:, features].values, axis=1))
    df['is_pareto'] = np.int64(is_pareto)
    return df


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

## MCDM
features = ['player_name', 'pts', 'reb', 'ast', 'net_rating',
       'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct']

## Rendering

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

fig = px.line_polar(r=np.zeros_like(features[1:]), theta=features[1:], line_close=True)
fig.update_traces(fill='toself')
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0,1]
    )),
  showlegend=False
)



seasons = all_seasons['season'].unique()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
        html.H1('Multi-Criteria-Decision-Making'),
        html.H4(
            '''This app is a basic application of a MCDM system applied
            to a simple dataset - the NBA players dataset.
            It is possible to choose which features to use in the decision process.
            In the plot are shown the alternatives, and the colour indicates its score.
                  '''),
        html.Div([

            html.Div([
                html.Label('Season'),
                dcc.Dropdown(
                    id='season',
                    options=[{'label': i, 'value': i} for i in seasons[1:]],
                    value=seasons[-1]
                ),
                html.Label('MCDM attributes'),
                dcc.Dropdown(
                    id='MCDM-attributes',
                    options=[
                        {'label': i, 'value': i} for i in features[1:]
                    ],
                    value=features[1:],
                    multi=True
                )
            ], style={'width': '48%', 'float': 'left'}),

            html.Div([
                html.Label('X Axis'),
                dcc.Dropdown(
                    id='xaxis-column',
                    options=[{'label': i, 'value': i} for i in features[1:]],
                    value=features[2]
                ),

                html.Label('Y Axis'),
                dcc.Dropdown(
                    id='yaxis-column',
                    options=[{'label': i, 'value': i} for i in features[1:]],
                    value=features[3]
                )
            ], style={'width': '48%', 'float': 'left', 'columns':2}),


            html.Div([
                dcc.Graph(id='main-graph'),
                dcc.Graph(id='sub-graph', figure=fig)
            ], style={'width': '100%','columnCount': 2})
        ])
    ])

@app.callback(
    Output('main-graph', 'figure'),
    [Input('MCDM-attributes', 'value'),
     Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('season', 'value')
     ])
def update_graph(attributes, xaxis_column_name, yaxis_column_name, season):
    df = all_seasons[all_seasons['season']==season].reset_index()
    TOPSIS(df, attributes)

    return {
        'data': [dict(
            x=df[xaxis_column_name],
            y=df[yaxis_column_name],
            text=df['player_name'],
            mode='markers',
            marker={
                'color': df['score'],
                'size': 15,
                'opacity': 0.9,
                'line': {'width': 0.5},
                'colorscale':'Viridis',
                'show_colorscale':True
            }
        )],
        'layout': dict(
            xaxis={
                'title': xaxis_column_name,
                'type': 'linear'
            },
            yaxis={
                'title': yaxis_column_name,
                'type': 'linear'
            },
            hovermode='closest'
        )
    }

@app.callback(
    Output('sub-graph', 'figure'),
    [Input('main-graph', 'hoverData'),
     Input('season', 'value')])
def update_radar_graph(hoverData, season):
    df = all_seasons[all_seasons['season']==season].reset_index()
    data = df.loc[hoverData['points'][0]['pointNumber'], features[1:]].values
    all_data = df[features[1:]].values
    fig.data = []
    fig.update_layout(
        title={
            'text':df.loc[hoverData['points'][0]['pointNumber'], 'player_name'],
            'x':0.5,
            'y':1,
            'xanchor':'center'
            }
        )
    return  fig.add_trace(go.Scatterpolar(
            r=(data-all_data.min(axis=0))/(all_data.max(axis=0)-all_data.min(axis=0)),
            theta=features[1:],
            fill='toself',
            name=df.loc[hoverData['points'][0]['pointNumber'], 'player_name']
            ))


if __name__ == '__main__':
    app.run_server(debug=False)