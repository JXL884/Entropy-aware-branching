import json
from pathlib import Path
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from entropix.model import Generation
from entropix.config import SamplerConfig

# Import your existing plot functions
from entropix.plot import plot_sampler, plot_entropy

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select a JSON file')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False
                ),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='sampler-plot', style={'height': '600px'}),
            ], width=12),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='entropy-plot', style={'height': '800px'}),
            ], width=12),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div(id='tokens-text', style={
                    'whiteSpace': 'pre-wrap',
                    'fontFamily': 'monospace',
                    'fontSize': '14px',
                    'padding': '20px',
                    'border': '1px solid #ddd',
                    'borderRadius': '5px',
                    'marginBottom': '20px'
                })
            ], width=12),
        ]),
    ], fluid=True)
])

@callback(
    [Output('sampler-plot', 'figure'),
     Output('entropy-plot', 'figure'),
     Output('tokens-text', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_plots(contents, filename):
    if contents is None:
        # return dash.no_update, dash.no_update, "No file selected"
        filename = "test.json"

    generation_data = Generation.load(filename)
    print(generation_data)
    sampler_config = SamplerConfig()  # Use default config or load from data if available

    # Generate plots using your existing functions
    sampler_fig = plot_sampler(generation_data, out=None)
    entropy_fig = plot_entropy(generation_data, sampler_config, out=None)

    # Format tokens text
    tokens_text = ' '.join(generation_data.tokens)

    return sampler_fig, entropy_fig, tokens_text

if __name__ == '__main__':
    app.run_server(debug=True)
