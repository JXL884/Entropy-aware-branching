import os
import base64
import json
from dataclasses import dataclass
import tyro
from pathlib import Path
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from entropix.model import Generation
from entropix.config import SamplerConfig, SamplerState

# Import your existing plot functions
from entropix.plot import plot_sampler, plot_entropy


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
default_file = None

app.layout = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div(["Drag and drop or click to load a data file"]),
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "10px",
                                    },
                                    multiple=False,
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Show/Hide Tokens",
                                    id="collapse-button",
                                    className="mb-3",
                                    color="primary",
                                ),
                                dbc.Collapse(
                                    html.Div(id="tokens-text"),
                                    id="collapse",
                                    is_open=False,
                                )
                            ],
                            width=12
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    "Click and drag the x-axis labels to scroll the plot.Click the legend to show/hide entropy and varentropy traces.",
                                    style={'fontStyle': 'italic', 'color': '#f00', 'marginBottom': '10px', 'textAlign': 'center'}
                                )
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Max Tokens:"),
                                dcc.Slider(
                                    id='max-tokens-slider',
                                    min=50,
                                    max=300,
                                    step=10,
                                    value=100,
                                    marks={**{i: str(i)
                                              for i in range(50, 301, 50)}, 300: 'all'},
                                )
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [dbc.Checklist(
                                options=[{"label": "Show tokens on x-axis", "value": True}],
                                value=[True],
                                id="show-labels-toggle",
                                switch=True,
                            )]
                        )
                    ]
                ),
                dbc.Row([
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="sampler-plot",
                                style={
                                    "height": "600px",
                                },
                            ),
                        ],
                        width=12,
                    ),
                ]),
                html.Div(
                    [
                        html.H4("Entropy Plot Controls"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Logits", style={'font-weight': 'bold'}),
                                        dbc.Checklist(
                                            options=[
                                                {"label": "Show Logits Plot", "value": "show_logits"},
                                                {"label": "Show Lines", "value": "show_logits_lines"},
                                                {"label": "Show Entropy Thresholds", "value": "show_logits_entropy"},
                                                {"label": "Show Varentropy Thresholds", "value": "show_logits_varentropy"},
                                            ],
                                            value=["show_logits"],
                                            id="logits-controls",
                                            switch=True,
                                        ),
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Attention", style={'font-weight': 'bold'}),
                                        dbc.Checklist(
                                            options=[
                                                {"label": "Show Attention Plot", "value": "show_attention"},
                                                {"label": "Show Lines", "value": "show_attention_lines"},
                                                {"label": "Show Entropy Thresholds", "value": "show_attention_entropy"},
                                                {"label": "Show Varentropy Thresholds", "value": "show_attention_varentropy"},
                                            ],
                                            value=[],
                                            id="attention-controls",
                                            switch=True,
                                        ),
                                    ],
                                    width=6
                                ),
                            ]
                        ),
                    ],
                    style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px', 'marginTop': '135px'}
                ),
                dbc.Row([
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="entropy-plot",
                                style={
                                    "height": "800px",
                                    "margin": "50px 5% 5% 5%",
                                },
                            ),
                        ],
                        width=12,
                    ),
                ]),
            ],
            fluid=True,
        )
    ]
)

@callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

@callback(
    [Output("sampler-plot", "figure"), Output("entropy-plot", "figure"),
     Output("tokens-text", "children")],
    [Input("max-tokens-slider", "value"),
     Input("show-labels-toggle", "value"),
     Input("logits-controls", "value"),
     Input("attention-controls", "value"),
     Input("upload-data", "contents")],
)
def update_plots(max_tokens, show_labels, logits_controls, attn_controls, contents):
    if contents is None and default_file is None:
        return dash.no_update, dash.no_update, "No file selected"
    elif contents is not None:
        filename = os.path.join(os.getcwd(), "tmp.json")
        with open(filename, "w") as f:
            content_type, content_string = contents.split(',') # noqa: F841
            decoded = base64.b64decode(content_string)
            f.write(decoded.decode('utf-8'))

    filename = filename or default_file
    assert filename is not None

    generation_data = Generation.load(filename)
    print("loaded generation data")
    if filename != default_file: os.remove(filename)
    sampler_config = SamplerConfig()  # TODO: load sampler in gen data

    max_tokens = float('inf') if max_tokens == 300 else max_tokens
    sampler_fig = plot_sampler(generation_data, max_tokens=max_tokens, show_labels=bool(show_labels))  # type: ignore
    entropy_fig = plot_entropy(generation_data, sampler_config)

    # Create visibility array based on controls
    visibility = []
    # Logits line [0]
    visibility.append("show_logits_lines" in logits_controls and "show_logits" in logits_controls)
    # Attention line [1]
    visibility.append("show_attention_lines" in attn_controls and "show_attention" in attn_controls)
    # Logits points [2]
    visibility.append("show_logits" in logits_controls)
    # Attention points [3]
    visibility.append("show_attention" in attn_controls)
    # Logits entropy thresholds [4:7]
    visibility.extend([("show_logits_entropy" in logits_controls and "show_logits" in logits_controls)] * 3)
    # Logits varentropy thresholds [7:10]
    visibility.extend([("show_logits_varentropy" in logits_controls and "show_logits" in logits_controls)] * 3)
    # Attention entropy thresholds [10:13]
    visibility.extend([("show_attention_entropy" in attn_controls and "show_attention" in attn_controls)] * 3)
    # Attention varentropy thresholds [13:16]
    visibility.extend([("show_attention_varentropy" in attn_controls and "show_attention" in attn_controls)] * 3)
    for i, vis in enumerate(visibility):
        if i < len(entropy_fig.data):  # type: ignore
            entropy_fig.data[i].visible = vis  # type: ignore

    # Format tokens with inline CSS styles
    tokens_html = list(generation_data.prompt)
    for token, state in zip(generation_data.tokens, generation_data.sampler_states):
        color = {
            SamplerState.FLOWING: '#87CEEB',  # lightblue
            SamplerState.TREADING: '#90EE90',  # lightgreen
            SamplerState.EXPLORING: '#FFA500',  # orange
            SamplerState.RESAMPLING: '#FFB6C1',  # pink
            SamplerState.ADAPTIVE: '#800080'  # purple
        }[state]

        tokens_html.append(html.Span(token + ' ', style={'color': color}))  # type: ignore

    return sampler_fig, entropy_fig, html.Div(
        tokens_html,
        style={
            'whiteSpace': 'pre-wrap',
            'fontFamily': 'monospace',
            'fontSize': '14px',
            'padding': '20px',
            'border': '1px solid #ddd',
            'borderRadius': '5px',
            'marginBottom': '20px',
        }
    )

@dataclass
class DashboardConfig:
    file: str | None = None
    port: int = 8050
    host: str = '127.0.0.1'
    debug: bool = True

def main(cfg: DashboardConfig = tyro.cli(DashboardConfig)):
    global default_file
    default_file = cfg.file
    app.run_server(debug=cfg.debug, host=cfg.host, port=cfg.port)

if __name__ == '__main__':
    main()
