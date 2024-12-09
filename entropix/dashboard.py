import base64
import json
import queue
from dataclasses import dataclass
from datetime import datetime
from threading import Thread

import dash
import dash_bootstrap_components as dbc
import tyro
from dash import (
    ALL,
    MATCH,
    Input,
    Output,
    State,
    callback,
    dcc,
    html,
)

from entropix.config import SamplerConfig, STATE_COLOR_MAP
from entropix.model import GenerationData, Model, load_weights, stream
from entropix.models.llama import LLAMA_1B
from entropix.plot import plot2d, plot3d
from entropix.tokenizer import Message, Tokenizer

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    [
        dcc.Store(id='stored-data', storage_type='local'),
        dcc.Store(id='generation-state', data={'generating': False}),
        dcc.Interval(id='generation-interval', interval=100, disabled=True),
        dcc.Store(id='generation-complete-trigger', data=datetime.now().isoformat()),
        dbc.Container(
            [
                dbc.Modal(
                    [
                        dbc.ModalHeader("Save Data"),
                        dbc.ModalBody([dbc.Input(id="filename-input", placeholder="Enter filename", type="text")]),
                        dbc.ModalFooter(
                            [
                                dbc.Button("Save", id="confirm-save", className="ms-auto", color="success"),
                                dbc.Button("Cancel", id="cancel-save", className="ms-2", color="secondary"),
                            ]
                        ),
                    ],
                    id="save-modal",
                    is_open=False,
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div(["Drag and drop or click to load a data file"]),
                                    style={
                                        "width": "100%", "height": "60px", "lineHeight": "60px", "borderWidth": "1px", "borderStyle": "dashed",
                                        "borderRadius": "5px", "textAlign": "center", "margin": "10px"
                                    },
                                    multiple=False
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button("Show/Hide Messages", id="collapse-button", className="mb-3", color="primary"),
                                dbc.Collapse(
                                    [
                                        dbc.ButtonGroup(
                                            [
                                                dbc.Button("Generate Response", id="generate-button", style={'display': 'none'}),
                                                dbc.Button("Save Data", id="save-data-button", style={'display': 'none'}),
                                            ],
                                            id="button-group",
                                            style={'display': 'none'}
                                        ),
                                        html.Div(id="messages-container"),
                                    ],
                                    id="collapse",
                                    is_open=False
                                ),
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
                                    "Click and drag the x-axis labels to scroll the plot. Click the legend to show/hide entropy and varentropy traces.",
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
                    [dbc.Col([dbc.Checklist(options=[{"label": "Show tokens on x-axis", "value": True}], value=[True], id="show-labels-toggle", switch=True)])]
                ),
                dbc.Row([
                    dbc.Col([dcc.Graph(id="sampler-plot", style={"height": "600px"})], width=12),
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
                                            switch=True
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
                    style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px'}
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(
                                    id="entropy-plot",
                                    style={
                                        "height": "60vw",
                                        "margin": "0 5% 0 5%",
                                        "padding": "0 5px 0 5px",
                                        "borderLeft": "1px solid #ddd",
                                        "borderRight": "1px solid #ddd",
                                    }
                                ),
                            ],
                            width=12,
                        ),
                    ]
                ),
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
    [Output({"type": "message-text", "role": "assistant", "index": MATCH}, "className"),
     Output({"type": "edit-button", "index": MATCH}, "children")],
    Input({"type": "edit-button", "index": MATCH}, "n_clicks"),
    prevent_initial_call=True
)
def toggle_edit_mode(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return "", "Done"
    return "d-none", "Edit"

@callback(
    Output("save-modal", "is_open"),
    [Input("save-data-button", "n_clicks"), Input("confirm-save", "n_clicks"),
     Input("cancel-save", "n_clicks")], [State("save-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_save_modal(save_clicks, confirm_clicks, cancel_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Only respond to actual button clicks
    if trigger_id == "save-data-button" and ctx.triggered[0]["value"]:
        return True
    elif trigger_id in ["confirm-save", "cancel-save"] and ctx.triggered[0]["value"]:
        return False

    raise dash.exceptions.PreventUpdate

@callback(
    Output("save-data-button", "children"),
    [Input("confirm-save", "n_clicks")],
    [State("filename-input", "value"), State("stored-data", "data")],
    prevent_initial_call=True,
)
def save_data(n_clicks, filename, stored_data):
    if not n_clicks or not stored_data or not filename:
        raise dash.exceptions.PreventUpdate

    if not filename.endswith('.json'):
        filename += '.json'

    gen_data = load_data_from_contents(stored_data)
    with open(filename, 'w') as f:
        json.dump(gen_data.to_dict(), f, indent=2)

    return "Saved!"

@callback(
    Output("messages-container", "children"),
    [Input("stored-data", "data")],
    [State("messages-container", "children")],
)
def update_messages(data, current_messages):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    if not data and not current_messages:
        return []

    gen_data = load_data_from_contents(data)
    message_boxes = []

    message_boxes.append(
        dbc.ButtonGroup(
            [
                dbc.Button(
                    "Generate Response",
                    id="generate-button",
                    color="info",
                ),
                dbc.Button(
                    "Save Data",
                    id="save-data-button",
                    color="success",
                    className="ms-1",
                ),
            ],
            className="mb-3",
        )
    )

    for i, msg in enumerate(gen_data.messages[:-1]):
        content = dcc.Textarea(
            value=msg.content,
            id={'type': 'message-text', 'role': msg.role, 'index': i},
            style={'width': '100%', 'height': '100px'},
            readOnly=False,
        )
        message_boxes.append(dbc.Card([dbc.CardHeader(msg.role), dbc.CardBody(content)], className="mb-3"))

    msg = gen_data.messages[-1]
    edit_button = dbc.Button("Edit", id={"type": "edit-button", "index": len(message_boxes)}, size="sm", className="mb-2")
    tokens_html = []
    for token, state in zip(gen_data.tokens, gen_data.sampler_states):
        color = STATE_COLOR_MAP[state]
        tokens_html.append(html.Span(token, style={'color': color}))
    display_div = html.Div(tokens_html, style={'whiteSpace': 'pre-wrap'})
    edit_textarea = dcc.Textarea(
        value=msg.content,
        id={'type': 'message-text', 'role': msg.role, 'index': len(message_boxes)},
        style={'width': '100%', 'height': '100px'},
        className="d-none",
    )
    content = html.Div([edit_button, display_div, edit_textarea], style={'whiteSpace': 'pre-wrap'})
    message_boxes.append(dbc.Card([dbc.CardHeader(msg.role), dbc.CardBody(content)], className="mb-3"))

    return message_boxes

def background_generate(messages, model, sampler_cfg, queue):
    response = ""
    for token, metrics, state, generation_data in stream(messages, model, sampler_cfg):
        if token:  # Only process non-empty tokens
            response += token
            queue.put((token, metrics, state, response, generation_data))
        # if generation_data: pass
    queue.put(None)  # Signal completion

@callback(
    [Output("stored-data", "data", allow_duplicate=True),
     Output("generation-state", "data"),
     Output("generation-interval", "disabled")],
    Input("generate-button", "n_clicks"), [
        State({'type': 'message-text', 'role': ALL, 'index': ALL}, 'value'),
        State({'type': 'message-text', 'role': ALL, 'index': ALL}, 'id'),
        State("stored-data", "data")
    ],
    prevent_initial_call=True
)
def start_generation(n_clicks, message_contents, message_ids, stored_data):
    if not n_clicks:
        return dash.no_update

    print("starting generation")

    messages = [{"role": id_dict["role"], "content": content} for content, id_dict in zip(message_contents, message_ids)]
    messages = [Message(**m) for m in messages]
    messages.append(Message(role="assistant", content=""))

    print("\nMESSAGES\n")
    for m in messages:
        print(m.role)
        print(m.content)
        print()

    sampler_cfg = SamplerConfig()
    model_params = LLAMA_1B
    tokenizer_path = f"weights/tokenizers/{model_params.name}.json"
    tokenizer = Tokenizer(tokenizer_path)
    weights_path = f"weights/{model_params.name}"
    weights = load_weights(weights_path, model_params)
    model = Model(weights, model_params, tokenizer)

    # Initialize generation data with all required fields
    gen_data = GenerationData(
        prompt=tokenizer.apply_chat_template(messages),
        response="",
        tokens=[],
        messages=messages,
        branches=[],
        metrics=[],
        sampler_cfg=sampler_cfg,
        sampler_states=[],
    )

    json_str = json.dumps(gen_data.to_dict())
    initial_data = 'data:application/json;base64,' + base64.b64encode(json_str.encode()).decode()

    q = queue.Queue()
    Thread(target=background_generate, args=(messages, model, sampler_cfg, q), daemon=True).start()

    app.queue = q  # type: ignore

    return initial_data, {'generating': True}, False

@callback(
    [
        Output("stored-data", "data", allow_duplicate=True),
        Output("generation-state", "data", allow_duplicate=True),
        Output("generation-interval", "disabled", allow_duplicate=True),
        Output("generation-complete-trigger", "data", allow_duplicate=True),
    ],
    Input("generation-interval", "n_intervals"),
    [State("generation-state", "data"), State("stored-data", "data")],
    prevent_initial_call=True,
)
def update_generation(n_intervals, state, stored_data):
    if not state['generating']:
        return dash.no_update

    print("update_generation")
    gen_data = load_data_from_contents(stored_data)
    new_data = "data:application/json;base64,"

    try:
        result = app.queue.get_nowait()  # type: ignore

        if result is None:  # Generation complete
            print("generation complete, using stored data")
            return stored_data, {'generating': False}, True, datetime.now().isoformat()

        token, metrics, sampler_state, response, complete_gen_data = result
        if not complete_gen_data:
            gen_data.tokens.append(token)
            gen_data.metrics.append(metrics)
            gen_data.sampler_states.append(sampler_state)
            gen_data.response = response
            gen_data.messages[-1].content = response
        else:
            gen_data = complete_gen_data

        json_str = json.dumps(gen_data.to_dict())
        new_data = 'data:application/json;base64,' + base64.b64encode(json_str.encode()).decode()

        print("\nMESSAGES\n")
        for m in gen_data.messages:
            print(m.role)
            print(m.content)
            print()

        return new_data, state, False, dash.no_update
    except queue.Empty:
        print("queue empty")
        return dash.no_update

def load_data_from_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    gen_data = GenerationData.from_dict(json.loads(decoded.decode('utf-8')))
    return gen_data

@callback(Output("stored-data", "data", allow_duplicate=True), Input("upload-data", "contents"), prevent_initial_call=True)
def update_output(contents):
    if contents is None:
        raise dash.exceptions.PreventUpdate

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'json' in content_type:
            # Assume this is a JSON file containing GenerationData
            data = json.loads(decoded)
            gen_data = GenerationData.from_dict(data)
        else:
            # For other file types, you might want to implement specific parsing logic
            raise ValueError("Unsupported file type")

        # Convert the GenerationData back to JSON and encode it
        json_str = json.dumps(gen_data.to_dict())
        new_data = 'data:application/json;base64,' + base64.b64encode(json_str.encode()).decode()
        return new_data

    except Exception as e:
        print(f"Error processing file: {e!s}")
        raise dash.exceptions.PreventUpdate

@callback(
    [
        Output("sampler-plot", "figure"),
        Output("entropy-plot", "figure"),
    ],
    [
        Input("max-tokens-slider", "value"),
        Input("show-labels-toggle", "value"),
        Input("logits-controls", "value"),
        Input("attention-controls", "value"),
        Input("stored-data", "data")
    ],
    [
        State("generation-state", "data"),
    ],
)
def update_plots(
    max_tokens,
    show_labels,
    logits_controls,
    attn_controls,
    # contents,
    stored_data,
    generation_state,
):
    if stored_data is not None:
        file_data = stored_data
    else:
        return dash.no_update, dash.no_update

    generation_data = load_data_from_contents(file_data)
    if len(generation_data.metrics) == 0:
        return dash.no_update, dash.no_update

    max_tokens = float('inf') if max_tokens == 300 else max_tokens
    fig2d = plot2d(generation_data, max_tokens=max_tokens, show_labels=bool(show_labels))  # type: ignore
    fig3d = plot3d(generation_data)
    if not fig2d or not fig3d:
        return dash.no_update, dash.no_update

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
        if i < len(fig3d.data):  # type: ignore
            fig3d.data[i].visible = vis  # type: ignore

    return fig2d, fig3d

@dataclass
class DashboardConfig:
    file: str | None = None
    port: int = 8050
    host: str = '127.0.0.1'
    debug: bool = True

def main(cfg: DashboardConfig = tyro.cli(DashboardConfig)):
    app.run_server(debug=cfg.debug, host=cfg.host, port=cfg.port)

if __name__ == '__main__':
    main()
