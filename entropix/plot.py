import logging
from typing import Sized
import plotly.graph_objects as go
import numpy as np

from entropix.config import SamplerState, STATE_COLOR_MAP
from entropix.model import GenerationData


def plot2d(generation_data: GenerationData, out: str | None = None, max_tokens: int = 100, show_labels=True):
    fig = go.Figure()

    tokens = generation_data.tokens
    entropies = np.array([token_metrics.logit_entropy for token_metrics in generation_data.metrics])
    varentropies = np.array([token_metrics.logit_varentropy for token_metrics in generation_data.metrics])
    sampler_states = generation_data.sampler_states

    thresholds = [
        # Entropy thresholds (blue)
        (generation_data.sampler_cfg.thresholds.logit_entropy.low, 'rgba(0,0,255,0.7)', 'Low Entropy'),
        (generation_data.sampler_cfg.thresholds.logit_entropy.medium, 'rgba(0,0,255,0.7)', 'Medium Entropy'),
        (generation_data.sampler_cfg.thresholds.logit_entropy.high, 'rgba(0,0,255,0.7)', 'High Entropy'),
        # Varentropy thresholds (red)
        (generation_data.sampler_cfg.thresholds.logit_varentropy.low, 'rgba(255,0,0,0.7)', 'Low Varentropy'),
        (generation_data.sampler_cfg.thresholds.logit_varentropy.medium, 'rgba(255,0,0,0.7)', 'Medium Varentropy'),
        (generation_data.sampler_cfg.thresholds.logit_varentropy.high, 'rgba(255,0,0,0.7)', 'High Varentropy'),
    ]
    for threshold, color, name in thresholds:
        fig.add_trace(
            go.Scatter(
                x=[0, len(tokens)],
                y=[threshold, threshold],
                mode='lines',
                line=dict(color=color, dash='dash', width=1),
                name=name,
                # visible='legendonly'  # Hidden by default
            )
        )

    # Main traces
    fig.add_trace(go.Scatter(name='Entropy', line=dict(color='blue'), x=list(range(len(entropies))), y=entropies, yaxis='y1'))
    fig.add_trace(go.Scatter(name='Varentropy', line=dict(color='red'), x=list(range(len(varentropies))), y=varentropies, yaxis='y1'))

    # Sampler states
    state_colors = [STATE_COLOR_MAP[state] for state in sampler_states]
    state_names = [state.value for state in sampler_states]
    fig.add_trace(
        go.Scatter(
            x=list(range(len(sampler_states))),
            y=[0] * len(sampler_states),
            mode='markers',
            marker=dict(color=state_colors, size=20, symbol='square'),
            customdata=list(zip(tokens if tokens else [''] * len(sampler_states), state_names)),
            hovertemplate='%{customdata[1]}<extra></extra>',
            yaxis='y2',
            showlegend=False
        )
    )
    # Custom legend
    for state, color in STATE_COLOR_MAP.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=color, size=10, symbol='square'), name=state.value, showlegend=True))

    # Update layout
    fig.update_layout(
        xaxis=dict(
            title='Token',
            showticklabels=show_labels,
            tickmode='array',
            ticktext=tokens,
            tickvals=list(range(len(tokens))),
            tickangle=45,
            range=[0, min(len(tokens), max_tokens)]
        ),
        yaxis=dict(title='Value', domain=[0.4, 0.95]),
        yaxis2=dict(domain=[0.1, 0.2], showticklabels=False, range=[-0.5, 0.5]),
        showlegend=True,
        legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
        hovermode='x',
    )

    if out:
        if not out.endswith(".html"): out += ".html"
        fig.write_html(out, include_plotlyjs=True, full_html=True)
        print(f"2D visualization saved to {out}")

    return fig

def plot3d(generation_data: GenerationData, out: str | None = None):
    tokens = generation_data.tokens

    # Extract data
    entropies = np.array([token_metrics.logit_entropy for token_metrics in generation_data.metrics])
    varentropies = np.array([token_metrics.logit_varentropy for token_metrics in generation_data.metrics])
    attn_entropies = np.array([token_metrics.attn_entropy for token_metrics in generation_data.metrics])
    attn_varentropies = np.array([token_metrics.attn_varentropy for token_metrics in generation_data.metrics])

    # Ensure all arrays have the same length
    safe_length = min(len(entropies), len(varentropies), len(attn_entropies), len(attn_varentropies), len(tokens))
    if safe_length == 0:
        logging.error("entropix.plot.plot3d: ERROR: missing metrics data")
        return
    entropies = entropies[:safe_length]
    varentropies = varentropies[:safe_length]
    attn_entropies = attn_entropies[:safe_length]
    attn_varentropies = attn_varentropies[:safe_length]
    tokens = tokens[:safe_length]

    positions = np.arange(safe_length)

    # Create hover text
    hover_text = [
        f"Token: {token or '<unk>'}<br>"
        f"Position: {i}<br>"
        f"Logits Entropy: {entropies[i]:.4f}<br>"
        f"Logits Varentropy: {varentropies[i]:.4f}<br>"
        f"Attention Entropy: {attn_entropies[i]:.4f}<br>"
        f"Attention Varentropy: {attn_varentropies[i]:.4f}" for i, token in enumerate(tokens)
    ]

    # Create the 3D scatter plot
    fig = go.Figure()

    # Create lines connecting dots based on entropy and varentropy
    entropy_lines = go.Scatter3d(
        x=positions,
        y=varentropies,
        z=entropies,
        mode='lines',
        line=dict(color='rgba(100,40,120,0.5)', width=2),
        name='Logits',
        visible=True,
    )
    varentropy_lines = go.Scatter3d(
        x=positions,
        y=attn_varentropies,
        z=attn_entropies,
        mode='lines',
        line=dict(color='rgba(255,10,10,0.5)', width=2),
        name='Attention',
        visible=False,
    )

    fig.add_trace(entropy_lines)
    fig.add_trace(varentropy_lines)

    # Add logits entropy/varentropy scatter
    fig.add_trace(
        go.Scatter3d(
            x=positions,
            y=varentropies,
            z=entropies,
            mode='markers',
            marker=dict(
                size=5,
                color=entropies,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Logits Entropy", x=0.85),
            ),
            text=hover_text,
            hoverinfo='text',
            name='Logits Entropy/Varentropy'
        )
    )

    # Add attention entropy/varentropy scatter
    fig.add_trace(
        go.Scatter3d(
            x=positions,
            y=attn_varentropies,
            z=attn_entropies,
            mode='markers',
            marker=dict(
                size=5,
                color=attn_entropies,
                colorscale='Plasma',
                opacity=0.8,
                colorbar=dict(title="Attention Entropy", x=1.0),
            ),
            text=hover_text,
            hoverinfo='text',
            name='Attention Entropy/Varentropy',
            visible=False,
        )
    )

    # Calculate the limits for x, y, and z
    x_min, x_max = min(positions), max(positions)
    logits_y_min, logits_y_max = min(varentropies), max(varentropies)
    logits_z_min, logits_z_max = min(entropies), max(entropies)
    attention_y_min, attention_y_max = min(attn_varentropies), max(attn_varentropies)
    attention_z_min, attention_z_max = min(attn_entropies), max(attn_entropies)

    # Function to create threshold planes
    def create_threshold_plane(threshold, axis, color, name, data_type):
        if data_type == 'logits':
            y_min, y_max = logits_y_min, logits_y_max
            z_min, z_max = logits_z_min, logits_z_max
        else:  # attention
            y_min, y_max = attention_y_min, attention_y_max
            z_min, z_max = attention_z_min, attention_z_max

        if axis == 'z':
            return go.Surface(
                x=[[x_min, x_max], [x_min, x_max]],
                y=[[y_min, y_min], [y_max, y_max]],
                z=[[threshold, threshold], [threshold, threshold]],
                colorscale=[[0, color], [1, color]],
                showscale=False,
                name=name,
                visible=False
            )
        elif axis == 'y':
            return go.Surface(
                x=[[x_min, x_max], [x_min, x_max]],
                y=[[threshold, threshold], [threshold, threshold]],
                z=[[z_min, z_min], [z_max, z_max]],
                colorscale=[[0, color], [1, color]],
                showscale=False,
                name=name,
                visible=False,
            )
        else:
            # Default case, return a dummy surface if axis is neither 'y' nor 'z'
            return go.Surface(
                x=[[x_min, x_max], [x_min, x_max]],
                y=[[y_min, y_min], [y_max, y_max]],
                z=[[z_min, z_min], [z_max, z_max]],
                colorscale=[[0, color], [1, color]],
                showscale=False,
                name=name,
                visible=False
            )

    thresholds = [
        (
            'logits_entropy', 'z', [
                (generation_data.sampler_cfg.thresholds.logit_entropy.low, 'rgba(255, 0, 0, 0.2)'),
                (generation_data.sampler_cfg.thresholds.logit_entropy.medium, 'rgba(0, 255, 0, 0.2)'),
                (generation_data.sampler_cfg.thresholds.logit_entropy.high, 'rgba(0, 0, 255, 0.2)'),
            ], 'logits'
        ),
        (
            'logits_varentropy', 'y', [
                (generation_data.sampler_cfg.thresholds.logit_varentropy.low, 'rgba(255, 165, 0, 0.2)'),
                (generation_data.sampler_cfg.thresholds.logit_varentropy.medium, 'rgba(165, 42, 42, 0.2)'),
                (generation_data.sampler_cfg.thresholds.logit_varentropy.high, 'rgba(128, 0, 128, 0.2)'),
            ], 'logits'
        ),
        (
            'attention_entropy', 'z', [
                (generation_data.sampler_cfg.thresholds.attn_entropy.low, 'rgba(255, 192, 203, 0.2)'),
                (generation_data.sampler_cfg.thresholds.attn_entropy.medium, 'rgba(0, 255, 255, 0.2)'),
                (generation_data.sampler_cfg.thresholds.attn_entropy.high, 'rgba(255, 255, 0, 0.2)'),
            ], 'attention'
        ),
        (
            'attention_varentropy',
            'y',
            [
                (generation_data.sampler_cfg.thresholds.attn_varentropy.low, 'rgba(70, 130, 180, 0.2)'),
                (generation_data.sampler_cfg.thresholds.attn_varentropy.medium, 'rgba(244, 164, 96, 0.2)'),
                (generation_data.sampler_cfg.thresholds.attn_varentropy.high, 'rgba(50, 205, 50, 0.2)'),
            ],
            'attention',
        ),
        # (
        #     'attention_varentropy',
        #     'z',
        #     [
        #         (generation_data.sampler_cfg.low_attention_varentropy_threshold, 'rgba(70, 130, 180, 0.2)'),
        #         (generation_data.sampler_cfg.medium_attention_varentropy_threshold, 'rgba(244, 164, 96, 0.2)'),
        #         (generation_data.sampler_cfg.high_attention_varentropy_threshold, 'rgba(50, 205, 50, 0.2)'),
        #     ],
        #     'attention',
        # )
    ]

    for threshold_type, axis, threshold_list, data_type in thresholds:
        for threshold, color in threshold_list:
            fig.add_trace(create_threshold_plane(threshold, axis, color, f'{threshold_type.replace("_", " ").title()} Threshold: {threshold}', data_type))

    assert isinstance(fig.data, Sized)
    fig.update_layout(
        scene=dict(
            xaxis_title='Token Position',
            yaxis_title='Varentropy',
            zaxis_title='Entropy',
            aspectmode='manual',
            aspectratio=dict(x=1, y=0.5, z=1),
            camera=dict(eye=dict(x=0.122, y=-1.528, z=1.528)),
            xaxis=dict(),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title='',
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                y=1.15,
                xanchor='left',
                yanchor='top',
                pad={"r": 10, "t": 10},
                buttons=[
                    dict(
                        label="Default View",
                        method="relayout",
                        args=[
                            {
                                "scene.camera": {"eye": {"x": 0.122, "y": -1.528, "z": 1.528}},
                                "scene.xaxis.title": "Token Position",
                                "scene.yaxis.title": "Varentropy",
                                "scene.zaxis.title": "Entropy",
                            }
                        ]
                    ),
                    dict(
                        label="Position vs Entropy",
                        method="relayout",
                        args=[
                            {
                                "scene.camera": {"eye": {"x": 0, "y": -2.5, "z": 0.1}},
                                "scene.xaxis.title": "Token Position",
                                "scene.yaxis.title": "",
                                "scene.zaxis.title": "Entropy",
                            }
                        ]
                    ),
                    dict(
                        label="Position vs Varentropy",
                        method="relayout",
                        args=[
                            {
                                "scene.camera": {"eye": {"x": 0, "y": 0.1, "z": 2.5}, "up": {"x": 0, "y": 1, "z": 0}},
                                "scene.xaxis.title": "Token Position",
                                "scene.yaxis.title": "Varentropy",
                                "scene.zaxis.title": "",
                            }
                        ]
                    ),
                    dict(
                        label="Entropy vs Varentropy",
                        method="relayout",
                        args=[
                            {
                                "scene.camera": {"eye": {"x": -2.5, "y": 0.1, "z": 0.1}, "up": {"x": 0, "y": 1, "z": 0}},
                                "scene.xaxis.title": "",
                                "scene.yaxis.title": "Varentropy",
                                "scene.zaxis.title": "Entropy",
                            }
                        ]
                    ),
                ],
            ),
        ],
        autosize=True,
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
    )

    if out:
        if not out.endswith(".html"):
            out += ".html"
        fig.write_html(out, include_plotlyjs=True, full_html=True)
        print(f"3D visualization saved to {out}")

    return fig
