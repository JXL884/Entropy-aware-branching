from pathlib import Path
import plotly.graph_objects as go
import numpy as np
import json
from datetime import datetime

from entropix.config import SamplerConfig, SamplerState
from entropix.model import Generation

def plot_sampler(generation_data: Generation, out: str | None):
    # Create a plotly figure with subplots
    fig = go.Figure()

    tokens = generation_data.tokens
    entropies = np.array([token_metrics.logits_entropy for token_metrics in generation_data.metrics])
    varentropies = np.array([token_metrics.logits_varentropy for token_metrics in generation_data.metrics])
    sampler_states = generation_data.sampler_states

    # Define colors for sampler states
    colors = {
        SamplerState.FLOWING: 'lightblue',
        SamplerState.TREADING: 'lightgreen',
        SamplerState.EXPLORING: 'orange',
        SamplerState.RESAMPLING: 'pink',
        SamplerState.ADAPTIVE: 'purple',
    }

    # Create unified hover text
    hover_template = ("Step: %{x}<br>" + "Value: %{y}<br>" + "Token: %{customdata[0]}<br>" + "State: %{customdata[1]}")

    # Add entropy trace
    fig.add_trace(
        go.Scatter(
            x=list(range(len(entropies))),
            y=entropies,
            name='Entropy',
            line=dict(color='blue'),
            yaxis='y1',
            customdata=list(zip(tokens if tokens else [''] * len(entropies), [state.value for state in sampler_states])),
            hovertemplate=hover_template
        )
    )

    # Add varentropy trace
    fig.add_trace(
        go.Scatter(
            x=list(range(len(varentropies))),
            y=varentropies,
            name='Varentropy',
            line=dict(color='red'),
            yaxis='y1',
            customdata=list(zip(tokens if tokens else [''] * len(varentropies), [state.value for state in sampler_states])),
            hovertemplate=hover_template
        )
    )

    # Create state indicators
    state_colors = [colors[state] for state in sampler_states]
    state_names = [state.value for state in sampler_states]

    # Add state indicators
    fig.add_trace(
        go.Scatter(
            x=list(range(len(sampler_states))),
            y=[0] * len(sampler_states),
            mode='markers',
            marker=dict(
                color=state_colors,
                size=20,
                symbol='square',
            ),
            customdata=list(zip(tokens if tokens else [''] * len(sampler_states), state_names)),
            hovertemplate=hover_template,
            yaxis='y2',
            showlegend=False,
        )
    )

    # Add state legend
    for state, color in colors.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    color=color,
                    size=10,
                    symbol='square',
                ),
                name=state.value,
                showlegend=True,
            )
        )

    # Update layout
    fig.update_layout(
        title='Entropy, Varentropy and Sampler States over Generation Steps',
        xaxis=dict(title='Generation Step', showticklabels=True, tickmode='linear', dtick=5),
        yaxis=dict(title='Value', domain=[0.25, 0.95]),
        yaxis2=dict(domain=[0.1, 0.2], showticklabels=False, range=[-0.5, 0.5]),
        height=750,
        showlegend=True,
        legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h")
    )

    # Add tokens
    formatted_text = ""
    line_length = 0
    max_line_length = 180  #some longer prompt overflow for some reason, keep it 270 for now

    for token, state in zip(tokens, sampler_states):
        color = colors[state]
        token_text = f"<span style='color: {color}'>{token}</span> "

        # Add newline if current line would be too long
        if line_length + len(token) > max_line_length:
            formatted_text += "<br>"
            line_length = 0

        formatted_text += token_text
        line_length += len(token) + 1  # +1 for the space

    # Add the text
    fig.add_annotation(
        text=formatted_text,
        xref="paper",
        yref="paper",
        x=0,
        y=0.07,
        showarrow=False,
        font=dict(size=20),
        align="left",
        xanchor="left",
        yanchor="top",
        xshift=5,
        yshift=0,
        bordercolor="gray",
        borderwidth=0,
    )

    num_lines = formatted_text.count('<br>') + 1
    bottom_margin = max(30, num_lines * 15)

    fig.update_layout(margin=dict(b=bottom_margin), yaxis=dict(domain=[0.25, 0.95]), yaxis2=dict(domain=[0.1, 0.2]))

    if not out:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"sampler_metrics_{timestamp}.html"
    elif not out.endswith(".html"):
        out += ".html"
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print(f"Sampler metrics visualization saved to {out}")

    return fig

def plot_entropy(generation_data: Generation, sampler_config: SamplerConfig, out: str | None):
    tokens = generation_data.tokens

    # Extract data
    entropies = np.array([token_metrics.logits_entropy for token_metrics in generation_data.metrics])
    varentropies = np.array([token_metrics.logits_varentropy for token_metrics in generation_data.metrics])
    attn_entropies = np.array([token_metrics.attn_entropy for token_metrics in generation_data.metrics])
    attn_varentropies = np.array([token_metrics.attn_varentropy for token_metrics in generation_data.metrics])

    # Ensure all arrays have the same length
    safe_length = min(len(entropies), len(varentropies), len(attn_entropies), len(attn_varentropies), len(tokens))
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
            name='Attention Entropy/Varentropy'
        )
    )

    # Calculate the limits for x, y, and z

    x_min, x_max = min(positions), max(positions)
    logits_y_min, logits_y_max = min(varentropies), max(varentropies)
    logits_z_min, logits_z_max = min(entropies), max(entropies)
    attention_y_min, attention_y_max = min(attn_varentropies), max(attn_varentropies)
    attention_z_min, attention_z_max = min(attn_entropies), max(attn_entropies)

    # logits_x_min, logits_x_max = min(entropies), max(entropies)
    # logits_y_min, logits_y_max = min(varentropies), max(varentropies)
    # attention_x_min, attention_x_max = min(attn_entropies), max(attn_entropies)
    # attention_y_min, attention_y_max = min(attn_varentropies), max(attn_varentropies)
    # z_min, z_max = min(positions), max(positions)

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
            # Add threshold planes

    thresholds = [
        (
            'logits_entropy', 'z', [
                (sampler_config.low_logits_entropy_threshold, 'rgba(255, 0, 0, 0.2)'),
                (sampler_config.medium_logits_entropy_threshold, 'rgba(0, 255, 0, 0.2)'),
                (sampler_config.high_logits_entropy_threshold, 'rgba(0, 0, 255, 0.2)'),
            ], 'logits'
        ),
        (
            'logits_varentropy', 'y', [
                (sampler_config.low_logits_varentropy_threshold, 'rgba(255, 165, 0, 0.2)'),
                (sampler_config.medium_logits_varentropy_threshold, 'rgba(165, 42, 42, 0.2)'),
                (sampler_config.high_logits_varentropy_threshold, 'rgba(128, 0, 128, 0.2)'),
            ], 'logits'
        ),
        (
            'attention_entropy', 'z', [
                (sampler_config.low_attention_entropy_threshold, 'rgba(255, 192, 203, 0.2)'),
                (sampler_config.medium_attention_entropy_threshold, 'rgba(0, 255, 255, 0.2)'),
                (sampler_config.high_attention_entropy_threshold, 'rgba(255, 255, 0, 0.2)'),
            ], 'attention'
        ),
        (
            'attention_varentropy',
            'y',
            [
                (sampler_config.low_attention_varentropy_threshold, 'rgba(70, 130, 180, 0.2)'),
                (sampler_config.medium_attention_varentropy_threshold, 'rgba(244, 164, 96, 0.2)'),
                (sampler_config.high_attention_varentropy_threshold, 'rgba(50, 205, 50, 0.2)'),
            ],
            'attention',
        ),
        (
            'attention_varentropy',
            'z',
            [
                (sampler_config.low_attention_varentropy_threshold, 'rgba(70, 130, 180, 0.2)'),
                (sampler_config.medium_attention_varentropy_threshold, 'rgba(244, 164, 96, 0.2)'),
                (sampler_config.high_attention_varentropy_threshold, 'rgba(50, 205, 50, 0.2)'),
            ],
            'attention',
        )
    ]

    for threshold_type, axis, threshold_list, data_type in thresholds:
        for threshold, color in threshold_list:
            fig.add_trace(create_threshold_plane(threshold, axis, color, f'{threshold_type.replace("_", " ").title()} Threshold: {threshold}', data_type))

    # Create buttons for toggling views
    buttons = [
        dict(label='Show All', method='update', args=[{'visible': [True] * len(fig.data)}]),  # type: ignore
        dict(label='Hide All', method='update', args=[{'visible': [True, True] + [False] * (len(fig.data) - 2)}]),  # type: ignore
        dict(label='Logits Only', method='update',
             args=[{'visible': [True, False] + [True if i < 6 else False for i in range(len(fig.data) - 2)]}]),  # type: ignore
        dict(label='Attention Only', method='update',
             args=[{'visible': [False, True] + [True if i >= 6 else False for i in range(len(fig.data) - 2)]}])  # type: ignore
    ]

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Token Position',
            yaxis_title='Varentropy',
            zaxis_title='Entropy',
            aspectmode='manual',
            aspectratio=dict(x=1, y=0.5, z=1),
            xaxis=dict(autorange='reversed'),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title='',
        updatemenus=[
            dict(type="buttons", direction="right", x=0.0, y=1.1, xanchor='left', yanchor='top', pad={"r": 10, "t": 10}, showactive=True, buttons=buttons)
        ],
        autosize=True,
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
    )

    if not out:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"entropy_plot_{timestamp}.html"
    elif not out.endswith(".html"):
        out += ".html"
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print(f"Entropy plot saved to {out}")

    return fig
