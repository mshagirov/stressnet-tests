from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def plot_scatter(y_tgt, y_pred, x_names, phase, save_dir):
    plot_title = {'train': 'Training Data', 'val': 'Validation/Test data'}[phase]
    y_tgt = np.concatenate(y_tgt).ravel()
    y_pred = np.concatenate(y_pred).ravel()

    print(f"Plot: y_tgt {y_tgt.shape}; y_pred {y_pred.shape}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[-1.3, 1.3], y=[-1.3, 1.3],
        mode='lines',
        line=dict(color='gray', width=1, dash='dash'),
        opacity=0.5,
        showlegend=False,
    ))

    hover_texts = [
        f'{n}<br>(tgt={t:.3f}, pred={p:.3f})'
        for n, t, p in zip(x_names, y_tgt, y_pred)
    ]
    fig.add_trace(go.Scatter(
        x=y_tgt, y=y_pred,
        mode='markers',
        text=hover_texts,
        hoverinfo='text',
        marker=dict(size=6),
        name=phase,
    ))

    fig.update_xaxes(range=[-1.3, 1.3], title='y_true [a.u.]')
    fig.update_yaxes(range=[-1.3, 1.3], title='y_predicted [a.u.]',
                     scaleanchor='x', scaleratio=1)
    fig.update_layout(title=f'{plot_title}', width=600, height=600)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f'{phase}_scatter.html'
    fig.write_html(path)
    print(f'Saved: {path}')
