import plotly.graph_objects as go
import numpy as np
import pandas as pd


def annotate_loadings(
    ldg_df: pd.DataFrame,
    plotly_fig: go.Figure | None = None,
    x_col: str = "PC1",
    y_col: str = "PC2",
    column_for_labels: str = "metals",
) -> go.Figure:
    """
    Annotate loadings on a plotly figure.
    Parameters
    ----------
    ldg_df : pd.DataFrame
        DataFrame containing loadings with columns for x and y coordinates and a column for the loading labels.
    plotly_fig : go.Figure, optional
        Existing plotly figure to annotate. If None, a new figure will be created. Default is None.
    x_col : str, optional
        Column name in ldg_df for x coordinates of loadings. Default is 'PC1'.
    y_col : str, optional
        Column name in ldg_df for y coordinates of loadings. Default is 'PC2
    Returns
    -------
    go.Figure
        Plotly figure with annotated loadings.
    """
    if plotly_fig is None:
        plotly_fig = go.Figure()
    for x, y, metal in zip(ldg_df[x_col], ldg_df[y_col], ldg_df[column_for_labels]):
        # Arrow annotation
        plotly_fig.add_annotation(
            x=x,
            y=y,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=0.8,
            arrowcolor="black",
        )

        # Text annotation slightly offset from the arrow tip
        direction = np.array([x, y]) / np.linalg.norm([x, y])
        offset_distance = 0.02
        text_offset = direction * offset_distance
        plotly_fig.add_annotation(
            x=x + text_offset[0],
            y=y + text_offset[1],
            xref="x",
            yref="y",
            text=metal,
            showarrow=False,
            font=dict(size=13, color="black"),
        )
    return plotly_fig
