# %%
import pandas as pd
from pathlib import Path
from clrutils.pca_plots import loadings_line_plot, axis_limits
import matplotlib.cm as cm
from importlib import reload
from clrutils.pca_preprop import npr_to_bins
import string
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns

# %%

test_df = pd.DataFrame(
    {
        "PC1": np.random.randn(100),
        "PC2": np.random.randn(100),
        "hue_col": np.random.choice(["sandstone", "shale", "granite"], 100),
        "style_col": np.random.choice(["quartz", "feldspar", "mica"], 100),
        "size_col": np.random.choice(["small", "medium", "large"], 100),
        "alpha_col": np.random.rand(100),
    }
)
test_ldg_mat = pd.DataFrame(
    {
        "PC1": np.random.randn(10),
        "PC2": np.random.randn(10),
        "metals": list(string.ascii_uppercase[:10]),
    }
)


# %%
def pca_plot(
    plot_df,
    ldg_mat,
    exp_var,
    x,
    y,
    alpha=1,
    bold=True,
    **kwargs,
) -> tuple:
    """
    Makes a PCA plot with loadings and data points.

    Parameters
    ----------
    plot_df : pandas dataframe
        Contains data to be plotted.

    ldg_mat : pandas dataframe
        Contains PCA loadings for metals, rows are metals, columns are PCA
        loadings and labels.

    exp_var : list
        List of floats with explained variance for PC1 and PC2.

    x : str
        Column in 'plot_df' to use for x-axis.
        Must match column in 'ldg_mat'.

    y : str
        Column in 'plot_df' to use for y-axis.
        Must match column in 'ldg_mat'.

    **kwargs
        Additional keyword arguments to pass to seaborn.scatterplot.
        Use documentation for seaborn.scatterplot to see available options.
        Common options include 'hue', 'size', 'style', 'palette', 'markers',
        'hue_order', 'style_order', 'sizes', 'size_order'.

    Returns
    -----
    fig
        Matplotlib.pyplot figure object

    ax
        Matplotlib.pyplot axis object
    """
    # set default values for kwargs
    # unpack alpha
    if isinstance(alpha, str):
        _alpha = plot_df[alpha]
    else:
        _alpha = alpha

    if not x in ldg_mat.columns and not y in ldg_mat.columns:
        raise ValueError(f"{x} and {y} must be columns in 'ldg_mat'.")
    if not x in plot_df.columns and not y in plot_df.columns:
        raise ValueError(f"{x} and {y} must be columns in 'plot_df'.")
    # plot loadings
    fig, ax = loadings_line_plot(
        ldg_mat,
        pca1=x,
        pca2=y,
        figsz=10,
        bold=bold,
    )
    _ = ax.set_xlabel(f"PC1 ({exp_var[0]:.0%} variance)")
    _ = ax.set_ylabel(f"PC2 ({exp_var[1]:.0%} variance)")

    # Clean kwargs
    _kwargs = {}
    _valid_kwargs = sns.scatterplot.__code__.co_varnames
    for key, value in kwargs.items():
        if not key in _valid_kwargs:
            # drop from kwargs and notify user
            print(f"'{key}' not a valid argument for sns.scatterplot.\nDropping {key}.")
        else:
            _kwargs[key] = value

    # use sns scatterplot to plot data points
    _ = sns.scatterplot(
        data=plot_df,
        x=x,
        y=y,
        alpha=_alpha,
        ax=ax,
        **_kwargs,
    )
    _pc_x_l, _pc_x_u, _pc_y_l, _pc_y_u = axis_limits(plot_df[x], plot_df[y])
    _ldg_x_l, _ldg_x_u, _ldg_y_l, _ldg_y_u = axis_limits(ldg_mat[x], ldg_mat[y])
    _plot_lims = (
        min(_pc_x_l, _ldg_x_l),
        max(_pc_x_u, _ldg_x_u),
        min(_pc_y_l, _ldg_y_l),
        max(_pc_y_u, _ldg_y_u),
    )
    _ = ax.set_xlim(_plot_lims[0] * 1.1, _plot_lims[1] * 1.1)
    _ = ax.set_ylim(_plot_lims[2] * 1.1, _plot_lims[3] * 1.1)
    return fig, ax


# %%
palette_dict = {"sandstone": "orange", "shale": "blue", "granite": "green"}
marker_dict = {"quartz": "P", "feldspar": "o", "mica": "s"}
sizes = (300, 50)
size_order = ["small", "medium", "large"]
test_df["size_col"] = ["small"] * 80 + ["medium"] * 10 + ["large"] * 10
style_order = ["quartz", "feldspar", "mica"]
np.random.shuffle(style_order)
hue_order = ["sandstone", "shale", "granite"]
alpha = "alpha_col"
bold = True

fig, ax = pca_plot(
    plot_df=test_df,
    ldg_mat=test_ldg_mat,
    exp_var=[0.5, 0.3],
    x="PC1",
    y="PC2",
    hue="hue_col",
    size="size_col",
    style="style_col",
    palette=palette_dict,
    markers=marker_dict,
    hue_order=hue_order,
    style_order=style_order,
    sizes=sizes,
    size_order=size_order,
    new_kwarg="new_kwarg",
)
# %%
np.random.shuffle(["quartz", "feldspar", "mica"])
