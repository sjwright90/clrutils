# contains the following functions:
# df_anynull
# get_max_max
# loadings_line_plot
# pca_plot
# pca_plot_old
# axis_limits
# color_bars
# factor_loading_plot
# pca_explore
# pca_loading_matrix
# loading_matrix

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from pandas import Categorical, DataFrame
from sklearn.decomposition import PCA
from clrutils import Lith_order
import seaborn as sns
import warnings


# quick test of any null/NaN values in df
def df_anynull(df):
    """
    Test is any NaN/null cells in a pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe

    Returns
    -----
    bool
        True if null present, false otherwise
    """

    return df.isnull().values.any()


# custom function to get max distance from origin
def get_max_max(x, y):
    """
    Returns absolute maximum of two lists

    Parameters
    ----------
    x : list-like, numeric
        First entity.

    y : list-like, numeric
        Second entity.

    Returns
    -----
    float or int
    """

    xmax = max(np.abs(x))
    ymax = max(np.abs(y))
    return max(xmax, ymax)


# define loadings_line_plot function
def loadings_line_plot(
    ldg_mat,
    pca1="PC1",
    pca2="PC2",
    alpha_lns=0.8,
    labels="metals",
    figsz=10,
    t_sz=11,
    color="y",
    bold=False,
) -> tuple:
    """
    Makes plot of labels loadings for principal components.

    Parameters
    ----------
    ldg_mat : pandas dataframe
        Contains PCA loadings for labels, rows are labels, columns are PCA
        loadings and labels.

    pca1 : str, default 'PCA1'
        Column in 'ldg_mat' with PCA1 values (x-axis).

    pca2 : str, default 'PCA2'
        Column in 'ldg_mat' with PCA2 values (y-axis).

    labels : str, default 'metals'
        Column in 'ldg_mat' with metal names

    alpha_lns : float(0-1), default 1
        Alpha value for metal loading lines.

    figsz : int, default 10

    t_size : int, default 11

    Returns
    -----
    fig
        Matplotlib.pyplot figure object

    ax
        Matplotlib.pyplot axis object
    """

    # make subplot
    fig, ax = plt.subplots(figsize=(figsz, figsz))
    # plot element loadings
    # matrix of 0s
    zros = np.zeros(len(ldg_mat))
    # plot lines from origin to points
    ax.plot([zros, ldg_mat[pca1]], [zros, ldg_mat[pca2]], color=color, alpha=alpha_lns)
    # add axis lines spanning the plot
    ax.axhline(0, color="k", linestyle="--", alpha=0.7)
    ax.axvline(0, color="k", linestyle="--", alpha=0.7)

    wgth = "normal"
    if bold:
        wgth = "bold"

    if not (pca1 in ldg_mat.columns) or not (pca2 in ldg_mat.columns):
        raise ValueError(f"Columns {pca1} and {pca2} must be in ldg_mat.")

    # set element labels
    for loc, lab in zip(ldg_mat[[pca1, pca2]].values, ldg_mat[labels]):
        ax.annotate(
            lab,
            xy=loc * 1.1,
            size=t_sz,
            xytext=loc,
            textcoords="offset points",
            va="center",
            ha="center",
            weight=wgth,
        )

    return fig, ax


def pca_plot(
    plot_df,
    ldg_mat,
    exp_var,
    hue_col,
    style_col,
    size_col,
    palette_dict,
    marker_dict,
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

    hue_col : str
        Column in 'plot_df' to use for color.

    style_col : str
        Column in 'plot_df' to use for style.

    size_col : str
        Column in 'plot_df' to use for size.

    palette_dict : dict
        Dictionary with color palette for 'hue_col'.
        Example: {'sandstone': 'orange', 'shale': 'blue'}
        Must correspond to unique values in 'hue_col'.

    marker_dict : dict
        Dictionary with markers for 'style_col'.
        Example: {'quartz': 'P', 'feldspar': 'o'}
        Must correspond to unique values in 'style_col'.

    **kwargs
        Additional keyword arguments to pass to the function.
        Options include:
        x : str, default 'PC1'
            Column in 'plot_df' to use for x-axis.
            Must match column in 'ldg_mat'.

        y : str, default 'PC2'
            Column in 'plot_df' to use for y-axis.
            Must match column in 'ldg_mat'.

        style_order : list, default None
            Order of style categories.

        sizes : tuple, default (200, 50)
            Tuple with sizes for 'size_order'.

        size_order : list, default None
            Order of size categories.

        alpha : float(0-1) or string, default 0.8
            Alpha value for data points. If string must be a column in 'plot_df' with values between 0 and 1.

        bold : bool, default True
            Whether to make loadings labels bold.

    Returns
    -----
    fig
        Matplotlib.pyplot figure object

    ax
        Matplotlib.pyplot axis object
    """
    # set default values for kwargs
    kwargs_dict = {
        "x": "PC1",
        "y": "PC2",
        "style_order": None,
        "sizes": (200, 50),
        "size_order": None,
        "alpha": 0.8,
        "bold": True,
    }
    # update kwargs_dict with kwargs
    kwargs_dict.update(kwargs)
    # unpack alpha
    if isinstance(kwargs_dict["alpha"], str):
        _alpha = plot_df[kwargs_dict["alpha"]]
    else:
        _alpha = kwargs_dict["alpha"]
    if not (0 <= _alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")

    if not (kwargs_dict["x"] in plot_df.columns) or not (
        kwargs_dict["y"] in plot_df.columns
    ):
        raise ValueError(
            f"Columns {kwargs_dict['x']} and {kwargs_dict['y']} must be in plot_df.\nConsider using 'x' and 'y' kwargs to set columns or rename columns in dataframe."
        )

    # plot loadings
    fig, ax = loadings_line_plot(
        ldg_mat,
        pca1=kwargs_dict["x"],
        pca2=kwargs_dict["x"],
        figsz=10,
        bold=kwargs_dict["bold"],
    )
    _ = ax.set_xlabel(f"PC1 ({exp_var[0]:.0%} variance)")
    _ = ax.set_ylabel(f"PC2 ({exp_var[1]:.0%} variance)")

    # use sns scatterplot to plot data points
    _ = sns.scatterplot(
        data=plot_df,
        x=kwargs_dict["x"],
        y=kwargs_dict["y"],
        hue=hue_col,
        palette=palette_dict,
        style=style_col,
        markers=marker_dict,
        style_order=kwargs_dict["style_order"],
        size=size_col,
        sizes=kwargs_dict["sizes"],  # needs to be inverse of size_order
        size_order=kwargs_dict[
            "size_order"
        ],  # first corresponds to sizes[1], second to sizes[0]
        alpha=_alpha,
        edgecolor="k",
    )
    _pc_x_l, _pc_x_u, _pc_y_l, _pc_y_u = axis_limits(
        plot_df[kwargs_dict["x"]], plot_df[kwargs_dict["y"]]
    )
    _ldg_x_l, _ldg_x_u, _ldg_y_l, _ldg_y_u = axis_limits(
        ldg_mat[kwargs_dict["x"]], ldg_mat[kwargs_dict["y"]]
    )
    _plot_lims = (
        min(_pc_x_l, _ldg_x_l),
        max(_pc_x_u, _ldg_x_u),
        min(_pc_y_l, _ldg_y_l),
        max(_pc_y_u, _ldg_y_u),
    )
    _ = ax.set_xlim(_plot_lims[0], _plot_lims[1])
    _ = ax.set_ylim(_plot_lims[2], _plot_lims[3])
    return fig, ax


# %%
def axis_limits(x, y):
    """
    Calculates x and y min and max for a square plot centered on the mean
    center of the data.

    Parameters
    ----------
    x : pandas series or list-like, 1-dimensional
        Values to be plotted along x.

    y : pandas series or list-like, 1-dimensional
        Values to be plotted along y.

    Returns
    -----
    xmin, xmax, ymin, ymax
        Limits of the plot.
    """
    xrange = np.abs((max(x) - min(x)))
    yrange = np.abs((max(y) - min(y)))
    xcenter = min(x) + (xrange / 2)
    ycenter = min(y) + (yrange / 2)

    diff = ((max([xrange, yrange])) / 2) * 1.05

    return xcenter - diff, xcenter + diff, ycenter - diff, ycenter + diff


def color_bars(axt, len_mat, first_break_2=False):
    """
    Creates filled horizontal color bars on a given axis

    Parameters
    ----------
    axt : pyplot axes object
        Contains data stored in Series. If data is a dict, argument order is
        maintained.

    len_mat : list like or numeric
        Defines length of the horizontal lines to be made,
        if numeric the number defines the terminal
        distance of the line, if list like the length of the list
        defines the terminal distance

    first_break_2 : bool, default False
        Whether to run color breaks from .2 up or from .4 up,
        if True breaks start at .2, else breaks start at .4,
        breaks move by steps of .2 and skip 1.0.

    Returns
    -----
    axt
        Pyplot axis object.
    """
    # set colors to fill, these are hardcoded for now
    # but we can change that if we want
    colors = ["skyblue", "green", "yellow", "orange"]

    # get horizontal line length, either with a numerical value
    # or the length of an input list
    if isinstance(len_mat, int) or isinstance(len_mat, float):
        x = np.arange(0, len_mat, 1)
    else:
        x = np.arange(0, len(len_mat), 1)

    # ones matrix of line length
    ones = np.ones(len(x))

    # fill between lines using breaks starting at .2 or .4
    # depending on user input
    if first_break_2:
        fill_mat = np.delete(np.arange(0.2, 1.21, 0.2), -2)
        # mirror fill across x axis
        for p in [1, -1]:
            for a, b, c in zip(p * fill_mat[:-1], p * fill_mat[1:], colors):
                axt.fill_between(x, ones * a, ones * b, facecolor=c, alpha=0.2)
    else:
        fill_mat = np.delete(np.arange(0.4, 1.41, 0.2), -3)
        for p in [1, -1]:
            for a, b, c in zip(p * fill_mat[:-1], p * fill_mat[1:], colors):
                axt.fill_between(x, ones * a, ones * b, facecolor=c, alpha=0.2)
    return axt


def factor_loading_plot(
    loading_matrix,
    sort_on="PC1",
    labels_col="metals",
    first_break_2=False,
    ttl="Principal Component 1 ",
):
    """
    Creates factor loading plot for principal components relative to metals

    Parameters
    ----------
    loading_matrix : pandas dataframe
        At minimum contains column of principal components and metals
        associated with those PCs

    sort_on : str, default 'PC1'
        Column name of principal components, loading_matrix will be sorted by
        this column

    labels_col : str, default 'metals'
        Column name of metals, or other labeled component associated with the
        PCs.

    first_break_2 : bool, default False
        Whether to run color breaks from .2 up or from .4 up, if True breaks
        start at .2, else breaks start at .4, breaks move by steps of .2 and
        skip 1.0.

    ttl : str, default 'Principal Component 1'
        Title for the plot

    Returns
    -----
    figx
        Matplotlib figure object.
    """

    # sort values by column of interest
    loading_matrix.sort_values(by=sort_on, inplace=True, ascending=False)
    # generate suplot for figure
    figx, axx = plt.subplots(1, figsize=(20, 8))
    # plot values of interest on subplot
    loading_matrix.plot(
        x=labels_col,
        y=sort_on,
        marker=".",
        ms=10,
        color="k",
        ax=axx,
        linewidth=2,
        legend=False,
    )

    # set xtick labels by indicated columns
    axx.set_xticks(range(len(loading_matrix)))
    axx.set_xticklabels(loading_matrix[labels_col], fontsize=15)
    axx.set_xlabel("Variable", weight="medium", fontsize=25, labelpad=15)

    # make y-ticks and label
    yticklabels = np.linspace(-2, 2, 21)
    yticklabels = np.around(yticklabels, decimals=1, out=None)
    axx.set_yticks(yticklabels)
    axx.set_yticklabels(yticklabels, fontsize=15)
    axx.set_ylabel("Factor Loading", weight="medium", fontsize=25, labelpad=15)

    # call function to fill plot colors
    axx = color_bars(axx, loading_matrix, first_break_2)

    # line along y=0
    axx.hlines(0, 0, len(loading_matrix) - 1, "k", linewidth=2, alpha=0.5)  # type: ignore

    # set x_lim by length of input matrix
    axx.set_xlim(left=-1, right=len(loading_matrix) + 1)

    # set y_lim by abs_val largest value in column being plotted
    ylim = (
        max(np.abs([loading_matrix[sort_on].min(), loading_matrix[sort_on].max()]))
        * 1.1
    )
    axx.set_ylim(bottom=-ylim, top=ylim)

    # title
    axx.set_title(
        ttl,
        weight="medium",
        size=25,
        x=0.01,
        y=1.05,
        horizontalalignment="left",
        verticalalignment="center",
    )
    # add grid lines
    axx.grid()

    # return the figure
    return figx


def pca_explore(
    df,
    prop_upto=0.8,
    ttl="Sulphurets explained variance\nvalues and cumulative, up to 80%",
    return_pca_obj=False,
):
    """
    Applies PCA to a df and returns plot of PCA explained variance and,
    optionally, the PCA object.

    Parameters
    ----------
    df : pandas dataframe
        Fully numeric dataframe to appy PCA to.

    prop_upto : float 0.0-1.0, default 0.8
        Cumulative proportion of variance to plot.

    ttl : str, default 'Sulphurets explained variance<br>values and
                        cumulative, up to 80%'
        Title for the plot.

    return_pca_obj : bool, default False
        Option to return PCA object if True function will return two objects,
        the figure first and the PCA object second.

    Returns
    -----
    figx [temp_pca]
        Matplotlib figure object.
        Optional - PCA object
    """
    # instansiate PCA object and fit to df
    temp_pca = PCA()
    temp_pca.fit(df)

    # objects for ploting
    figx, axx = plt.subplots()
    # extract explained variance
    exp_var = temp_pca.explained_variance_ratio_
    # get cumulative sum of explained variance
    cumu_var = np.cumsum(temp_pca.explained_variance_ratio_)

    # subset by proportion of variance explained
    upto = cumu_var[cumu_var < prop_upto]
    exp_var_upto = exp_var[: len(upto)]

    # plot
    axx.plot(upto)
    axx.bar(height=exp_var_upto, x=range(len(upto)))
    axx.set_title(ttl)
    axx.set_xlabel("PCA component")
    axx.set_ylabel("Proportion explained variance")

    # return figure and PCA obj or just figure
    if return_pca_obj:
        return figx, temp_pca
    else:
        return figx


def pca_loading_matrix(df, labels_list=None, exp_var_adj=False, n_components=6):
    """
    Performs PCA and calculates loading matrix.

    Parameters
    ----------
    df : pandas dataframe
        Fully numeric dataframe to appy PCA to.

    label_list : list-like, default None
        List of labels for loading matrix, if None the columns of df will be
        used.

    exp_var_adj : bool, default False
        Whether to adjust components by explained variance in loading matrix.

    n_components : int, default 6
        Number of PCA components.

    Returns
    -----
    temp_pca
        sklearn PCA object.

    df_pca
        numpy.array object of PCA transformed df

    ld_mat
        pd.dataframe format of loading matrix
    """

    # if labels list not given assign columns of df
    if labels_list is None:
        labels_list = df.columns

    # build PCA
    temp_pca = PCA(n_components=n_components)

    # transform
    df_pca = temp_pca.fit_transform(df)

    # call loading matrix function
    ld_mat = loading_matrix(temp_pca, labels_list, exp_var_adj)

    return temp_pca, df_pca, ld_mat


def loading_matrix(pca_obj, labels_list, exp_var_adj=False):
    """
    Calculates loading matrix from a PCA object.

    Parameters
    ----------
    pca_obj : sklearn PCA object
        Used to get PCA components.

    label_list : list-like
        Labels associated with PCA object must have #rows==#columns in PCA
        object.

    exp_var_adj : bool, default False
        Whether to adjust components by explained variance in loading matrix.

    Returns
    -----
    ld_mat
        pandas dataframe.
    """

    # if adjust by explained variance
    if exp_var_adj:
        mat = pca_obj.components_.T * np.sqrt(pca_obj.explained_variance_)
    else:
        mat = pca_obj.components_.T
    # build data frame
    ld_mat = DataFrame(
        mat,
        columns=[f"PC{x+1}" for x in range(pca_obj.components_.shape[0])],
        index=labels_list,
    )
    # add metals to data frame
    ld_mat["metals"] = labels_list
    return ld_mat
