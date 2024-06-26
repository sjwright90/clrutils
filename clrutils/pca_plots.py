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


def loadings_line_plot(
    pca_df,
    pca1="PC1",
    pca2="PC2",
    alpha_lns=0.8,
    metals="metals",
    figsz=10,
    t_sz=11,
    bold=False,
):
    """
    Makes plot of metals loadings for principal components.

    Parameters
    ----------
    pca_df : pandas dataframe
        Contains PCA loadings for metals, rows are metals, columns are PCA
        loadings and labels.

    pca1 : str, default 'PCA1'
        Column in 'pca_df' with PCA1 values (x-axis).

    pca2 : str, default 'PCA2'
        Column in 'pca_df' with PCA2 values (y-axis).

    metals : str, default 'metals'
        Column in 'pca_df' with metal names

    alpha_lns : float(0-1), default 1
        Alpha value for metal loading lines.

    figsz : int, default 10

    t_size : int, default 11

    Returns
    -----
    figt
        Matplotlib.pyplot figure object

    axt
        Matplotlib.pyplot axis object
    """

    # make subplot
    figt, axt = plt.subplots(figsize=(figsz, figsz))
    # plot element loadings
    # matrix of 0s
    zros = np.zeros(len(pca_df))
    # plot lines from origin to points
    axt.plot([zros, pca_df[pca1]], [zros, pca_df[pca2]], color="y", alpha=alpha_lns)
    # add axis lines
    axt.hlines([0], [-1.5], [1.5], colors="k", linestyles="--", alpha=0.6)  # type: ignore
    axt.vlines([0], [-1.5], [1.5], colors="k", linestyles="--", alpha=0.6)  # type: ignore

    wgth = "normal"
    if bold:
        wgth = "bold"

    # set element labels
    for loc, lab in zip(pca_df[[pca1, pca2]].values, pca_df[metals]):
        axt.annotate(
            lab,
            xy=loc * 1.1,
            size=t_sz,
            xytext=loc,
            textcoords="offset points",
            va="center",
            ha="center",
            weight=wgth,
        )

    return figt, axt


# Edge color not working for the exploration plots, not sure why, the exp plots also have other problems
# cannot reproduce the problem, but keep an eye out for it
def pca_plot(
    df,
    pca_df,
    pca_obj,
    pca1="PC1",
    pca2="PC2",
    metals="metals",
    lith="lithology_relog",
    npr_size="npr_size",
    cmapin=cm.turbo,  # type: ignore
    pca1a="PC1",
    pca2a="PC2",
    alpha_sct=1.0,
    alpha_lns=0.8,
    lith_order_in=None,
    title="PCA Bi-Plot",
    x_axispca=1,
    y_axispca=2,
    plot_npr=True,
    edgecolor=True,
    loading_lines=True,
    tpbbx=0.985,
    tpbby=1.04,
    btbbx=0.985,
    btbby=0,
    topledgettl="Lithology",
    bottomledgettl="NPR",
    btmldglbls=None,
    bold=False,
    thrdbby=0.1,
    thrdbbx=0.985,
    l3_title="Deposit",
    **kwargs,
):
    """
    Makes PCA bi-plot with metals loading, this is a rebuild using
    seaborn instead of matplotlib, experimental version

    Parameters
    ----------
    df  pandas dataframe
        At minimum has columns for sample lithology, NPR sizes,
        and 2 principal components.

    pca_df : pandas dataframe | None
        Contains PCA loadings for metals, structured as follows:
        rows are metals, columns are PCA loadings and labels
        to be passed to loadings_line_plot.
        If loading_lines=False, set 'pca_df=None'.

    pca_obj : array-like, with len>=2
        The percentage of variance explained by the PCA values being plotted.

    pca1 : str, default 'PCA1', optional
        Name of column in pca_df with PCA1 values (x-axis)
        to be passed to loadings_line_plot.
        If loading_lines=False, this variable will be ignored.

    pca2 : str, default 'PCA2', optional
        Name of column in pca_df with PCA2 values (y-axis)
        to be passed to loadings_line_plot.
        If loading_lines=False, this variable will be ignored.

    metals : st, default 'metals', optional
        Name of column in pca_df with metal names for labeling,
        to be passed to loadings_line_plot.
        If loading_lines=False, this variable will be ignored.

    lith : str, default 'Lithology'
        Name of column in 'df' with the lithologies.

    npr_size : str or int or list like len df, default 'npr_size', optional
        Name of column in df with NPR size categories,
        the values in this column will correspond directly to the size of
        the plotted points, column needs to be numerical.

    cmapin = matplotlib colormap object, default cm.turbo
        Colormap to be used for coloring the points by lithology.

    pca1a : str, default 'PCA1'
        Name of column in df with PCA1 (x-axis).

    pca2a : str, default 'PCA2'
        Name of column in df with PCA2 (y-axis).

    alpha_sct : float(0-1) or str, default 1
        If float, alpha value for scatterplot points
        If str, column in df to use for alpha values, the column
        must have numerical values (0-1).

    alpha_lns : float(0-1), default 1
        Alpha value for metal loading lines, passed to loadings_line_plot.
        If loading_lines=False this variable is ignored.

    lith_order_in : list-like strings, default Lith_order
        The order in which to list the lithologies in the legend.

    title : str, default 'PCA Bi-Plot'
        Title to give the plot.

    x_axispca : int, default 1
        Index+1 location in pca_obj of variance explained for PCA1.
        Must be less than len(pca_obj).

    y_axispca : int, default 2
        Index+1 location in pca_obj of variance explained for PCA2.
        Must be less than len(pca_obj).

    plot_npr : bool, default True
        Whether to include NPR size legend

    edgecolor : bool, default True
        Use edgecolor on markers in plot, if True edgecolor='k'.

    loading_lines : bool, default True
        Whether to call loadings_line_plot() function.
        If False set pca_df=None

    tpbbx : float
        x of bounding box to anchor for main legend

    tpbby : float
        y of bounding box to anchor for main legend

    btbbx : float
        x of bounding box to anchor for NPR legend

    btbby : float
        y of bounding box to anchor for NPR legend

    topledgettl : str
        Title for top legend

    bottomledgettl : str
        Title for bottom legend

    btmldglbls : list-like strings
        Labels for bottom legend

    bold : bool, default False
        Whether to bold the metal labels on the loading lines

    thrdbby : float
        y of bounding box to anchor for third legend

    thrdbbx : float
        x of bounding box to anchor for third legend

    l3_title : str
        Title for third legend, if using

    **kwargs : dict
        keywords, pretty much reserved for style and markers to pass to sns,
        style is the column in df to use for style, markers is the list of
        markers to use for each style. will likely end up with a shitty legend so have fun with that.
        Also have to pass "style_order" to get the legend to work, this is the order of the styles,
        take it from the column you are using for style, but can be any order you want.

        an example would look like this:
            params = {
                "style": "deposit",
                "markers": ["P", "*"],
                "style_order": sorted(plot_df["deposit"].unique(), reverse=True),
                "style_title": "Deposit",
            }

    Returns
    -----
    figt
        matplotlib.pyplot figure

    axt
        matplotlib.pyplot axes
    """
    print(
        "This is an experimental version of pca_plot() if it breaks, let me know and use pca_plot_old() instead"
    )

    if btmldglbls is None:
        btmldglbls = ["NPR<0.2", "0.2<NPR<3", "NPR>3"]

    if pca_df is None:  # automatically set loading_lines to False if pca_df is None
        loading_lines = False

    # call line plot function
    if loading_lines:
        figt, axt = loadings_line_plot(
            pca_df=pca_df,
            pca1=pca1,
            pca2=pca2,
            alpha_lns=alpha_lns,
            metals=metals,
            bold=bold,
        )
    else:
        figt, axt = plt.subplots(figsize=(10, 10))

    # make copy of df to avoid altering original
    temp = df.copy()

    temp.dropna(subset=[lith], inplace=True)  # drop NaNs in lith column

    temp.reset_index(inplace=True, drop=True)

    if lith_order_in is None:
        lith_order_in = Lith_order

    unique_lith_in = df[lith].unique()

    lith_present = [l for l in lith_order_in if l in unique_lith_in]

    if sorted(unique_lith_in) != sorted(
        lith_present
    ):  # NaNs break this find a solution
        warnings.warn(
            "Lithologies in sample not present in chosen lith order, appending to end"
        )
        lith_present = lith_present + [
            l for l in unique_lith_in if l not in lith_present
        ]

    # get color range
    colors = np.linspace(0, 1, len(unique_lith_in))

    # dictionary to map lithologies to unique color spectrum
    color_dict = dict(zip(lith_present, colors))

    # map lithologies to new 'color' column
    # apply cmap to turn linspace into rgb color from matplotlib cm
    # if cmap is a dictionary, use it to map colors, else use cmap as is
    if isinstance(cmapin, dict):
        print("using cmap as dictionary")
        temp["color"] = temp[lith].replace(cmapin)
    # else if cmap is a pyplot colormap, use it to map colors
    elif isinstance(cmapin, cm.colors.Colormap):
        temp["color"] = temp[lith].replace(color_dict).apply(cmapin)
        cmapin = cm.colors.ListedColormap([cmapin(i) for i in colors])
        # make cmapin a dictionary for use in legend
        cmapin = {l: cmapin(i) for i, l in enumerate(lith_present)}
    # else print warning and use turbo
    else:
        warnings.warn("Colormap not recognized, using matplotlib.cm.turbo as default")
        temp["color"] = temp[lith].replace(color_dict).apply(cm.turbo)
        # make cmapin a dictionary for use in legend
        cmapin = {l: cm.turbo(i) for i, l in enumerate(lith_present)}

    # make lithology a categorical column
    # have to do second due to multi-index problems
    # when using 'replace' function on lith column
    temp[lith] = Categorical(temp[lith], categories=lith_present, ordered=True)
    # sort on lith column
    temp.sort_values(by=lith, inplace=True)

    temp.reset_index(inplace=True, drop=True)

    if isinstance(alpha_sct, str):
        alpha_sct = temp[alpha_sct]

    if edgecolor:
        edg = "k"
    else:
        edg = None

    if isinstance(npr_size, str):
        npr_size = temp[npr_size]

    params = {
        "alpha": alpha_sct,
        "s": npr_size,
        "edgecolor": edg,
        "c": temp["color"],
        "facecolors": "none",
    }
    for key, value in kwargs.items():
        params[key] = value

    _ = sns.scatterplot(
        data=temp,
        x=pca1a,
        y=pca2a,
        ax=axt,
        **params,
    )

    # calc plot limits
    xmin, xmax, ymin, ymax = axis_limits(df[pca1a], df[pca2a])

    # set plot limits to make square
    axt.set_xlim(xmin, xmax)
    axt.set_ylim(ymin, ymax)

    if plot_npr:
        # npr labels
        npr_labels = btmldglbls
        # size of npr circles in legend (different from plot because
        # they are built on different scales)
        # plot_sizes = sorted(set(npr_size))
        # legend_sizes = [np.sqrt(s) for s in plot_sizes]
        # CLOSE BUT NOT QUITE, STILL USING MANUAL SIZES
        legend_sizes = np.linspace(7, 16, len(btmldglbls))
        # make legend handles for NPR size legend
        # number of 'o' markers equal to number of sizes
        leg_1_handles = [
            Line2D(
                [],
                [],
                markersize=size,
                marker="o",
                color="w",
                markerfacecolor="grey",
                label=label,
            )
            for size, label in zip(legend_sizes, npr_labels)
        ]
        # construct first legend
        # locate in bottom right
        first_legend = plt.legend(
            handles=leg_1_handles,
            loc="lower left",
            bbox_to_anchor=(btbbx, btbby),
            borderaxespad=1.0,
            frameon=False,
        )

        # set first legend title
        first_legend.set_title(bottomledgettl, prop={"size": 15})  # type: ignore
        # add first legend as artist
        axt.add_artist(first_legend)
    else:
        pass
    if "style" and "markers" and "style_order" in params.keys():
        print("making 3rd legend")
        # build second legend
        # markers of color associated with
        # lithologies in the plot
        leg_3_handles = [
            Line2D(
                [],
                [],
                markersize=10,
                marker=b,
                color="white",
                label=a,
                markerfacecolor="black",
            )
            for a, b in zip(params["style_order"], params["markers"])
        ]
        # construct second legend, pad to avoid cutting off legend 1
        third_legend = plt.legend(
            handles=leg_3_handles,
            bbox_to_anchor=(thrdbbx, thrdbby),
            loc="lower left",
            frameon=False,
        )
        third_legend.set_title(l3_title, prop={"size": 15})  # type: ignore

        axt.add_artist(third_legend)

    # build second legend
    # markers of color associated with
    # lithologies in the plot
    leg_elem = [
        Line2D(
            [],
            [],
            markersize=10,
            marker="o",
            color="white",
            label=a,
            markerfacecolor=cmapin[a],
            # alpha=alpha_sct, # this works, not sure I like it though
        )
        for a in lith_present
    ]
    # construct second legend, pad to avoid cutting off legend 1
    axt.legend(
        handles=leg_elem,
        borderaxespad=2.1,
        bbox_to_anchor=(tpbbx, tpbby),
        loc="upper left",
        frameon=False,
    )
    # set legend 2 title
    axt.get_legend().set_title(topledgettl, prop={"size": 15})  # type: ignore
    # set plot title
    axt.set_title(title, size=20)

    if pca_obj is not None:  # only label if pca_obj given
        # set x-axis label
        axt.set_xlabel(
            f"Principal component {x_axispca} ({round(pca_obj[x_axispca-1]*100,2)}%)",
            size=15,
        )
        # set y-axis label
        axt.set_ylabel(
            f"Principal component {y_axispca} ({round(pca_obj[y_axispca-1]*100,2)}%)",
            size=15,
        )
    else:
        axt.set_xlabel(
            f"{pca1a}", size=15
        )  # if pca not given use column names to label axis
        axt.set_ylabel(f"{pca2a}", size=15)
    # turn on grid
    axt.grid()

    return figt, axt


# preserve old version of pca_plot
# until new version is fully tested
def pca_plot_old(
    df,
    pca_df,
    pca_obj,
    pca1="PC1",
    pca2="PC2",
    metals="metals",
    lith="lithology_relog",
    npr_size="npr_size",
    cmapin=cm.turbo,  # type: ignore
    pca1a="PC1",
    pca2a="PC2",
    alpha_sct=1.0,
    alpha_lns=0.8,
    lith_order_in=None,
    title="PCA Bi-Plot",
    x_axispca=1,
    y_axispca=2,
    plot_npr=True,
    edgecolor=True,
    loading_lines=True,
    tpbbx=0.985,
    tpbby=1.05,
    mkrin="o",
    btbbx=0.985,
    btbby=0,
    topledgettl="Lithology",
    bottomledgettl="NPR",
    btmldglbls=None,
    bold=False,
):
    """
    Makes PCA bi-plot with metals loading

    Parameters
    ----------
    df  pandas dataframe
        At minimum has columns for sample lithology, NPR sizes,
        and 2 principal components.

    pca_df : pandas dataframe | None
        Contains PCA loadings for metals, structured as follows:
        rows are metals, columns are PCA loadings and labels
        to be passed to loadings_line_plot.
        If loading_lines=False, set 'pca_df=None'.

    pca_obj : array-like, with len>=2
        The percentage of variance explained by the PCA values being plotted.

    pca1 : str, default 'PCA1', optional
        Name of column in pca_df with PCA1 values (x-axis)
        to be passed to loadings_line_plot.
        If loading_lines=False, this variable will be ignored.

    pca2 : str, default 'PCA2', optional
        Name of column in pca_df with PCA2 values (y-axis)
        to be passed to loadings_line_plot.
        If loading_lines=False, this variable will be ignored.

    metals : st, default 'metals', optional
        Name of column in pca_df with metal names for labeling,
        to be passed to loadings_line_plot.
        If loading_lines=False, this variable will be ignored.

    lith : str, default 'Lithology'
        Name of column in 'df' with the lithologies.

    npr_size : str, default 'NPR', optional
        Name of column in df with NPR size categories,
        the values in this column will correspond directly to the size of
        the plotted points, column needs to be numerical.
        If plot_npr=False this variable is ignored.

    cmapin = matplotlib colormap object, default cm.turbo
        Colormap to be used for coloring the points by lithology.

    pca1a : str, default 'PCA1'
        Name of column in df with PCA1 (x-axis).

    pca2a : str, default 'PCA2'
        Name of column in df with PCA2 (y-axis).

    alpha_sct : float(0-1) or str, default 1
        If float, alpha value for scatterplot points
        If str, column in df to use for alpha values, the column
        must have numerical values (0-1).

    alpha_lns : float(0-1), default 1
        Alpha value for metal loading lines, passed to loadings_line_plot.
        If loading_lines=False this variable is ignored.

    lith_order_in : list-like strings, default Lith_order
        The order in which to list the lithologies in the legend.

    title : str, default 'PCA Bi-Plot'
        Title to give the plot.

    x_axispca : int, default 1
        Index+1 location in pca_obj of variance explained for PCA1.
        Must be less than len(pca_obj).

    y_axispca : int, default 2
        Index+1 location in pca_obj of variance explained for PCA2.
        Must be less than len(pca_obj).

    plot_npr : bool, default True
        Whether to plot NPR sizes

    edgecolor : bool, default True
        Use edgecolor on markers in plot, if True edgecolor='k'.

    loading_lines : bool, default True
        Whether to call loadings_line_plot() function.
        If False set pca_df=None

    tpbbx : float
        x of bounding box to anchor for main legend

    tpbby : float
        y of bounding box to anchor for main legend

    mkrin : str
        matplotlib marker code

    btbbx : float
        x of bounding box to anchor for NPR legend

    btbby : float
        y of bounding box to anchor for NPR legend

    topledgettl : str
        Title for top legend

    bottomledgettl : str
        Title for bottom legend

    btmldglbls : list-like strings
        Labels for bottom legend

    Returns
    -----
    figt
        matplotlib.pyplot figure

    axt
        matplotlib.pyplot axes
    """

    # set mutable defaults
    if btmldglbls is None:
        btmldglbls = ["NPR<0.2", "0.2<NPR<2 ", "2<NPR<3", "NPR>3"]

    # call line plot function
    if loading_lines:
        figt, axt = loadings_line_plot(
            pca_df=pca_df,
            pca1=pca1,
            pca2=pca2,
            alpha_lns=alpha_lns,
            metals=metals,
            bold=bold,
        )
    else:
        figt, axt = plt.subplots(figsize=(10, 10))

    if lith_order_in is None:
        print("none")
        lith_order_in = Lith_order
        # get color range
        unique_lith_in = df[lith].unique()

        colors = np.linspace(0, 1, len(unique_lith_in))

        lith_present = [l for l in lith_order_in if l in unique_lith_in]
        # dictionary to map lithologies to unique color spectrum
        color_dict = dict(zip(lith_present, colors))
    else:
        lith_present = lith_order_in
        unique_lith_in = df[lith].unique()
        colors = np.linspace(0, 1, len(lith_present))
        color_dict = dict(zip(lith_present, colors))
        print(color_dict)

    # if sorted(unique_lith_in) != sorted(lith_present):
    #     warnings.warn(
    #         "Lithologies in sample not present in chosen lith order, appending to end"
    #     )
    #     lith_present = lith_present + [
    #         l for l in unique_lith_in if l not in lith_present
    #     ]

    # make copy of df to avoid altering original
    temp = df.copy()

    # map lithologies to new 'color' column
    # apply cmap to turn linspace into rgb color from matplotlib cm
    temp["color"] = temp[lith].replace(color_dict).apply(cmapin)

    # make lithology a categorical column
    # have to do second due to multi-index problems
    # when using 'replace' function on lith column
    temp[lith] = Categorical(temp[lith], categories=lith_present, ordered=True)
    # sort on lith column
    temp.sort_values(by=lith, inplace=True)

    temp.reset_index(inplace=True, drop=True)

    if isinstance(alpha_sct, str):
        alpha_sct = temp[alpha_sct]

    if edgecolor:
        edg = "k"
    else:
        edg = None

    # plot
    temp.plot(
        kind="scatter",
        x=pca1a,
        y=pca2a,
        c="color",
        facecolors="none",
        edgecolors=edg,
        marker=mkrin,
        ax=axt,
        alpha=alpha_sct,
        s=npr_size,
        legend=False,
    )

    # calc plot limits
    xmin, xmax, ymin, ymax = axis_limits(df[pca1a], df[pca2a])

    # set plot limits to make square
    axt.set_xlim(xmin, xmax)
    axt.set_ylim(ymin, ymax)

    if plot_npr:
        # npr labels, these are hard coded but could be changed if we want
        npr_labels = btmldglbls
        # size of npr circles in legend (different from plot because
        # they are built on different scales)
        npr_leg_size = np.linspace(5, 15, len(btmldglbls))

        # make legend handles for NPR size legend
        # number of 'o' markers equal to number of sizes
        leg_1_handles = [
            Line2D(
                [],
                [],
                markersize=a,
                marker="o",
                color="w",
                markerfacecolor="grey",
                label=b,
            )
            for a, b in zip(npr_leg_size, npr_labels)
        ]
        # construct first legend
        # locate in bottom right
        first_legend = plt.legend(
            handles=leg_1_handles,
            loc="lower left",
            bbox_to_anchor=(btbbx, btbby),
            borderaxespad=1.0,
            frameon=False,
        )
        # set first legend title
        first_legend.set_title(bottomledgettl, prop={"size": 15})  # type: ignore
        # add first legend as artist
        axt.add_artist(first_legend)
    else:
        pass

    # build second legend
    # markers of color associated with
    # lithologies in the plot
    leg_elem = [
        Line2D(
            [], [], markersize=10, marker="o", color="white", label=a, markerfacecolor=b
        )
        for a, b in zip(lith_present, cmapin(colors))
    ]
    # construct second legend, pad to avoid cutting off legend 1
    axt.legend(
        handles=leg_elem,
        borderaxespad=2.1,
        bbox_to_anchor=(tpbbx, tpbby),
        loc="upper left",
        frameon=False,
    )
    # set legend 2 title
    axt.get_legend().set_title(topledgettl, prop={"size": 15})  # type: ignore
    # set plot title
    axt.set_title(title, size=20)

    # set x-axis label
    axt.set_xlabel(
        f"Principal component {x_axispca} ({round(pca_obj[x_axispca-1]*100,2)}%)",
        size=15,
    )
    # set y-axis label
    axt.set_ylabel(
        f"Principal component {y_axispca} ({round(pca_obj[y_axispca-1]*100,2)}%)",
        size=15,
    )
    # turn on grid
    axt.grid()

    return figt, axt


# %%


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
