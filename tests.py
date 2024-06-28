# %%
import pandas as pd
from pathlib import Path
from clrutils import Lith_order
import matplotlib.cm as cm
from importlib import reload
from clrutils.pca_preprop import npr_to_bins
import string
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt


# %%
def plot_examples(colormaps):
    """
    Helper function to plot data with associated colormap.
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)
    n = len(colormaps)
    fig, axs = plt.subplots(
        1, n, figsize=(n * 2 + 2, 3), layout="constrained", squeeze=False
    )
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()


# %%

cmap = ListedColormap(
    [
        "darkorange",
        "gold",
        "lawngreen",
        "lightseagreen",
        "royalblue",
        "darkblue",
        "darkviolet",
        "magenta",
        "crimson",
    ]
)
plot_examples([cmap])
# %%

test_df = pd.DataFrame(
    {
        st: np.random.choice(list(string.ascii_uppercase), 20)
        for st in string.ascii_lowercase[:5]
    }
)
# %%
order = list(string.ascii_uppercase)
np.random.shuffle(order)

# %%
test = test_df.a.unique().tolist()

indices = {i: order.index(i) for i in order}

sorted(test, key=indices.get)
# %%
import clrutils

# reload(clrutils.pca_plots)
from clrutils.pca_plots import pca_plot

# %%
liths = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
lith_in = np.random.choice(liths, 100)
palt = ["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii"]
palt_in = np.random.choice(palt, 100)
salt = ["mm", "nn", "oo", "pp", "qq", "rr", "ss", "tt", "uu"]
salt_in = np.random.choice(salt, 100)
deposit = ["deposit1", "deposit2", "deposit3"]
deposit_in = np.random.choice(deposit, 100)
npr = np.random.choice([0.15, 1, 2.5, 3.5], 100)
npr_jitter = np.random.rand(100) * 0.1
npr_binary = np.random.choice([0, 1], 100)
npr_binary = np.where(npr_binary == 0, -1, npr_binary)
npr = npr + npr_jitter * npr_binary
npr_labels, npr_sizes = npr_to_bins(pd.Series(npr), min_size=35, max_size=350)
pc1 = (np.random.rand(100) - 0.5) * 2
pc2 = (np.random.rand(100) - 0.5) * 2

test_df = pd.DataFrame(
    {
        "PC1": pc1,
        "PC2": pc2,
        "lithology_relog": lith_in,
        "deposit": deposit_in,
        "npr_calc": npr,
        "npr_labels": npr_labels,
        "npr_sizes": npr_sizes,
        "primary_alterationn": palt_in,
        "secondary_alteration": salt_in,
    }
)
# %%
test_ldg = pd.DataFrame(
    {
        "PC1": (np.random.rand(9) - 0.5) * 2,
        "PC2": (np.random.rand(9) - 0.5) * 2,
        "metals": ["Al", "Ca", "K", "Pb", "Zn", "Ni", "Co", "Fe", "Mn"],
    }
)

pca_obj = [0.5, 0.5]
# %%
cmap_dict = {lith: cmap.colors[i] for i, lith in enumerate(liths)}
# %%
sorted(
    cmap_dict.keys(),
    key=np.random.choice(liths, len(liths), replace=False).tolist().index,
)
# %%
t1, a1 = pca_plot(
    test_df.sample(frac=1, random_state=42, replace=False),
    test_ldg,
    pca_obj,
    lith="lithology_relog",
    lith_order_in=np.random.choice(liths, len(liths), replace=False),
    npr_size="npr_sizes",
    alpha_sct=0.5,
    alpha_lns=0.5,
    edgecolor=True,
    loading_lines=True,  # type: i
    cmapin=cmap_dict,
    thrdbby=0.15,
    btbbx=0.989,
    # **params,
)
t1.set_size_inches(12, 12)
# %%
cmap_dict = {
    "A": "darkorange",
    "B": "gold",
    "C": "lawngreen",
    "D": "lightseagreen",
    "E": "royalblue",
    "F": "darkblue",
    "G": "darkviolet",
    "H": "magenta",
    "I": "crimson",
}

# %%
liths_linspace = np.linspace(0, 1, len(liths))

turbo_colors_from_linspace = [cm.turbo(x) for x in liths_linspace]

turbo_cmap_dict = {lith: turbo_colors_from_linspace[i] for i, lith in enumerate(liths)}
# %%
params = {
    "style": "deposit",
    "markers": ["*", "v", "o"],
}
for test_group in [
    ["A", "C", "D", "G"],
    ["G", "B", "E", "F", "H", "I"],
    ["C", "B", "A"],
]:
    cmap = ListedColormap([turbo_cmap_dict[i] for i in sorted(test_group)])  # type: ignore
    t1, a1 = pca_plot(
        test_df[test_df.lithology_relog.isin(test_group)],
        test_ldg,
        pca_obj,
        lith="lithology_relog",
        lith_order_in=sorted(test_group),
        npr_size="npr_sizes",
        btmldglbls=test_df.npr_labels.cat.categories.tolist(),
        alpha_sct=0.5,
        alpha_lns=0.5,
        edgecolor=True,
        loading_lines=True,  # type: ignore
        cmapin=cmap,  # type: ignore
        thrdbby=0.15,
        btbbx=0.989,
        # **params,
    )
    t1.set_size_inches(12, 12)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %%


# %%
figx, axx = plt.subplots(figsize=(10, 10))
params = {
    "s": test_df["npr_calc"],
    "alpha": 0.5,
    "palette": "tab10",
    "hue_order": Lith_order,
    "style_order": test_df["deposit"].unique(),
    "legend": "full",
    "edgecolor": "k",
    "linewidth": 0.5,
    "markers": True,
    "sizes": (20, 200),
    "size_order": test_df["deposit"].unique(),
    "size_norm": (0, 1),
    "size": None,
    "hue_norm": None,
    "legend": True,
    "ax": axx,
}

sns_obj = sns.scatterplot(
    data=test_df, x="PC1", y="PC2", hue="lithology_relog", style="deposit", **params
)
# %%
legend_handles, legend_labels = sns_obj.get_legend_handles_labels()
legend_sizes = [line.get_sizes()[0] for line in legend_handles]
# %%


plt_obj = plt.scatter(
    test_df["PC1"],
    test_df["PC2"],
    c=test_df["npr_calc"],
    s=test_df["npr_sizes"],
    cmap=cmap,
    alpha=0.5,
    edgecolor="k",
    linewidth=0.5,
    marker="o",
)
# %%
