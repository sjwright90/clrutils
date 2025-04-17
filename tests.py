# %%
import pandas as pd
from pathlib import Path
from clrutils.pca_plots import loadings_line_plot, axis_limits
import matplotlib.cm as cm
from importlib import reload
from clrutils.pca_preprop import npr_to_bins, clr_trans_scale, CLR
import string
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns

from clrutils.pca_plots import pca_loading_matrix, pca_plot

from sklearn.decomposition import PCA

# %%


def pc_scaler(series):
    """
    Min-max scaler

    Parameters
    ----------
    trnf_data : pandas series, pandas df, or numpy array
        Values to be scaled.

    Returns
    -----
    series
        All items in series with a min-max scaler applied.
    """
    return series / (series.max() - series.min())


# %%

df_make_pca = pd.DataFrame(
    {
        "a": np.random.random(10),
        "b": np.random.random(10),
        "c": np.random.random(10),
        "d": np.random.random(10),
    }
)
idx_drop = np.random.choice(df_make_pca.index, 5, replace=False)
df_make_pca.drop(index=idx_drop, inplace=True)
df_nes, df_clr = clr_trans_scale(df_make_pca, subset_start="b", scale=False)
# %%
temp_out = CLR(df_make_pca.values)
df_new = pd.DataFrame(temp_out, columns=df_make_pca.columns, index=df_make_pca.index)

# %%
a, b, c = pca_loading_matrix(df_make_pca, n_components=4)

# %%
n_obs = 1000
test_df = pd.DataFrame(
    {
        "PC1": np.random.randn(n_obs),
        "PC2": np.random.randn(n_obs),
        "hue_col": np.random.choice(["sandstone", "shale", "granite"], n_obs),
        "style_col": np.random.choice(["quartz", "feldspar", "mica"], n_obs),
        "size_col": np.random.choice(["small", "medium", "large"], n_obs),
        "alpha_col": np.random.rand(n_obs),
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
palette_dict = {"sandstone": "orange", "shale": "blue", "granite": "green"}
marker_dict = {"quartz": "P", "feldspar": "o", "mica": "s"}
sizes = (300, 50)
size_order = ["small", "medium", "large"]
test_df["size_col"] = (
    ["small"] * int(0.8 * n_obs)
    + ["medium"] * int(0.1 * n_obs)
    + ["large"] * int(0.1 * n_obs)
)
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
    s=100,
    edgecolor="black",
)
# %%
sns.scatterplot(
    test_df,
    x="PC1",
    y="PC2",
    edgecolor="black",
    s=1000,
)
# %%
plt.scatter(
    test_df["PC1"],
    test_df["PC2"],
    s=1000,
    edgecolor="red",
)
# %%
np.random.shuffle(["quartz", "feldspar", "mica"])
