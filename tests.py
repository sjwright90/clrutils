# %%
import pandas as pd
from pathlib import Path
from clrutils import Lith_order
import matplotlib.cm as cm
from importlib import reload
from clrutils.pca_preprop import npr_to_bins
# %%
import clrutils
reload(clrutils.pca_plots)
from clrutils.pca_plots import pca_plot
# %%
wdir_test = Path(
    r"C:\Users\SamWright\Life Cycle Geo, LLC\LCG Server - Documents\General\PROJECTS\2019\05 19KSM01 - Seabridge Gold - KSM Project\03 Technical Work\00 Python_R\01_a_Combined_Mitchell\03_Processed_Data/envpca".replace(
        "\\", "/"
    )
)

# %%
test_df = pd.read_csv(wdir_test / "plot_df.csv", index_col=0)
test_df["deposit"] = test_df["deposit"].str.title()
test_ldg = pd.read_csv(wdir_test / "ldg_mat.csv")
pca_obj = [0.5, 0.5]
test_df["lithology_relog"].fillna("No Data", inplace=True)
test_df = test_df[test_df["lithology_relog"].isin(Lith_order)].copy()
assert all(test_df["lithology_relog"].isin(Lith_order))
test_df["labels_temp"], test_df["size_temp"] = npr_to_bins(
    test_df["npr_calc"], bins=[0.2, 3], min_size=50, max_size=250
)
# %%
params = {"style":"deposit",
        "markers":["*", "v"],
}
t1, a1 = pca_plot(
    test_df,
    test_ldg,
    pca_obj,
    lith="lithology_relog",
    lith_order_in=Lith_order,
    npr_size="size_temp",
    btmldglbls=["NPR<0.2","2<NPR<3", "NPR>3"],
    alpha_sct=0.5,
    alpha_lns=0.5,
    edgecolor=True,
    loading_lines=True,  # type: ignore
    cmapin=cm.cool,  # type: ignore
    thrdbby=0.15,
    btbbx=0.989,
    **params,

)
t1.set_size_inches(12,12)
# %%
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
    lith_order_in=Lith_order,
    title="PCA Bi-Plot",
    x_axispca=1,
    y_axispca=2,
    plot_npr=True,
    edgecolor=True,
    loading_lines=True,
    tpbbx=0.985,
    tpbby=1.05,
    mkrin="lith_deposit",
    btbbx=0.985,
    btbby=0,
    topledgettl="Lithology",
    bottomledgettl="NPR",
    btmldglbls=["NPR<0.2", "0.2<NPR<2 ", "2<NPR<3", "NPR>3"],
    bold=False,
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

sns.scatterplot(
    data=test_df, x="PC1", y="PC2", hue="lithology_relog", style="deposit", **params
)
# %%
