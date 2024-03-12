# contains the following functions:
# impute_df

from pandas import DataFrame
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


def impute_df(df, subset=None, scale=False, neigh=5):
    """
    Applies imputation to a dataframe of numeric values.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe of chemical values to impute.

    subset : list-like, default None
        Subset of columns to apply numeric to.

    scale : bool, default False
        Whether to scale the data before imputation.

    neigh : int, default 5
        Number of neighbors to use for nearest neighbor imputation.

    Returns
    -----
    temp_ND_NaN
        Pandas dataframe of imputed values.
    """

    imputer_0 = KNNImputer(n_neighbors=neigh)
    imputer_1 = KNNImputer(n_neighbors=neigh)
    if subset is None:
        subset = df.columns

    temp = np.abs(df[subset].copy().reset_index(drop=True))

    if scale:
        scaler = MinMaxScaler()
        temp = DataFrame(scaler.fit_transform(temp), columns=temp.columns)

    temp_knn = DataFrame(imputer_0.fit_transform(temp), columns=temp.columns)

    if not temp_knn.index.equals(temp.index):
        temp.index = temp_knn.index

    temp.update(temp_knn)

    temp_ND_replace = df[subset].copy().reset_index(drop=True)

    temp_ND_NaN = temp_ND_replace.mask(temp_ND_replace > 0, np.NaN)

    if not temp_ND_replace.index.equals(temp_ND_NaN.index):
        temp_ND_NaN.index = temp_ND_replace.index

    temp_ND_NaN.fillna(temp, inplace=True)

    assert temp_ND_NaN.isna().sum().sum() == 0

    temp_ND_NaN.mask(temp_ND_NaN < 0, np.NaN, inplace=True)

    try:
        assert temp_ND_NaN.isna().sum().sum() > 0
    except:
        print("No negative values in your dataset")

    temp_ND_NaN_imputed = DataFrame(
        imputer_1.fit_transform(temp_ND_NaN), columns=temp_ND_NaN.columns
    )

    if not temp_ND_NaN_imputed.index.equals(temp_ND_NaN.index):
        temp_ND_NaN_imputed.index = temp_ND_NaN.index

    if scale:
        # Inverse transform the imputed values
        temp_ND_NaN_imputed = scaler.inverse_transform(temp_ND_NaN_imputed)

    temp_ND_NaN.update(temp_ND_NaN_imputed)

    return temp_ND_NaN
