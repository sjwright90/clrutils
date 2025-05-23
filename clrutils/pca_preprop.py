# contains the following functions:
# CLR
# clr_trans_scale
# pc_scaler
# make_df_for_biplot
# npr_to_bins

from pandas import DataFrame, cut
from sklearn.preprocessing import StandardScaler
from clrutils.submods.helperfncs import df_anynull
import numpy as np
from typing import Tuple


def CLR(X):
    """
    Centred Log Ratio transformation.

    Parameters
    ---------------
    X : :class:`numpy.ndarray`
        2D array on which to perform the transformation, of shape :code:`(N, D)`.

    Returns
    ---------
    :class:`numpy.ndarray`
        CLR-transformed array, of shape :code:`(N, D)`.
    """
    X = np.array(X)
    X = np.divide(X, np.sum(X, axis=1).reshape(-1, 1))  # Closure operation
    Y = np.log(X)  # Log operation
    nvars = max(X.shape[1], 1)  # if the array is empty we'd get a div-by-0 error
    G = (1 / nvars) * np.nansum(Y, axis=1)[:, np.newaxis]
    Y -= G
    return Y


def clr_trans_scale(
    df: DataFrame, subset_start: str = None, subset_end: str = None, scale: bool = True
):  # for some reason some versions of python are not happy with the return type hinting
    """
    Apply center log ratio transformation and scale results

    Parameters
    ----------
    df : pandas dataframe, or array-like

    subset_start : str, default None, optional
        Column name of start column to sample for CLR, columns to be sampled
        must be sequential. If specified, 'df' must be a pandas data frame

    subset_end : str, default None, optional
        Column name of last column to sample for CLR. If not specified,
        the function will sample columns from 'subset_start' to the
        end of the columns. Will be ignored if 'subset_start'
        not specified.

    Returns
    -----
    temp_sc [temp_full]
        Dataframe of CLR and scaled values. If 'subset_start' specified, copy
        of the orginal dataframe is returned with treated columns replaced.
    """
    temp_idx = df.index.copy()  # save index for later

    if subset_start is not None:
        if subset_end is not None:
            temp = df.loc[:, subset_start:subset_end].copy().reset_index(drop=True)
        else:
            temp = df.loc[:, subset_start:].copy().reset_index(drop=True)
    else:
        temp = df.copy().reset_index(drop=True)
    if df_anynull(temp):
        print("Null values in dataframe, CLR will not work\n")
        raise ValueError
    temp_clr = CLR(temp.values)

    if scale:
        try:
            temp_sc = DataFrame(
                StandardScaler().fit_transform(temp_clr),
                columns=temp.columns,
                index=temp.index,
            )
        except Exception as e:
            print(
                "0s in the dataframe cause CLR to kick out -np.inf which breaks StandardScaler\nTry removing 0s from original df\n"
            )
            print(e)
            raise
    else:
        temp_sc = DataFrame(temp_clr, columns=temp.columns, index=temp_idx)

    if subset_start is not None:
        assert df.index.equals(
            temp_sc.index
        ), "Index of original dataframe and CLR dataframe do not match"
        temp_full = df.drop(columns=temp_sc.columns).copy().join(temp_sc)
        return temp_sc, temp_full
    else:
        return temp_sc


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


def make_df_for_biplot(
    trnf_data: np.ndarray,
    full_df: DataFrame,
    col_list: list = None,
    num_comp: int = 2,
    scale: bool = True,
    prefix: str = "PC",
) -> DataFrame:
    """
    Extract PCs and relevant columns for bi-plots. Scale the PCs with min-max scaler.

    Parameters
    ----------
    trnf_data : numpy array
        Matrix output of dimension reduction algorithm of form nxm where
        n is observations and m are dimensions.

    full_df : pandas df
        Pandas dataframe of full data set, from which to extract non-numeric
        columns to be used in bi-plot

    col_list : str, default ['lith','lab']
        Columns to extract from full_df.
        To extract nothing from full_df pass an empty list.

    num_comp : int, default 2
        Number of dimensions to extract from 'trnf_data'.
        Must have num_comp <= number of columns in trnf_data

    scale : bool, default True
        Whether to apply min-max scaler to extracted columns.

    Returns
    -----
    temp
        Dataframe. Components have a min-max scaler applied to them.
    """

    if col_list is None:
        raise ValueError("No columns to extract from full_df")

    colnames = [f"{prefix}{x+1}" for x in range(num_comp)]
    temp = DataFrame(
        trnf_data[:, 0:num_comp], columns=colnames, index=full_df.index
    ).join(full_df[col_list])
    temp.columns = colnames + col_list

    if scale:
        temp[colnames] = pc_scaler(temp[colnames])

    return temp


def npr_to_bins(
    nprseries, bins=None, min_size=15, max_size=300, reverse=False, label="NPR"
):
    """
    Convert NPR values to bins

    Parameters
    ----------
    nprseries : pandas series
        Series of NPR values

    bins : list, default [0.2, 3] # JVG Removed 2 bin
        List of bin edges, -np.inf and np.inf are added to the
        beginning and end of the list respectively

    min_size : int, default 15
        Minimum size of label to return

    max_size : int, default 300
        Maximum size of label to return

    reverse : bool, default False
        Whether to reverse the order of the bins

    label : str, default 'NPR'
        Label for the NPR values
    Returns
    -----
    splits_cat, splits_size
        Categorical and numerical splits of NPR values
    """
    if bins is None:
        bins = [0.2, 3]  # JVG Removed 2 bin
    bins = [-np.inf] + bins + [np.inf]
    lbstrt = [f"{label}<{bins[1]}"]
    lbend = [f"{label}>{bins[-2]}"]
    lbmid = [f"{bins[x]}<{label}<{bins[x+1]}" for x in range(1, len(bins) - 2)]
    splits_cat = cut(
        nprseries,
        bins=bins,
        labels=lbstrt + lbmid + lbend,
    )
    sizes = np.linspace(min_size, max_size, len(bins) - 1).astype(int)
    if reverse:
        sizes = sizes[::-1]
    splits_size = splits_cat.map(dict(zip(splits_cat.cat.categories, sizes))).astype(  # type: ignore
        int
    )
    return splits_cat, splits_size
