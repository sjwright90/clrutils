from pandas import DataFrame, cut
from sklearn.preprocessing import StandardScaler
from clrutils.submods.helperfncs import df_anynull
import numpy as np


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


def clr_trans_scale(df, subset_start=None, subset_end=None, scale=True):
    """
    Apply center log ratio transformation and scale results

    Parameters
    ----------
    df : pandas dataframe, or array-like
        Contains data stored in Series. If data is a dict, argument order is
        maintained.

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
    if subset_start is not None:
        if subset_end is not None:
            temp = df.loc[:, subset_start:subset_end].copy()
        else:
            temp = df.loc[:, subset_start:].copy()
    else:
        temp = df.copy()
    temp_clr = CLR(temp.values)

    if not scale:  # if not scaling, return CLR only
        return DataFrame(temp_clr, columns=temp.columns)
    try:
        temp_sc = DataFrame(
            StandardScaler().fit_transform(temp_clr), columns=temp.columns
        )
    except Exception as e:
        print(
            "0s in the dataframe cause CLR to kick out -np.inf which breaks StandardScaler\nTry removing 0s from original df\n"
        )
        print(e)
        raise
    if subset_start is not None:
        temp_full = df.drop(columns=temp_sc.columns).copy().join(temp_sc)
        return temp_sc, temp_full
    else:
        return temp_sc


def isolate_metals_match_dfs(
    env_df, exp_df, env_st="al_ppm", env_en="zr_ppm", exp_st="AU_GPT", exp_en="ZN_PPM"
):
    """
    Extracts metals columns and finds shared metals between environmental and exploration data

    Parameters
    ----------
    env_df : pandas df
        Environmental dataframe, either full dataframe or already isolated to metals.

    exp_df : pandas df
        Experimental dataframe, either full dataframe or already isolated to metals.

    env_st : str, default 'al_ppm'
        Name of first metals in environmental dataset

    env_ed :  str, default 'zr_ppm'
        Name of last metals column in environmental dataset

    exp_st : str, default 'AU_GPT'
        Name of first metals in exploration dataset

    exp_en : str, default 'ZN_PPM'
        Name of last metals column in exploration dataset

    Returns
    -----
    env_temp_met, exp_temp_met
        Pandas dataframes of environmental and exploration
        subset by metals found in both.
    """
    env_temp = env_df.loc[:, env_st:env_en].copy()
    exp_temp = exp_df.loc[:, exp_st:exp_en].copy()

    # rename metals to match b/t datasets
    env_met_incl = [a.split("_")[0].lower() for a in env_temp]
    exp_met_incl = [a.split("_")[0].lower() for a in exp_temp]

    # find matching columns
    in_both = list(set(env_met_incl) & set(exp_met_incl))

    # rename df columns
    env_temp.columns = env_met_incl
    exp_temp.columns = exp_met_incl

    # extract only columns that are present in both
    env_temp_met = env_temp[in_both].copy()
    exp_temp_met = exp_temp[in_both].copy()

    return env_temp_met, exp_temp_met


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
    trnf_data, full_df, col_list=["lith", "lab"], num_comp=2, scale=True
):
    """
    Extract PCs and relevant columns for bi-plots

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

    colnames = [f"PC{x+1}" for x in range(num_comp)]
    temp = DataFrame(zip(*trnf_data[:, 0:num_comp].T)).join(full_df[col_list])
    temp.columns = colnames + col_list

    if scale:
        temp[colnames] = pc_scaler(temp[colnames])

    return temp


def npr_to_bins(
    nprseries, bins=[0.2, 2, 3], min_size=15, max_size=300, reverse=False, label="NPR"
):
    """
    Convert NPR values to bins

    Parameters
    ----------
    nprseries : pandas series
        Series of NPR values

    bins : list, default [0.2, 2, 3]
        List of bin edges, -np.inf and np.inf are added to the
        beginning and end of the list respectively

    min_size : int, default 15
        Minimum size of label to return

    max_size : int, default 300
        Maximum size of label to return

    reverse : bool, default False
        Whether to reverse the order of the binsS
    Returns
    -----
    splits_cat, splits_size
        Categorical and numerical splits of NPR values
    """
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
