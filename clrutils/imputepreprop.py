# %%
import re
from numpy import nan

# %%


def regex_pad_sample_name(srs):
    """
    Pads sample names to hundreds place.
    Parameters
    ----------
    srs : str

    Returns
    -----
    str
    """
    srs = re.sub(r"(?<=\()(?=\d{1}[\.|-])", "00", srs)
    srs = re.sub(r"(?<=\()(?=\d{2}[\.|-])", "0", srs)
    return srs


# %%
def id_unmarked_nd(
    df,
    subset=None,
    inplace=False,
    lowerbound=0.2,
    upperbound=0.5,
    method="neg",
    checkall=False,
):
    """Identifies non-detects based on maximum or minimun value in a column
       being above a certain threshold of the value counts of the columns.
    Parameters
    ----------
    df : pandas dataframe
    subset : list-like, default None
        Subset of columns to apply drop to.
    inplace : bool, default False
        Whether to convert values inplace.
    lowerbound :  int (0-1), default 0.2
        Proportional cutoff for maximum or minimum value in the column.
    upperbound :  int (0-1), default 0.5
        Proportional cutoff of value count for any value, i.e. any value with
        value count proportion > upperbound will be flagged.
    method : one of 'neg','nan','half', default "neg"
        How to convert if 'inplace' True. 'neg' converts to negative, 'nan' converts
        to NaN, 'half' converts to half the value.
    checkall : bool, default False
        Whether to check all values in the column, or just the maximum and minimum.
        Tests all values against the upperbound.
    Returns
    -----
    dict
        Dictionary of columns and values to convert.
        Changes the data in place if inplace=True.
    """
    convertdict = {"neg": -1, "nan": nan, "half": 0.5}
    if subset is None:
        subset = df.columns
    temp = df[subset].copy()
    nd_col = {}
    for col in temp.select_dtypes(exclude="O"):
        # only count positive values as negatives would already be identified as ND
        vcount = temp[temp[col] >= 0][col].value_counts(normalize=True).sort_index()
        # bypass if no values
        if len(vcount) == 0:
            continue
        if vcount.iloc[-1] > lowerbound:
            nd_col[col] = [vcount.index[-1]]
        if vcount.iloc[0] > lowerbound:
            nd_col[col] = nd_col.setdefault(col, []) + [vcount.index[0]]
        if vcount.max() > upperbound:
            nd_col[col] = nd_col.setdefault(col, []) + [vcount.idxmax()]
        if checkall:
            nd_col[col] = nd_col.setdefault(col, []) + [
                a for a in vcount.index if vcount[a] > upperbound
            ]
    for k, v in nd_col.items():
        nd_col[k] = list(set(v))
    if inplace:
        # THIS ALGORITHM NEEDS LOOKING AT AND IMPROVING
        if method == "half":
            print(
                "Warning, attempts to identify only lowerbound values to half, but it is advised to double check the work."
            )
            for col, nd in nd_col.items():
                if len(nd) > 1:
                    nd_col[col] = [min(nd)]
                elif nd[0] == temp[col].min():
                    nd_col[col] = [nd[0]]
                elif nd[0] < temp[col].quantile(0.25):
                    nd_col[col] = [nd[0]]
                else:
                    nd_col[col] = []
            nd_col = {k: v for k, v in nd_col.items() if len(v) > 0}
        print(f"Converting following to {method}: ", nd_col)
        for col, nd in nd_col.items():
            for n in nd:
                df[col] = df[col].replace(n, convertdict[method] * n)
    return nd_col


# %%
def drop_missing(
    df, subset=None, cutoff=0.5, consider_ND=False, ND_raw=False, return_details=True
):
    """
    Drops columns with missing and, optionally, Non-Detect values from a data
    frame given a percentile cutoff.

    Parameters
    ----------
    df : pandas dataframe

    subset : list-like, default None
        Subset of columns to apply drop to.

    cutoff :  int (0-1), default 0.5
        Cutoff percentage of missing values, higher value is more stringent.

    consider_ND : bool, default False
        Whether to include Non-Detect in determination of missing.

    ND_raw : bool, default False
        If Non-Detect values are in their raw form, i.e. strings with < or >.

    return_details : bool, default True
        Whether to return list of column names dropped.

    Returns
    -----
    df [list]
        Return a dataframe with columns over missing limited dropped.
        Optionally returns a list with name of columns dropped.
    """
    if subset is None:
        subset = df.columns
    temp = df[subset].copy()
    if consider_ND:
        if ND_raw:
            for col in temp.select_dtypes("O"):
                temp[col] = temp[col].str.replace(r"<|>", "-", regex=True).apply(float)
        temp[temp < 0] = float("nan")
    temp.dropna(axis=1, thresh=int(cutoff * len(temp)), inplace=True)

    todrop = [c for c in subset if c not in temp.columns]

    if return_details:
        return df.drop(columns=todrop).copy(), todrop
    else:
        return df.drop(columns=todrop).copy()


# %%
def check_double_metals(
    subset, df, remove="pct", keep="ppm", splt="_", drop=True, remove_mostna=True
):
    """
    Determines if any metals are repeated.

    Parameters
    ----------

    subset : list-like
        Subset of columns to determine duplicates.

    df : pandas dataframe

    remove : str, default 'pct'
        Suffix type to remove.

    keep : str, default 'ppm'
        Suffix type to keep.

    splt : str, default '_'
        Substring to split the column name by.

    drop : bool, default True
        Whether to apply the changes in place.

    Returns
    -----
    None [to_rem]
        Changes the data in place or returns list.
    """
    strts = [
        a.split(splt)[0] for a in subset if a.split(splt)[1].lower() in [remove, keep]
    ]
    strts = {a for a in strts if strts.count(a) > 1}

    if remove_mostna:
        to_rem = []
        for s in strts:
            to_rem.append(
                df[casematch([s + splt + remove, s + splt + keep], df.columns)]
                .isna()
                .sum()
                .idxmax()
            )

    else:
        to_rem = casematch([a.lower() + splt + remove for a in strts], df.columns)

    if drop:
        df.drop(columns=to_rem, inplace=True)
        print(f"Dropping: {to_rem}")
    else:
        return to_rem


# %%
def casematch(val, cols):
    """
    Matches strings in a list to a list of strings, ignoring case.

    Parameters
    ----------
    val : list-like
        List of strings to match.

    cols : list-like
        List of strings to match to.

    Returns
    -----
    list
        List of strings that match.
    """
    if isinstance(val, str):
        val = [val]
    return [a for a in cols for v in val if a.lower() == v.lower()]


# %%


def pct_to_ppm(df, subset=None, rename=True, pct_tag=None, new_tag="ppm"):
    """
    Changes scale of pct columns to ppm.

    Parameters
    ----------
    df : pandas dataframe

    subset : list-like, default None
        Subset of columns to apply numeric to.

    rename : bool, default True
        Whether to rename columns in place.

    pct_tag : list, default ['pct']
        List of substrings to search for to change.

    new_tag : str, default 'ppm'
        Substring to rename the columns.

    Returns
    -----
    None
        Changes the data in place.
    """
    if pct_tag is None:
        pct_tag = ["pct"]
    new_names = {}
    if subset is not None:
        subset = [s for s in subset if s in df.columns]
        for col in df[subset]:
            for tag in pct_tag:
                if tag in col:
                    try:
                        assert df[col].max() <= 50
                    except AssertionError:
                        print(f"Warning, {col} has values greater than 50 pct.")
                    df[col] = df[col] * 10000
                    new_names[col] = col.replace(tag, new_tag)
    else:
        for col in df:
            for tag in pct_tag:
                if tag in col:
                    try:
                        assert df[col].max() <= 100
                    except AssertionError:
                        print(f"Warning, {col} has values greater than 100.")
                    df[col] = df[col] * 10000
                    new_names[col] = col.replace(tag, new_tag)
    if rename:
        df.rename(columns=new_names, inplace=True)


# %%
def isolate_metals_match_dfs(
    env_df,
    exp_df,
    env_st="al_ppm",
    env_en="zr_ppm",
    exp_st="al_ppm",
    exp_en="zr_ppm",
    dfs_isolated=False,
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

    dfs_isolated : bool, default False
        Whether the dataframes are already isolated to metals. If True, env_st, env_en, exp_st, exp_en are ignored.

    Returns
    -----
    env_temp_met, exp_temp_met
        Pandas dataframes of environmental and exploration
        subset by metals found in both.
    """
    if not dfs_isolated:
        env_temp = env_df.loc[:, env_st:env_en].copy()
        exp_temp = exp_df.loc[:, exp_st:exp_en].copy()
    else:
        env_temp = env_df.copy()
        exp_temp = exp_df.copy()

    # get the names
    env_prefix = env_temp.columns.str.split("_").str[0].str.lower()
    exp_prefix = exp_temp.columns.str.split("_").str[0].str.lower()

    inboth = list(set(env_prefix) & set(exp_prefix))

    # find matching names
    # both ways for possible mismatch on suffix
    env_met_incl = [a for a in env_temp.columns if a.split("_")[0].lower() in inboth]
    exp_met_incl = [a for a in exp_temp.columns if a.split("_")[0].lower() in inboth]

    # extract only columns that are present in both
    env_temp_met = env_temp[env_met_incl].copy()
    env_temp_met = env_temp_met.reindex(sorted(env_temp_met.columns), axis=1)
    exp_temp_met = exp_temp[exp_met_incl].copy()
    exp_temp_met = exp_temp_met.reindex(sorted(exp_temp_met.columns), axis=1)

    return env_temp_met, exp_temp_met


# %%
