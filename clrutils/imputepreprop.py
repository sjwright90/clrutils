import re


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
    if isinstance(val, str):
        val = [val]
    return [a for a in cols for v in val if a.lower() == v.lower()]


# %%


def pct_to_ppm(df, subset=None, rename=True, pct_tag="pct", new_tag="ppm"):
    """
    Changes scale of pct columns to ppm.

    Parameters
    ----------
    df : pandas dataframe

    subset : list-like, default None
        Subset of columns to apply numeric to.

    rename : bool, default True
        Whether to rename columns in place.

    pct_tag : str, default 'pct'
        Substring identifying percentile columns.

    new_tag : str, default 'ppm'
        Substring to rename the columns.

    Returns
    -----
    None
        Changes the data in place.
    """
    new_names = {}
    if subset is not None:
        subset = [s for s in subset if s in df.columns]
        for col in df[subset]:
            if pct_tag in col:
                df[col] = df[col] * 10000
                new_names[col] = col.replace(pct_tag, new_tag)
    else:
        for col in df:
            if pct_tag in col:
                df[col] = df[col] * 10000
                new_names[col] = col.replace(pct_tag, new_tag)

    if rename:
        df.rename(columns=new_names, inplace=True)
