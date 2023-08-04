import re


def standardize_metals_name(df, subset=None, replace=False, return_newnames=False):
    """
    Renames columns based on Xx_[pct|ppm] convention.

    Parameters
    ----------
    df : pandas dataframe

    subset : list-like, default None
        List of column names to subset and apply renaming to.

    replace : bool, default False
        Whether to replace the names within the function or return the names
        as a list.

    return_newnames : bool, default True
        If replace==True, also return names list.

    Returns
    -----
    [df] [cols_new]
        Either dataframe with renamed columns, list of new column names,
        or both.
    """

    if subset is not None:
        cols_rename = df[subset].columns
    else:
        cols_rename = df.columns
    cols_new = cols_rename.str.replace(" (", "_", regex=False)
    cols_new = cols_new.str.replace(")", "", regex=False)
    cols_new = cols_new.str.replace("%", "pct", regex=False)
    cols_new = cols_new.str.replace("gpt", "ppm", regex=False)

    if replace:
        if return_newnames:
            return df.rename(columns=dict(zip(cols_rename, cols_new))).copy(), cols_new
        else:
            return df.rename(columns=dict(zip(cols_rename, cols_new))).copy()
    else:
        return cols_new


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


# ANNOTATE
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


# ANNOTATE
def make_numeric(df, subset=None, as_neg=True):
    """
    Identifies numeric strings preceeded by < or > and turns them into numeric
    type.

    Parameters
    ----------
    df : pandas dataframe

    subset : list-like, default None
        Subset of columns to apply numeric to.

    as_neg : bool, default True
        Whether to convert observations with < or > to negative
        values.

    Returns
    -----
    None
        Changes the data in place.
    """
    if subset is not None:
        for col in df[subset].select_dtypes("O"):
            try:
                if as_neg:
                    df[col] = df[col].str.replace(r"<|>", "-", regex=True).apply(float)
                else:
                    df[col] = df[col].str.replace(r"<|>", "", regex=True).apply(float)
            except:
                print(f"{col} not numeric")
    else:
        for col in df.select_dtypes("O"):
            try:
                if as_neg:
                    df[col] = df[col].str.replace(r"<|>", "-", regex=True).apply(float)
                else:
                    df[col] = df[col].str.replace(r"<|>", "", regex=True).apply(float)
            except:
                print(f"{col} not numeric")


def check_double_metals(subset, df=None, remove="pct", keep="ppm", splt="_", drop=True):
    """
    Determines if any metals are repeated.

    Parameters
    ----------

    subset : list-like
        Subset of columns to apply numeric to.

    df : pandas dataframe, default None

    remove : str, default 'pct'
        Suffix type to remove.

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
    to_rem = [a.lower() + splt + remove for a in strts]

    if drop:
        casematch = [a for a in df.columns if a.lower() in to_rem]
        df.drop(columns=casematch, inplace=True)
        print(f"Dropping: {casematch}")
    else:
        return to_rem


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
