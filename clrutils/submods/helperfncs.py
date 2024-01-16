# contains the following functions:
# df_anynull
# get_max_max
# quick test of any null/NaN values in df
def df_anynull(df):
    """
    Test is any NaN/null cells in a pandas dataframe

    Inputs
    ----------
    df : pandas data frame

    Returns
    -----

        bool, True if NaN present, False otherwise
    """
    return df.isnull().values.any()


# custom function to get max distance from origin
def get_max_max(x, y):
    """
    Returns absolute maximum of two lists

    Inputs
    ----------
    x : array like numeric
        first entity

    y : array like numeric
        second entity

    Returns
    -----
        float or int
    """
    xmax = max(abs(x))
    ymax = max(abs(y))
    return max(xmax, ymax)
