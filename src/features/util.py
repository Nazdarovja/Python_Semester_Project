import pandas as pd
import os

def normalize(df, new_col_name, col_to_norm):
    '''
    ref: https://en.wikipedia.org/wiki/Normalization_(statistics)
    '''
    df = df.copy()
    max = df[col_to_norm].max()
    min = df[col_to_norm].min()

    df[new_col_name] = df[col_to_norm].apply(lambda val: (val-min)/(max-min))

    return df

def create_pickle(df, file_name):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        genre/lyrics dataframe
    file_name: str
        file name to create
    """

    file_path = os.path.join('data','processed',file_name + '.pkl')
    
    if os.path.isfile(file_path):
        os.remove(file_path)

    df.to_pickle(file_path)