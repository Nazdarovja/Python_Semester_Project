import pandas as pd
import os

def normalize(df, new_col_name, col_to_norm):
    '''
    ref: https://en.wikipedia.org/wiki/Normalization_(statistics)
    '''
    if (col_to_norm == 'word_count'):
        max = 1718
        min = 74
    elif (col_to_norm == 'avg_word_len'):
        max = 0.0010981580557913647
        min = 1.431355415135597e-06

    
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

    file_path = os.path.join('data','processed',file_name)
    
    if os.path.isfile(file_path):
        os.remove(file_path)

    df.to_pickle(file_path)