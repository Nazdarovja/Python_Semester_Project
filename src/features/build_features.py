import pandas as pd
import collections
import numpy as np

def count_top_words_in_genre(genre, lyrics_df):
    """
    Detect the language of the text.
    Parameters
        ----------
        genre : str
            genre like 'Hip-Hop' or 'Pop'
        language : pandas dataframe
            dataframe with column 'most_used_words'
        Returns
            return list of top words of genre
    """
    arr = np.array(lyrics_df[lyrics_df['genre'] == genre]['most_used_words'].tolist()) # merges row's most_used_word column to list
    arr = arr[~pd.isna(arr)] # removing nans'
    flat_list = [item for sublist in arr for item in sublist] # converts array of arrays to one big array
    genre_dict = {}
    for tupl in flat_list: 
        genre_dict[tupl[0]] = genre_dict.get(tupl[0], 0) + tupl[1] # sums up total occurances of each word
    top_words = collections.Counter(genre_dict)
    return top_words.most_common(10)

def word_count_series(series):
    '''
    returns pandas.Series object with num of words

    params:
        series (pandas.Series): string of words

    returns:
        (pandas.Series): num of words
    '''
    series_word_count = series.apply(lambda words: _count_words(words))
    return series_word_count

def _count_words(words):
    try:
        return len(words.split())
    except:
        return 0 #TODO: better error handling, maybe not return 0
