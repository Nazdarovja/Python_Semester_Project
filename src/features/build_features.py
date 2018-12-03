import pandas as pd

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