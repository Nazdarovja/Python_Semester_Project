from textblob import TextBlob
import pandas as pd
from tqdm import tqdm

def analyze_sentiment(df):
    '''
    returns pandas.DataFrame with added columns with polarity and subjectivity 

    params:
        df (pandas.DataFrame): DataFrame of song lyrics

    returns:
        (pandas.DataFrame): DataFrame with added sentiment analysis columns (polarity, subjectivity)
    '''
    df = df.copy()
    tqdm.pandas(desc="Analyzing sentiment...")
    res = df['lyrics'].progress_apply(lambda txt : TextBlob(txt).sentiment)
    df['polarity'] = res.progress_apply(lambda x: x[0])
    df['subjectivity'] = res.progress_apply(lambda x: x[1])
    return df

def analyze_word_class(df):
    '''
    returns pandas.DataFrame with added columns with polarity and subjectivity 

    params:
        df (pandas.DataFrame): DataFrame of song lyrics

    returns:
        (pandas.DataFrame): DataFrame with added
    '''
    df = df.copy()
    tqdm.pandas(desc="Preparing Text classe analysis...")
    blobs = df['lyrics'].progress_apply(lambda txt : TextBlob(txt).tags)
    tqdm.pandas(desc="Analyzing classes...")
    df['nouns'] = blobs.progress_apply(lambda word_list: _count_word_class(word_list, 'NN'))


def _count_word_class(words, word_class):
    pass