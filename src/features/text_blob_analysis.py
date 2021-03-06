from textblob import TextBlob
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from multiprocessing import Pool

def analyze_sentiment(df):
    '''
    returns pandas.DataFrame with added columns with polarity and subjectivity 

    params:
        df (pandas.DataFrame): DataFrame of song lyrics

    returns:
        (pandas.DataFrame): DataFrame with added sentiment analysis columns (polarity, subjectivity)
    '''
    tqdm.pandas(desc="Analyzing sentiment...")
    res = df['lyrics'].progress_apply(lambda txt : TextBlob(txt).sentiment)
    tqdm.pandas(desc="Analyzing polarity...")
    df['polarity'] = res.progress_apply(lambda x: x[0])
    tqdm.pandas(desc="Analyzing subjectivity...")
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
    tqdm.pandas(desc="Preparing Text class analysis...")
    blobs = df['lyrics'].progress_apply(lambda txt : TextBlob(txt).tags)

    tqdm.pandas(desc="Analyzing classes...")
    df['nouns'] = blobs.progress_apply(lambda word_list: _count_word_class(word_list, 'NN'))
    df['adverbs'] = blobs.progress_apply(lambda word_list: _count_word_class(word_list, 'RB'))
    df['verbs'] = blobs.progress_apply(lambda word_list: _count_word_class(word_list, 'VB'))
    
    return df

def _count_word_class(words, word_class):
    count = 0
    for w in words:
        if w[1] == word_class:
            count = count + 1
    return count / 100
