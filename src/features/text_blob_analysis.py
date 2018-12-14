from textblob import TextBlob
import pandas as pd
from tqdm import tqdm

def analyze_lyrics(series):
    '''
    returns pandas.Series 

    params:
        series (pandas.Series): series of song lyrics

    returns:
        (pandas.Series): of tuples with Sentiment objects (polarity, subjectivity)
    '''
    tqdm.pandas(desc="Analyzing sentiment...")
    res = series.progress_apply(lambda txt : TextBlob(txt).sentiment)
    
    return res