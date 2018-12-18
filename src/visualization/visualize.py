import matplotlib.pyplot as plt 
import pandas as pd
from textblob import TextBlob
from src.features.build_features import normalize, word_count, sentence_avg_word_length

genre_dict = {
    'g':'Rock',
    'b':'Hip-Hop',
    'r':'Pop'
}

def plot_genre_and_word_count(df):
    """
    plots genre and word_count.
    Parameters
        ----------
        df : Dataframe
            Pandas Dataframe with our filtered songs
        Returns
            Generates a graph with word count on the x axis and genre on the y axis
            with different colors representing each genre.
    """
    df = word_count(df, 'word_count', 'lyrics')
    for color, genre in genre_dict.items():
        filtered_df = df[df['genre'] == genre]
        plt.scatter(filtered_df['word_count'], filtered_df['genre'], c=color, label=genre)

    plt.title('Word count pr. genre')
    plt.xlabel('Word Count')
    plt.ylabel('Genre')
    plt.legend()
    plt.show()

def plot_genre_and_normalized_word_count(df):
    """
    plots genre and normalized word_count.
    Parameters
        ----------
        df : Dataframe
            Pandas Dataframe with our filtered songs
        Returns
            Generates a graph with normalized word count on the x axis and genre on the y axis
            with different colors representing each genre.
    """
    df = word_count(df, 'word_count', 'lyrics')
    df = normalize(df, 'normalized_word_count', 'word_count')
    for color, genre in genre_dict.items():
        filtered_df = df[df['genre'] == genre]
        plt.scatter(filtered_df['normalized_word_count'], filtered_df['genre'], c=color, label=genre)

    plt.title('Normalized Word count pr. genre')
    plt.xlabel('Normalized Word Count')
    plt.ylabel('Genre')
    plt.legend()
    plt.show()
    
def plot_genre_and_avg_word_len(df):
    """
    plots genre and average word length.
    Parameters
        ----------
        df : Dataframe
            Pandas Dataframe with our filtered songs
        Returns
            Generates a graph with average word length on the x axis and genre on the y axis
            with different colors representing each genre.
    """
    df = sentence_avg_word_length(df, 'word_count', 'lyrics')
    for color, genre in genre_dict.items():
        filtered_df = df[df['genre'] == genre]
        plt.scatter(filtered_df['word_count'], filtered_df['genre'], c=color, label=genre)

    plt.title('Average Word count pr. genre')
    plt.xlabel('Average Word Count')
    plt.ylabel('Genre')
    plt.legend()
    plt.show()

def plot_sentiment_analysis(df):
    """
    plots sentiment analysis.
    Parameters
        ----------
        df : Dataframe
            Pandas Dataframe with our filtered songs
        Returns
            Generates a graph with polarity on the x axis and subjectivity on the y axis
            with different colors representing each genre.
    """
    df['sentiment'] = analyze_sentiment(df['lyrics']) # returns a series which is set on a new column of the dataframe
    for color, genre in genre_dict.items():
        filtered_df = df[df['genre'] == genre]
        x = filtered_df['sentiment'].apply(lambda x: x[0]) # polarity
        y = filtered_df['sentiment'].apply(lambda x: x[1]) # subjectivity
        plt.scatter(x, y, c=color, label=genre)

    plt.title('Sentiment Analysis pr. genre')
    plt.xlabel('Polarity')
    plt.ylabel('Subjectivity')
    plt.legend()
    plt.show()

def analyze_sentiment(series):
    '''
    returns pandas.Series 

    params:
        series (pandas.Series): series of song lyrics

    returns:
        (pandas.Series): of tuples with Sentiment objects (polarity, subjectivity)
    '''
    res = series.apply(lambda txt : TextBlob(txt).sentiment)
    return res