import matplotlib.pyplot as plt 
import pandas as pd
from textblob import TextBlob
from src.features.build_features import normalize, word_count, sentence_avg_word_length
from src.features.text_blob_analysis import analyze_sentiment, analyze_word_class

def plotting(df):
    plot_genre_and_word_count(df)
    plot_genre_and_avg_word_len(df)
    plot_sentiment_analysis(df)
    plot_genre_and_normalized_word_count(df)
    plot_word_class_pr_genre(df)

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
    plotting_helper_method('word_count', 'genre', df)

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

    plotting_helper_method('normalized_word_count', 'genre', df)

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
    df = sentence_avg_word_length(df, 'avg_word_len', 'lyrics')

    plotting_helper_method('avg_word_len', 'genre', df)

    plt.title('Average Word Length pr. genre')
    plt.xlabel('Average Word Length')
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
    df = analyze_sentiment(df) # returns a series which is set on a new column of the dataframe

    plotting_helper_method('polarity', 'subjectivity', df)

    plt.title('Sentiment Analysis pr. genre')
    plt.xlabel('Polarity')
    plt.ylabel('Subjectivity')
    plt.legend()
    plt.show()

def plot_word_class_pr_genre(df):
    """
    plots sentiment analysis.
    Parameters
        ----------
        df : Dataframe
            Pandas Dataframe with our filtered songs
        Returns
            Generates three graph with respectively nouns, verbs and adverbs on the x axis and then genre on the y axis
            with different colors representing each genre.
    """

    df = analyze_word_class(df)

    # plotting nouns
    plotting_helper_method('nouns', 'genre', df)
    plt.title('Amount of nouns pr song pr. genre')
    plt.xlabel("Percentage of nouns in each song")
    plt.ylabel('Genre')
    plt.legend()
    plt.show()

    # plotting verbs
    plotting_helper_method('verbs', 'genre', df)
    plt.title('Amount of verbs pr song pr. genre')
    plt.xlabel('Percentage of verbs in each song')
    plt.ylabel('Genre')
    plt.legend()
    plt.show()

    # plotting adverbs
    plotting_helper_method('adverbs', 'genre', df)
    plt.title('Amount of adverbs pr song pr. genre')
    plt.xlabel('Percentage of adverbs in each song')
    plt.ylabel('Genre')
    plt.legend()
    plt.show()
    

def plotting_helper_method(x_axis, y_axis, df):
    genre_dict = {
        'g':'Rock',
        'b':'Hip-Hop',
        'r':'Pop'
    }
    for color, genre in genre_dict.items():
        filtered_df = df[df['genre'] == genre]
        plt.scatter(filtered_df[x_axis], filtered_df[y_axis], c=color, label=genre)