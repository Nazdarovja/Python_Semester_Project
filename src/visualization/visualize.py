import matplotlib.pyplot as plt 
import pandas as pd
import os

def plotting(df):
    directory = 'src/visualization/feature_plots'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plot_genre_and_word_count(df)
    plot_genre_and_avg_word_len(df)
    plot_sentiment_analysis(df)
    plot_genre_and_normalized_word_count(df)
    plot_word_class_pr_genre(df)
    plot_pie_charts_of_word_class_distribution(df)

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
    plotting_helper_method('word_count', 'genre', df)

    plt.title('Word count pr. genre')
    plt.xlabel('Word Count')
    plt.ylabel('Genre')
    plt.legend()
    plt.show()
    # plt.savefig('src/visualization/feature_plots/word_count_plot')

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
    plotting_helper_method('word_count_nm', 'genre', df)
    plt.title('Normalized Word count pr. genre')
    plt.xlabel('Normalized Word Count')
    plt.ylabel('Genre')
    plt.legend()
    plt.show()
    # plt.savefig('src/visualization/feature_plots/normalized_word_count_plot')
    
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
    plotting_helper_method('avg_word_len_nm', 'genre', df)
    plt.xlim(0, 1)

    plt.title('Normalized Average Word Length pr. genre')
    plt.xlabel('Normalized Average Word Length')
    plt.ylabel('Genre')
    plt.legend()
    plt.show()
    # plt.savefig('src/visualization/feature_plots/average_word_length')

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
    plotting_helper_method('polarity', 'subjectivity', df)

    plt.title('Sentiment Analysis pr. genre')
    plt.xlabel('Polarity')
    plt.ylabel('Subjectivity')
    plt.legend()
    plt.show()
    # plt.savefig('src/visualization/feature_plots/sentiment_analysis_plot')

def plot_word_class_pr_genre(df):
    """
    plots amount of word class pr genre.
    Parameters
        ----------
        df : Dataframe
            Pandas Dataframe with our filtered songs
        Returns
            Generates three graph with respectively nouns, verbs and adverbs on the x axis and then genre on the y axis
            with different colors representing each genre.
    """
    df['nouns'] = df['nouns'] * 100
    df['verbs'] = df['verbs'] * 100
    df['adverbs'] = df['adverbs'] * 100
    # plotting nouns
    plotting_helper_method('nouns', 'genre', df)
    plt.title('Amount of nouns pr song pr. genre')
    plt.xlabel("Amount of nouns in each song")
    plt.ylabel('Genre')
    plt.legend()
    plt.show()
    # plt.savefig('src/visualization/feature_plots/nouns_pr_genre_plot')

    # plotting verbs
    plotting_helper_method('verbs', 'genre', df)
    plt.title('Amount of verbs pr song pr. genre')
    plt.xlabel('Amount of verbs in each song')
    plt.ylabel('Genre')
    plt.legend()
    plt.show()
    # plt.savefig('src/visualization/feature_plots/verbs_pr_genre_plot')

    # plotting adverbs
    plotting_helper_method('adverbs', 'genre', df)
    plt.title('Amount of adverbs pr song pr. genre')
    plt.xlabel('Amount of adverbs in each song')
    plt.ylabel('Genre')
    plt.legend()
    plt.show()
    # plt.savefig('src/visualization/feature_plots/adverbs_pr_genre_plot')

def plotting_helper_method(x_axis, y_axis, df):
    """
    Helper method to scatter plots running through each genre
    """
    genre_dict = {
        'g':'Rock',
        'b':'Hip-Hop',
        'r':'Pop'
    }
    for color, genre in genre_dict.items():
        filtered_df = df[df['genre'] == genre]
        plt.scatter(filtered_df[x_axis], filtered_df[y_axis], c=color, label=genre)

def plot_pie_charts_of_word_class_distribution(df):
    """
    Circle diagrams of each genre's average word classes distribution.
    Parameters
        ----------
        df : Dataframe
            Pandas Dataframe with our filtered songs
        Returns
            Generates three circle diagrams of each genre's average word classes distribution
    """
    genre_dict = {
        'g':'Rock',
        'b':'Hip-Hop',
        'r':'Pop'
    }
    for _, genre in genre_dict.items():
        filtered_df = df[df['genre'] == genre]
    
        # plotting circle diagram for the specific genre
        avg_percentage_nouns = filtered_df['nouns'].mean()
        avg_percentage_verbs = filtered_df['verbs'].mean()
        avg_percentage_adverbs = filtered_df['adverbs'].mean()

        total = avg_percentage_nouns + avg_percentage_nouns + avg_percentage_nouns
        nouns = avg_percentage_nouns / total * 100
        verbs = avg_percentage_verbs / total * 100
        adverbs = avg_percentage_adverbs / total * 100

        # Pie chart
        labels = ['Nouns', 'Verbs', 'Adverbs']
        sizes = [nouns, verbs, adverbs]

        _, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')  
        plt.tight_layout()
        plt.title(f'Circle diagram of the genre "{genre}"s average word classes distribution')
        plt.show()
        # plt.savefig(f'src/visualization/feature_plots/{genre}_word_class_distribution')

