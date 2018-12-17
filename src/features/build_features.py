import pandas as pd
import collections
import numpy as np
from src.data.make_dataset import create_dataset

def count_top_words_in_genre(genre, lyrics_df):
    """
    Detect the language of the text.
    Parameters
        ----------
        genre : str
            genre like 'Hip-Hop' or 'Pop'
        lyrics_df : pandas dataframe
            clean dataframe
        Returns
            return list of top words of genre
    """
    lyrics_df['most_used_words'] = pd.Series(collections.Counter(lyrics.split())
                                    .most_common(10) for _, lyrics in lyrics_df['lyrics'].iteritems())
    arr = np.array(lyrics_df[lyrics_df['genre'] == genre]['most_used_words'].tolist()) # merges row's most_used_word column to list
    arr = arr[~pd.isna(arr)] # removing nans'
    flat_list = [item for sublist in arr for item in sublist] # converts array of arrays to one big array
    genre_dict = {}
    for tupl in flat_list: 
        genre_dict[tupl[0]] = genre_dict.get(tupl[0], 0) + tupl[1] # sums up total occurances of each word
    top_words = collections.Counter(genre_dict)
    return top_words.most_common(10)

def word_count(df, new_col_name, col_with_lyrics):
    df = df.copy()
    df[new_col_name] = df[col_with_lyrics].apply(lambda words: _count_words(words))
    return df

def _count_words(words):
    try:
        return len(words.split())
    except:
        return 0 #TODO: better error handling, maybe not return 0

def sentence_avg_word_length(df, new_col_name, col_with_lyrics):
    df = df.copy()
    df[new_col_name] = df[col_with_lyrics].apply(_sentence_avg_word_length)
    return df

def _sentence_avg_word_length(sentence):
    return sum(len(word.split()) for word in sentence) / len(sentence.split())

def normalize(df, new_col_name, col_to_norm):
    df = df.copy()
    max = df[col_to_norm].max()

    df[new_col_name] = df['word_count'].apply(lambda val: val / max)
    return df

