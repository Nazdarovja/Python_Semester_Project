import pandas as pd
import collections
import numpy as np
from src.data.make_dataset import create_dataset
from src.data.util import _clean_lyrics,_detect_english_string, clean_puta_words

if __name__ == "__main__":
    # Creating the dataset, unzip, 
    lyrics_df = create_dataset()

    lyrics_df = lyrics_df[(lyrics_df['genre'] != 'Not Available') & (lyrics_df['genre'] != 'Other')] # irrelevant genres
    lyrics_df['lyrics'] = lyrics_df['lyrics'].apply(lambda x: clean_puta_words(x)) # cleaning the dataset for puta words
    lyrics_df['most_used_words'] = pd.Series(collections.Counter(lyrics.split())
                                    .most_common(10) for _, lyrics in lyrics_df['lyrics'].iteritems())
                                    # making a new column for each song and counts the most used words for each song
    top_words_of_pop = count_top_words_in_genre('Pop', lyrics_df)
    genres = lyrics_df['genre'].groupby(lyrics_df['genre']).count() # groups the dataset by genre and counts the amount of each genre