import pandas as pd
import collections
import numpy as np
from src.data.make_dataset import create_dataset
from src.data.util import _clean_lyrics,_detect_english_string, clean_puta_words

def count_words_in_genre(genre, lyrics_df):
    arr = np.array(lyrics_df[lyrics_df['genre'] == genre]['most_used_words'].tolist()) # merges row's most_used_word column to list
    arr = arr[~pd.isna(arr)] # removing nans'
    flat_list = [item for sublist in arr for item in sublist] # converts array of arrays to one big array
    genre_dict = {}
    for tupl in flat_list: # sums up total occurances of each word
        genre_dict[tupl[0]] = genre_dict.get(tupl[0], 0) + tupl[1]
    return genre_dict


if __name__ == "__main__":
    # Creating the dataset, unzip, 
    lyrics_df = create_dataset()

    lyrics_df = lyrics_df[(lyrics_df['genre'] != 'Not Available') & (lyrics_df['genre'] != 'Other')] # irrelevant genres

    lyrics_df['lyrics'] = lyrics_df['lyrics'].apply(lambda x: clean_puta_words(x)) # cleaning the dataset for puta words
    lyrics_df['most_used_words'] = pd.Series(collections.Counter(lyrics.split())
                                    .most_common(10) for _, lyrics in lyrics_df['lyrics'].iteritems())
                                    # making a new column for each song and counts the most used words for each song

    words_genre = count_words_in_genre('Pop', lyrics_df)

    genres = lyrics_df['genre'].groupby(lyrics_df['genre']).count() # groups the dataset by genre and counts the amount of each genre
    print(lyrics_df.head()['most_used_words'])
    print(genres)
    word_counter = collections.Counter(words_genre)
    print(word_counter.most_common(20))

# What is the minimum length of the lyrics in order to be defined as a song? hmm
# minimum 500 length and removal of 'Not Available' 
# and 'Other' filter 61000 elements away from the dataset