import pandas as pd
import collections
from src.data.make_dataset import create_dataset
from src.data.util import _clean_lyrics,_detect_english_string, clean_puta_words


if __name__ == "__main__":
    # Creating the dataset, unzip, 
    lyrics_df = create_dataset()
    print(len(lyrics_df))
    lyrics_df = lyrics_df[lyrics_df['lyrics'].str.len() > 500] # making sure it's actually a "song"
    lyrics_df = lyrics_df[(lyrics_df['genre'] != 'Not Available') & (lyrics_df['genre'] != 'Other')] # irrelevant genres
    print(len(lyrics_df))

    # lyrics_df = lyrics_df[:100].reset_index()
    # print(len(lyrics_df.iloc[0].lyrics))
    # lyrics_df.lyrics.apply(lambda lyrics: _clean_lyrics(lyrics))

    lyrics_df['lyrics'] = lyrics_df['lyrics'].apply(lambda x: clean_puta_words(x)) # cleaning the dataset for puta words
    lyrics_df['most_used_words'] = pd.Series(collections.Counter(lyrics.split())
                                    .most_common(10) for _, lyrics in lyrics_df['lyrics'].iteritems())
                                    # making a new column for each song and counts the most used words for each song

    genres = lyrics_df['genre'].groupby(lyrics_df['genre']).count() # groups the dataset by genre and counts the amount of each genre
    # print(lyrics_df.head()['most_used_words'])
    print(genres)

# What is the minimum length of the lyrics in order to be defined as a song? hmm
# minimum 500 length and removal of 'Not Available' 
# and 'Other' filter 61000 elements away from the dataset