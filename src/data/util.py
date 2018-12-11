import zipfile
import os
import pandas as pd
import numpy as np
from langdetect import detect
from tqdm import tqdm
from src.features.build_features import count_top_words_in_genre
from multiprocessing import Pool

def unzip_file():
    """
        If not done yet unzips the raw file and adds to ~/data/external folder

    """
    file_name = 'lyrics.csv.zip'
    path_to_zip_file = os.path.join('data','raw',file_name)
    directory_to_extract_to = os.path.join('data','external')
    
    if os.path.isfile(os.path.join(directory_to_extract_to,'lyrics.csv')):
        print('Skipping unzip...')
    else:
        print('Unzipping file...')
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)



def get_dataframe_from_path(csv_file_path):
    """
        Given a file path for lyrics dataset, will return dataframe, with only (genre, lyrics)

        Parameters
        ----------
        csv_file_path : str
            Path to the csv file to get.
        Returns
        -------
        pandas.DataFrame
            Complete dataframe, with utf-8 encoding and ignoring blank lines
    """

    lyrics_df = pd.read_csv(csv_file_path, 
                            encoding='utf-8', 
                            skip_blank_lines=True,
                            error_bad_lines=False,
                            usecols=['genre', 'lyrics'],
                            converters={'lyrics': _clean_lyrics},
                             )
    # Remove rows with no lyrics
    lyrics_df = lyrics_df.dropna()

    return lyrics_df

def filter_dataframe(lyrics_df, no_of_songs = 5000, list_of_genres= ['Pop', 'Hip-Hop','Rock'], language = 'en'):
    """
        Given filter parameters, returns new filtered pandas.DataFrame

        Parameters
        ----------
            lyrics_df : pandas.DataFrame
                Cleaned dataset
            no_of_songs : int (default value : 5000)
                Number of songs to return from each category
            list_of_genres : list (default value : ['Pop', 'Hip-Hop','Rock'] )
                Genres to filter by

        Returns
        -------
        pandas.DataFrame
            Dataframe with requested objects.
    """
    
    filtered_df = pd.DataFrame() # empty df for data
    tqdm.pandas(desc="Processing data...") # setup tqdm
    
    #
    for genre in list_of_genres:
        filtered_df = filtered_df.append(lyrics_df[lyrics_df['genre'] == genre][:no_of_songs] )

    # filtered_df['lyrics'] = filtered_df['lyrics'].progress_apply(lambda x: clean_words(x)) # cleaning the dataset for meaningless words

    # Parallel lang detection
    filtered_df = _parallelize_language_detection(filtered_df)
    
    # DEBUG FOR TESTING
    # genres = filtered_df['genre'].groupby(filtered_df['genre']).count() # groups the dataset by genre and counts the amount of each genre
    # print(genres)

    # Create pickle file for training data
    traning_data_df = filtered_df[:4000].reset_index(drop=True)
    _create_pickle(traning_data_df, 'training_data.pkl')
    # Create pickle file for test data
    test_data_df = filtered_df[4000:4250].reset_index(drop=True)
    _create_pickle(test_data_df, 'test_data.pkl')


def _create_pickle(df, file_name):
    """
        Removes punctuations and lowercases the string.

        Parameters
        ----------
        df : pandas.DataFrame
            genre/lyrics dataframe
    """
    path = os.path.join('data','interim',file_name)
    df.to_pickle(path)

def _clean_lyrics(lyrics):
    """
        Removes punctuations and lowercases the string.

        Parameters
        ----------
        lyrics : str
            lyrics for a single song
        Returns
        -------
        str
            String if there is text
        None
            Returns None if empty string
            
    """
    if len(lyrics) > 500:
        lyrics = lyrics.replace(",","")
        lyrics = lyrics.replace(".","")
        lyrics = lyrics.replace(":","")
        lyrics = lyrics.replace(";","")
        lyrics = lyrics.replace("\"","")
        lyrics = lyrics.replace("\n"," ")
        lyrics = lyrics.replace("/","")
        lyrics = lyrics.replace("?","")
        lyrics = lyrics.replace("!","")
        lyrics = lyrics.replace("â€œ","")
        lyrics = lyrics.replace("â€˜","")
        lyrics = lyrics.replace("æ","")
        lyrics = lyrics.replace("ø","")
        lyrics = lyrics.replace("å","")
        lyrics = lyrics.replace("*","")
        return lyrics.lower()
    
    return None

def _detect_english_string(input_string, language= 'en'):
    """
    Detect the language of the text.
    Parameters
        ----------
        lyrics : str
            lyrics for a single song
        language : str (default = 'en')
            language to sort by
        Returns
        -------
        bool
            True if given language, False if other.
    """
    try:
        val = detect(input_string) == 'en'
    except:
        return False
    return val

def _clean_words(lyrics):
    stop_words = ['i', 'like', 'me', 'you', 'it', "it's", 'too', 'to', 'nan', 'the', 'and', 'a']
    words = lyrics.split()
    for idx, word in enumerate(words): 
        if word in stop_words: # makes it O(1) because it's a Set (unique values), just like hashamp
            words.pop(idx)
    return " ".join(words)

def _process(df):
    tqdm.pandas(desc="Checking language...")
    filtered_mask = df['lyrics'].progress_apply(lambda x: _detect_english_string(x))
    return df[filtered_mask]

def _parallelize_language_detection(df):

    # Get cpu_count
    CPUS = os.cpu_count()

    # Split dataframe into cpu amount
    array_of_dfs = np.split(df, CPUS)

    # parralize
    p = Pool(CPUS)
    array_of_dfs = p.map(_process, array_of_dfs)

    # concat arrays of genres_dfs
    result_df = pd.DataFrame()
    for genre_df in array_of_dfs:
        result_df = result_df.append(genre_df)
    return result_df