import zipfile
import os.path
import pandas as pd
from langdetect import detect

file_name = 'lyrics.csv.zip'
path_to_zip_file = f'data/raw/{file_name}'
directory_to_extract_to = f'data/external/{file_name}'

def unzip_file():
    """
        If not done yet unzips the raw file and adds to ~/data/external folder

    """
    if os.path.isfile(f'{directory_to_extract_to}/lyrics.csv'):
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

def filter_dataframe(lyrics_df, no_of_songs = 100, list_of_genres= ['Pop', 'Hip-Hop','Country'], language = 'en'):
    """
        Given filter parameters, returns new filtered pandas.DataFrame

        Parameters
        ----------
            lyrics_df : pandas.DataFrame
                Cleaned dataset
            no_of_songs : int (default value : 100)
                Number of songs to return from each category
            list_of_genres : list (default value : ['Pop', 'Hip-Hop','Country'] )
                Genres to filter by

        Returns
        -------
        pandas.DataFrame
            Dataframe with requested objects.
    """
    ### insert cool method here
    ##################### TBD ################ 

    #### Language thing is veeeery slow, so first filter data.
    pass


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

def clean_puta_words(lyrics):
    stop_words = ['i', 'like', 'me', 'you', 'it', "it's", 'too', 'to', 'nan', 'the', 'and', 'a']
    words = lyrics.split()
    for idx, word in enumerate(words): 
        if word in stop_words: # makes it O(1) because it's a Set (unique values), just like hashamp
            words.pop(idx)
    return " ".join(words)