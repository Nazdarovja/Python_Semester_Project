from src.data.util import unzip_file, get_dataframe_from_path, filter_dataframe

def create_dataset():
    """
    Unzips file, then filters away non-english songs, and "massages" the data, 
    removing non text characters and rows without lyrics.

    Returns
    -------
    pandas.DataFrame
        containing the training data for the model.

    """
    # Unzips file if not unzipped yet
    unzip_file()

    # Get and create dataframe
    EXTERNAL_PATH = 'data/external/lyrics.csv'
    lyrics_df = get_dataframe_from_path(EXTERNAL_PATH) ########### TBD : add check so we dont get the dataframe again if we have an already filtered and ready one.

    # Filter dataset
    # lyrics_df = filter_dataframe(lyrics_df) ## function not done

    return lyrics_df