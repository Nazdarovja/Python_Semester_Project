import os
import pandas as pd
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

    # Constants
    TRANING_DATA_PATH = os.path.join('data','interim','training_data.pkl')
    TEST_DATA_PATH = os.path.join('data','interim','test_data.pkl')

    if (not os.path.isfile(TRANING_DATA_PATH)) and (not os.path.isfile(TEST_DATA_PATH)):
        print('Filtering data...')
        
        # Get and create dataframe
        EXTERNAL_PATH = os.path.join('data','external','lyrics.csv.zip','lyrics.csv')
        lyrics_df = get_dataframe_from_path(EXTERNAL_PATH)

        # Filter dataset
        filter_dataframe(lyrics_df) ## function not done
    else: 
        print('Skipping data filtering...')
    
    traning_data_df = pd.read_pickle(TRANING_DATA_PATH)
    test_data_df = pd.read_pickle(TEST_DATA_PATH)
    
    return (traning_data_df, test_data_df)