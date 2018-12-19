import os
import pandas as pd
from .util import unzip_file, get_dataframe_from_path, filter_dataframe


def create_dataset():
    """
    Unzips file, then filters away non-english songs, and "massages" the data, 
    removing non text characters and rows without lyrics.

    Returns
    -------
    pandas.DataFrame
        containing the training data for the model.

    """

    # Create paths if not existent
    _create_paths()
    # Unzips file if not unzipped yet
    unzip_file()

    # Constants
    TRAINING_DATA_PATH = os.path.join('data','interim','training_data.pkl')
    TEST_DATA_PATH = os.path.join('data','interim','test_data.pkl')
    
    if (not os.path.isfile(TRAINING_DATA_PATH)) and (not os.path.isfile(TEST_DATA_PATH)):
        print('Filtering data...')
        
        # Get and create dataframe
        EXTERNAL_PATH = os.path.join('data','external','lyrics.csv')
        lyrics_df = get_dataframe_from_path(EXTERNAL_PATH)

        # Filter dataset
        filter_dataframe(lyrics_df) ## function not done

    else: 
        print('Skipping data filtering...')
    
    training_data_df = pd.read_pickle(TRAINING_DATA_PATH)
    test_data_df = pd.read_pickle(TEST_DATA_PATH)

    ## Shuffle datasets
    training_data_df = training_data_df.sample(frac=1).reset_index(drop=True)
    test_data_df = test_data_df.sample(frac=1).reset_index(drop=True)

    return (training_data_df, test_data_df)

def _create_paths():
    print('Creating missing paths...')
    if not os.path.isdir('data/external'):
        os.mkdir('data/external')
    
    if not os.path.isdir('data/interim'):
        os.mkdir('data/interim')

    if not os.path.isdir('data/processed'):
        os.mkdir('data/processed')