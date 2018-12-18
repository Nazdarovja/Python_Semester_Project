import pandas as pd
import os

from src.features.util import normalize, create_pickle
from src.features.word_count_analysis import sentence_avg_word_length,word_count
from src.features.text_blob_analysis import analyze_sentiment, analyze_word_class

def create_features(training_df, test_df, TRAINING_PKL = 'training_data', TEST_PKL = 'test_data'):
    
    training_path = os.path.isfile(os.path.join('data','processed',TRAINING_PKL))
    test_path = os.path.isfile(os.path.join('data','processed',TEST_PKL))

    if training_path and test_path:
        print('Do you wish to override existing feature data ? y/n')
        choice = input()
        if choice == 'y':
            # adding features as series to the dataframe
            training_df = _add_features(training_df)
            test_df = _add_features(test_df)

            create_pickle(training_df,TRAINING_PKL)
            create_pickle(test_df,TRAINING_PKL)
    else:
        training_df = _add_features(training_df)
        test_df = _add_features(test_df)

        create_pickle(training_df,TRAINING_PKL)
        create_pickle(test_df,TRAINING_PKL)
    
    training_df = pd.read_pickle(os.path.join('data','processed',TRAINING_PKL))
    test_df = pd.read_pickle(os.path.join('data','processed',TEST_PKL))

    return (training_df, test_df)

def _add_features(df):
    df = sentence_avg_word_length(df,"avg_word_len", 'lyrics')
    df = normalize(df, 'avg_word_len_nm', 'avg_word_len')
    df = word_count(df,"word_count", 'lyrics')
    df = normalize(df, 'word_count_nm', 'word_count')
    df = analyze_sentiment(df)
    df = analyze_word_class(df)
    return df