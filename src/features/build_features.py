import pandas as pd
import os

from src.features.util import normalize, create_pickle
from src.features.word_count_analysis import sentence_avg_word_length,word_count
from src.features.text_blob_analysis import analyze_sentiment, analyze_word_class

def create_feature_pickles(training_df, test_df, TRAINING_PKL = 'training_data.pkl', TEST_PKL = 'test_data.pkl'):
    
    training_path = os.path.join('data','processed',TRAINING_PKL)
    test_path = os.path.join('data','processed',TEST_PKL)

    if os.path.isfile(training_path) and os.path.isfile(test_path):
        print('Do you wish to override existing feature data ? y/n')
        choice = input()
        if choice == 'y':
            # adding features as series to the dataframe
            training_df = _add_features(training_df)
            test_df = _add_features(test_df)

            create_pickle(training_df,TRAINING_PKL)
            create_pickle(test_df,TEST_PKL)
    else:
        training_df = _add_features(training_df)
        test_df = _add_features(test_df)
        
        create_pickle(training_df,TRAINING_PKL)
        create_pickle(test_df,TEST_PKL)
    
    training_df = pd.read_pickle(os.path.join('data','processed',TRAINING_PKL))
    test_df = pd.read_pickle(os.path.join('data','processed',TEST_PKL))

    return (training_df, test_df)

def build_homemade_network_input_list(df):
    avg_word_len = df['avg_word_len_nm']
    words = df["word_count_nm"]
    polarity = df['polarity']
    subjectivity = df['subjectivity']
    nouns = df['nouns']
    adverbs = df['adverbs']
    verbs = df['verbs']

    # Create feature list
    inputs = [[f, p, s, n, a, v, wl] for f, p, s, n, a, v, wl in zip(words, polarity, subjectivity, nouns, adverbs, verbs, avg_word_len)]
    return inputs

def create_labels(df):
    # Create targets / labels
    series = df['genre'].value_counts()
    genre_labels = series.keys() # getting genre labels
    targets = [[1 if i == j else 0 for i in genre_labels] for j in df['genre']]

    return (targets, genre_labels)

def _add_features(df):
    df = sentence_avg_word_length(df,"avg_word_len", 'lyrics')
    df = normalize(df, 'avg_word_len_nm', 'avg_word_len')
    df = word_count(df,"word_count", 'lyrics')
    df = normalize(df, 'word_count_nm', 'word_count')
    df = analyze_sentiment(df)
    df = analyze_word_class(df)
    
    return df