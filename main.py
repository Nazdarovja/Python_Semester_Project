import pandas as pd
import collections
import numpy as np
from src.data.make_dataset import create_dataset
from src.features.build_features import sentence_avg_word_length, normalize, word_count
from src.features.text_blob_analysis import analyze_sentiment, analyze_word_class
from src.visualization.visualize import plotting
import src.neural_network as nn
import os

from src.models.train_model import train
from src.models.predict_model import predict

if __name__ == "__main__":
    # Creating the dataset, unzip, 
    training_data_df, test_data_df = create_dataset()
    res = sentence_avg_word_length(training_data_df, "avg_word_len", "lyrics")
    print(res)

    plotting(res)
    # # plotting(res)
    TRAINING_PKL = 'training_data.pkl'
    df = pd.read_pickle(os.path.join('data','processed',TRAINING_PKL))

    # targets
    series = df['genre'].value_counts()
    genre_labels = series.keys() # getting genre labels
    targets = [[1 if i == j else 0 for i in genre_labels] for j in df['genre']]

    avg_word_len = df['avg_word_len_nm']
    words = df["word_count_nm"]
    polarity = df['polarity']
    subjectivity = df['subjectivity']
    nouns = df['nouns']
    adverbs = df['adverbs']
    verbs = df['verbs']

    # Create feature list
    inputs = [[f, p, s, n, a, v, wl] for f, p, s, n, a, v, wl in zip(words, polarity, subjectivity, nouns, adverbs, verbs, avg_word_len)]
    
    network = train(inputs, targets, 500)

    for i in inputs[:100]:
    # res = predict([0.71148825065274152, 0.22561965811965812, 0.129914529914531, 0.2506896551724138, 0.0513455968010067,1,1], network)
        res = predict(i, network)
        print(res)