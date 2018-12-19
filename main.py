<<<<<<< HEAD
=======
import pandas as pd
import collections
import numpy as np
from src.data.make_dataset import create_dataset
from src.features.build_features import sentence_avg_word_length, normalize, word_count
from src.features.text_blob_analysis import analyze_sentiment, analyze_word_class
from src.visualization.visualize import plotting
#import src.neural_network as nn
>>>>>>> 2f655a513bfdf2031eb3863bf7d3e4a2061c7a28
import os

from src.data.make_dataset import create_dataset
from src.models.train_model import train
from src.models.predict_model import predict
from src.features.build_features import create_feature_pickles, build_homemade_network_input_list, create_labels

from src.visualization.visualize import plotting
if __name__ == "__main__":
    
    # Creating the dataset, unzip, 
    training_data_df, test_data_df = create_dataset()
    training_data_df, test_data_df = create_feature_pickles(training_data_df, test_data_df)
    inputs_training = build_homemade_network_input_list(training_data_df)
    inputs_test = build_homemade_network_input_list(test_data_df)
    targets, genre_labels = create_labels(training_data_df)

    print(inputs_training[10:20])
    network = train(inputs_training, targets, 10)

    count = 0
    for i, t in zip(inputs_test, test_data_df['genre']):
        test_res = predict(i, network)
        # print(test_res)
        maxnum = test_res.index(max(test_res))
        # print(t)
        if genre_labels[maxnum] == t:
            count = count +1
            print(f'{test_res} genre = {genre_labels[maxnum]}')
    success_rate = (count/len(inputs_test))*100
    print(f'Success Rate : {success_rate}% of {len(inputs_test)} lyrics')
