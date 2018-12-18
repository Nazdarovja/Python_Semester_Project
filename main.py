import pandas as pd
import collections
import numpy as np
from src.data.make_dataset import create_dataset
from src.features.build_features import sentence_avg_word_length
from src.visualization.visualize import plotting

if __name__ == "__main__":
    # Creating the dataset, unzip, 
    training_data_df, test_data_df = create_dataset()
    res = sentence_avg_word_length(training_data_df, "avg_word_len", "lyrics")
    print(res)

    # plotting(res)
    

