import pandas as pd
import collections
import numpy as np
from src.data.make_dataset import create_dataset
from src.features.text_blob_analysis import analyze_word_class

if __name__ == "__main__":
    # Creating the dataset, unzip, 
    training_data_df, test_data_df = create_dataset()