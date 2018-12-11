import pandas as pd
import collections
import numpy as np
from src.data.make_dataset import create_dataset

if __name__ == "__main__":
    # Creating the dataset, unzip, 
    traning_data_df, test_data_df = create_dataset()

    print(traning_data_df.head())
    print(test_data_df.head())