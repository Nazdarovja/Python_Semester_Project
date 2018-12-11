import pandas as pd
import collections
import numpy as np
from src.data.make_dataset import create_dataset


if __name__ == "__main__":
    # Creating the dataset, unzip, 
    lyrics_df = create_dataset()