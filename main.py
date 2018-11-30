import pandas as pd

from src.data.make_dataset import create_dataset

if __name__ == "__main__":
    # Creating the dataset, unzip, 
    lyrics_df = create_dataset()

    print(lyrics_df.head())