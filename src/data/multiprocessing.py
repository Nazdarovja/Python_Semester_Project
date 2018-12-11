import multiprocessing as Pool
from util import _detect_english_string

def process(df):
    filtered_mask = df.apply(lambda x: _detect_english_string(x))
    return df[filtered_mask]

if __name__ == '__multiprocessing__':
    p = Pool(5)
