import zipfile
from langdetect import detect

file_name = 'lyrics.csv.zip'
path_to_zip_file = f'data/raw/{file_name}'
directory_to_extract_to = f'data/external/{file_name}'

def unzip_file():
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

def detect_english_string(input_string):
    '''
    Detect the language of the text. Returns 'en' for english
    '''
    return detect(input_string) == 'en'