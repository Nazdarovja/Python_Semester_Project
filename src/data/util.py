import zipfile
import os.path
file_name = 'lyrics.csv.zip'
path_to_zip_file = f'data/raw/{file_name}'
directory_to_extract_to = f'data/external/{file_name}'

def unzip_file():
    """
        If not done yet unzips the raw file and adds to external folder

    """
    if os.path.isfile(f'{directory_to_extract_to}/lyrics.csv'):
        print('Skipping unzip...')
    else:
        print('Unzipping file...')
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)