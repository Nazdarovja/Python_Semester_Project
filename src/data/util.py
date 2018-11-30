import zipfile

file_name = 'lyrics.csv.zip'
path_to_zip_file = f'data/raw/{file_name}'
directory_to_extract_to = f'data/external/'

def unzip_file():
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)