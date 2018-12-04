# Python_Semester_Project

# Dependency requirements
- Anacondas Python distribution
- `$pip install langdetect` - Google language detection lib
- `$pip install nltk` - Natural language toolkit lib

## Project Idea:

- Machine Learning with lyrics from [kaggle.com or other] to identify a songs genre by its lyrics; Input a song and get an estimation of its genre. 
- Supervised learning -> we give the "Machine" a song's lyrics which we know the genre of.
- (extra?) Perhaps train it to identify the artist by a songs lyrics - if possible, might be too difficult?

### NOTES
- n-grams i bog "datascience from scraps/scratch"
- -> generere en popsang ud fra en masse lyrics.


## Brainstorm - 30-11-2018
Our initial idea is to obtain a dataset of lyrics from kaggle.com (if possible), ideally genres. 

Thereby, it would be possible to do analysis on the lyrics BY GENRE, and thereby obtain knowledge on, ex: 

- top X most used words / wordcloud
- sentiment analysis 
- length?

The idea is to get data to differentiate characteristic for each genre, and thereby get data for supervised learning of our neural network.

The goal is to be able to feed the network lyrics from an "unknown" song, and have the network classify the song as a certain genre (by a percentage score).


### ideas / sources
https://tmthyjames.github.io/2018/february/Predicting-Musical-Genres/
https://www.kaggle.com/corizzi/lyrics-genre-analysis-machine-learning
https://pypi.org/project/langdetect/


## Authors
Alexander W. Hørsted-Andersen - developer - awha86
Mathias Bigler - developer - Zurina
Mikkel Emil Larsen - developer - mikkel7emil
Stanislav Novitski - developer - Stani2980

### Initial To Do
- Identify X genres to focus on, and how many songs/lyrics to use from each
- Massage data: remove non english songs, cleanup date etc.
- Write logic to identify top X most used words (per genre)
- 
