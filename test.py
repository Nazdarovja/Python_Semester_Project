import collections
from nltk.corpus import stopwords
en_stops = set(stopwords.words('english'))

# clean lyrics for one song
def clean_data(sentence):
    lyrics = lyrics.replace(",","")
    lyrics = lyrics.replace(".","")
    lyrics = lyrics.replace(":","")
    lyrics = lyrics.replace(";","")
    lyrics = lyrics.replace("\"","")
    lyrics = lyrics.replace("\n","")
    lyrics = lyrics.replace("/","")
    lyrics = lyrics.replace("?","")
    lyrics = lyrics.replace("!","")
    lyrics = lyrics.replace("â€œ","")
    lyrics = lyrics.replace("â€˜","")
    lyrics = lyrics.replace("æ","")
    lyrics = lyrics.replace("ø","")
    lyrics = lyrics.replace("å","")
    lyrics = lyrics.replace("*","")
    return sentence.lower()

# Cleaning stop words for each song's lyrics
def clean_puta_words(lyrics):
    for word in lyrics.split(): 
        if word in en_stops: # makes it O(1) because it's a Set (unique values), just like hashamp
            lyrics = lyrics.replace(word, "")
    return lyrics

# count top 10 most common words for lyrics of one song
def count_words(lyrics): 
    word_count = collections.Counter(lyrics.split())
    return word_count.most_common(10)

# cleaning all songs' lyrics
dataframe.lyrics.apply(lambda lyrics: clean_data(lyrics))
dataframe.lyrics.apply(lambda lyrics: clean_puta_words(lyrics))

# making a new column for each song and counts the most used words for each song
dataframe['most_used_words'] = pd.Series(count_words(lyrics) for _, lyrics in dataframe['lyrics'].iteritems()])
