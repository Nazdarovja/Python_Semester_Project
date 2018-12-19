# Song Genre Classification by Lyrics using Neural Network
A Neural Network / Machine Learning implementation to classify song lyrics by genre

## Prerequisites / Dependency requirements
````
- Anacondas Python distribution
- langdetect - Google language detection lib
- nltk - Natural language toolkit lib
- Tensorflow - Machine Learning Framework
- textblob - Natural Language Processing
````

## [MISSING] How to Run
````
stuff here
````
## About this project - "Handmade" Neural Network
 
### Project Idea
Our initial idea was to investigate, if we could build a Neural Network / Machine Learning Tool, to classify song lyrics by genre, by identifying certain characteristics about the song lyrics (features), and use these features to train our network. 

### Project Description and Artifacts
#### Dataset
To be able to check the identified genres form our neural network, we needed a dataset of song lyrics, with an already noted genre. Therefore, we chose to obtain a dataset of song info from kaggle.com that included just that. 

The dataset is ~100mb in size (zipped), and contains ~380,000 songs, with the information:
- song (name)
- year
- artist
- genre
- lyrics

View of the dataset struture from kaggle.com:
![dataset from kaggle](/readme_images/dataset.png)

##### Filtering of the dataset
The datset is filtered before being used in our neural network. 

- Language: We only focus on songs with English lyrics, by using a language detection library.  
- Genres: We have (arbitrarily, by the 3 largest genres) selected the following 3 genres to use:
    - Rock
    - Pop
    - Hip Hop  
- Length: We focus on songs with lyrics that are longer than 500 and less than 15,000 in characters. 
- Segment: We use 4000 random songs from each genre to train our model with (randomized), and 250 random other songs from each genre to test our model with afterwards.

#### Features
We have used a rather ad-hoc strategy to identify features from lyrics to use in our neural network, by trial and error, by intuition and by freestyling feature ideas. The networks ability to correctly classify the songs genre is directly correlated to this, and it might not be best feautes to use - we simply chose to use these:

- Word count (normalized)
- Average word length (normalized)
- Polarity
- Subjectivity
- Nouns
- Adverbs
- Verbs

### [MORE?] Improvements, further development ideas
We use a mixture of handmade functions and premade libraries, and as the focus of this project was the construction of the neural networks itself, the precise results from these feature functions have only been checked superficially.

### Feature Visualization

Song Word Count by Genre:

![1](/readme_images/graphs/word_count_pr_genre.png)

---

Song Word Count by Genre (normalized):

![2](/readme_images/graphs/nm_word_count_pr_genre.png)

---

Average Word length (normalized):

![3](/readme_images/graphs/nm_avg_word_len_1.png)

---

Sentiment Analysis by genre:

![4](/readme_images/graphs/sentiment_analysis.png)

---

Amount of Nouns by genre:

![5](/readme_images/graphs/amount_of_nouns_pr_genre.png)

Amount of Verbs by genre:

---

![6](/readme_images/graphs/amount_of_verbs_pr_genre.png)

---

Amount of Adverbs by genre:

![7](/readme_images/graphs/amount_of_adverbs_pr_genrepng.png)

---

Pie Chart of Genre Rocks average word class distribution:

![8](/readme_images/graphs/circle_diagram_rock_wordclass_distribution.png)

---

Pie Chart of Genre Hip-Hops average word class distribution:

![9](/readme_images/graphs/circle_diagram_hiphop_wordclass_distribution.png)

---

Pie Chart of Genre Pops average word class distribution:

![10](/readme_images/graphs/circle_diagram_pop_wordclass_distribution.png)


### Literature, Sources
 - "Data Science from Scratch: First Principles with Python", Chapter 18: Neural Networks, by. Joel Grus
-  Lecture Notes: "24. Neural Networks", by Rolf-Helge Pfeiffer

## Authors

* **Alexander W. Hørsted-Andersen** - *developer* - [awha86](https://github.com/awha86)
* **Mathias Bigler** - *developer* - [Zurina](https://github.com/Zurina)
* **Mikkel Emil Larsen** - *developer* - [mikkel7emil](https://github.com/mikkel7emil)
* **Stanislav Novitski** - *developer* - [Stani2980](https://github.com/Stani2980)
