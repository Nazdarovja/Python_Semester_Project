from matplotlib import pyplot as plt
import numpy as np
import math,random
from tqdm import tqdm
from random import shuffle
import pandas as pd
from .features.build_features import word_count, sentence_avg_word_length, normalize
from .features.text_blob_analysis import analyze_sentiment, analyze_word_class
from .data.make_dataset import create_dataset
from .data.util import unzip_file

def step_function(x):
    return 1 if x >= 0 else 0

def perceptron_output(weights, bias, x):
    '''Returns 1 if the perceptrion 'fires', 0 if not '''
    return step_function(np.dot(weights, x) + bias)

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(np.dot(weights, inputs))

def predict(input, network):
    return feed_forward(network, input)[-1]

def feed_forward(neural_network, input_vector):
    """takes in a neural network (represented as a list of lists of lists of weights)
    and returns the output from forward-propagating the input"""

    outputs = []

    for layer in neural_network:

        input_with_bias = input_vector + [1]             # add a bias input
        output = [neuron_output(neuron, input_with_bias) # compute the output
                for neuron in layer]                   # for this layer
        outputs.append(output)                           # and remember it

        # the input to the next layer is the output of this one
        input_vector = output

    return outputs
    
def backpropagate(network, input_vector, targets):
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # the output * (1 - output) is from the derivative of sigmoid
    output_deltas = [output * (1 - output) * (output - target) for output, target in zip(outputs, targets)]
        # adjust weights for output layer, one neuron at a time
    for i, output_neuron in enumerate(network[-1]):
    # focus on the ith output layer neuron
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            # adjust the jth weight based on both
            # this neuron's delta and its jth input
            output_neuron[j] -= output_deltas[i] * hidden_output
    # back-propagate errors to hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) * np.dot(output_deltas, [n[i] for n in output_layer])for i, hidden_output in enumerate(hidden_outputs)]
        
    # adjust weights for hidden layer, one neuron at a time
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

def train(inputs, targets, network):
    print("Training network...:")
    for __ in  tqdm(range(1000)):
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)

    # res = predict([0.71148825065274152, 0.22561965811965812, 0.129914529914531, 0.2506896551724138, 0.0513455968010067], network)
    # print(res)

    return network



test_df, df = create_dataset()
# df, test_df = create_dataset()
df = df.sample(frac=1).reset_index(drop=True)

###########
# Opsætning af Neural Network
###########
random.seed(0) # to get repeatable results
input_size = 7 # antal af input noder (samme antal som feautures)
num_hidden = 4 # antal af hidden noder
output_size = 3 # antal af output noder (i vores tilfælde, genres)

series = df['genre'].value_counts()
genre_labels = series.keys() # getting genre labels

# adding features as series to the dataframe
df = sentence_avg_word_length(df,"avg_word_len", 'lyrics')
df = normalize(df, 'avg_word_len_nm', 'avg_word_len')
df = word_count(df,"word_count", 'lyrics')
df = normalize(df, 'word_count_nm', 'word_count')
df = analyze_sentiment(df)
df = analyze_word_class(df)

# grapping features for training
avg_word_len = df['avg_word_len_nm']
words = df["word_count_nm"]
polarity = df['polarity']
subjectivity = df['subjectivity']
nouns = df['nouns']
adverbs = df['adverbs']
verbs = df['verbs']

#inputs & targets
targets = [[1 if i == j else 0 for i in genre_labels] for j in df['genre']]
inputs = [[f, p, s, n, a, v, wl] for f, p, s, n, a, v, wl in zip(words, polarity, subjectivity, nouns, adverbs, verbs, avg_word_len)]

# each hidden neuron has one weight per input, plus a bias weight
hidden_layer = [[random.random() for __ in range(input_size + 1)] for __ in range(num_hidden)]
# each output neuron has one weight per hidden neuron, plus a bias weight
output_layer = [[random.random() for __ in range(num_hidden + 1)] for __ in range(output_size)]
# the network starts out with random weights
network = [hidden_layer, output_layer]

network = train(inputs, targets, network)


######################
# TESTING
######################

# features
test_df = test_df.copy()
test_df = test_df.sample(100)

test_df = sentence_avg_word_length(test_df,"avg_word_len", 'lyrics')
test_df = normalize(test_df, 'avg_word_len_nm', 'avg_word_len')

test_df = word_count(test_df,"word_count", 'lyrics')
test_df = normalize(test_df, 'word_count_nm', 'word_count')
test_df = analyze_sentiment(test_df)
test_df = analyze_word_class(test_df)

avg_word_len = test_df['avg_word_len_nm']
words = test_df["word_count_nm"]
polarity = test_df['polarity']
subjectivity = test_df['subjectivity']
nouns = test_df['nouns']
adverbs = test_df['adverbs']
verbs = test_df['verbs']

# Create feature list
test_features = [[f, p, s, n, a, v, wl] for f, p, s, n, a, v, wl in zip(words, polarity, subjectivity, nouns, adverbs, verbs, avg_word_len)]

# Testing
print("Testing...:") 

count = 0
for test, t in zip(test_features, test_df['genre']):
    test_res = predict(test,network)
    
    maxnum = test_res.index(max(test_res))
    if genre_labels[maxnum] == t:
        count = count +1
        print(test_res)
        print(t)
        
print(count)