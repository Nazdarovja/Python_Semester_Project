import numpy as np
import math,random
from tqdm import tqdm

def step_function(x):
    return 1 if x >= 0 else 0

def perceptron_output(weights, bias, x):
    '''Returns 1 if the perceptrion 'fires', 0 if not '''
    return step_function(np.dot(weights, x) + bias)

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(np.dot(weights, inputs))

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
    hidden_deltas = [hidden_output * (1 - hidden_output) * np.dot(output_deltas, [n[i] for n in network[-1]])for i, hidden_output in enumerate(hidden_outputs)]
        
    # adjust weights for hidden layer, one neuron at a time
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

def train(inputs, targets, training_iterations=1000):
    print("Training network...:")
    ###########
    # Opsætning af Neural Network
    ###########
    random.seed(0) # to get repeatable results
    input_size = 7 # antal af input noder (samme antal som feautures)
    num_hidden = 5 # antal af hidden noder
    output_size = 7 # antal af output noder (i vores tilfælde, genres)


    # each hidden neuron has one weight per input, plus a bias weight
    hidden_layer = [[random.random() for __ in range(input_size + 1)] for __ in range(num_hidden)]

    # each output neuron has one weight per hidden neuron, plus a bias weight
    output_layer = [[random.random() for __ in range(num_hidden + 1)] for __ in range(output_size)]

    # the network starts out with random weights
    network = [hidden_layer, output_layer]

    for __ in  tqdm(range(training_iterations)):
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)

    return network