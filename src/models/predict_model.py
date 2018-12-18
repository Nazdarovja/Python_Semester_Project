from src.models.train_model import feed_forward

def predict(input, network):
    return feed_forward(network, input)[-1]