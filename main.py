import os

from src.data.make_dataset import create_dataset
from src.models.train_model import train
from src.models.predict_model import predict
from src.features.build_features import create_feature_pickles, build_homemade_network_input_list, create_labels

from src.visualization.visualize import plotting
if __name__ == "__main__":
    
    # Creating the dataset, unzip, 
    training_data_df, test_data_df = create_dataset()
    training_data_df, test_data_df = create_feature_pickles(training_data_df, test_data_df)
    inputs_training = build_homemade_network_input_list(training_data_df)
    inputs_test = build_homemade_network_input_list(test_data_df)
    targets = create_labels(test_data_df)

    network = train(inputs_training, targets, 500)

    for i in inputs_test[:50]:
    # res = predict([0.71148825065274152, 0.22561965811965812, 0.129914529914531, 0.2506896551724138, 0.0513455968010067,1,1], network)
        res = predict(i, network)
        print(res)