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
    targets, genre_labels = create_labels(training_data_df)

    print(inputs_training[10:20])
    network = train(inputs_training, targets, 10)

    count = 0
    for i, t in zip(inputs_test, test_data_df['genre']):
        test_res = predict(i, network)
        # print(test_res)
        maxnum = test_res.index(max(test_res))
        # print(t)
        if genre_labels[maxnum] == t:
            count = count +1
            print(f'{test_res} genre = {genre_labels[maxnum]}')
    success_rate = (count/len(inputs_test))*100
    print(f'Success Rate : {success_rate}% of {len(inputs_test)} lyrics')