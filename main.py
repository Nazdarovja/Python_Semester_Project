from src.data.make_dataset import create_dataset
from src.models.train_model import train
from src.models.predict_model import predict
from src.features.build_features import create_feature_pickles, build_homemade_network_input_list, create_labels, predict_features
import sys
import pandas as pd
import os


from src.visualization.visualize import plotting
if __name__ == "__main__":
    args = sys.argv[1:]
    
    # Choose model
    print('''
    Choose a model:
        1 - Custom Neural Network
        2 - Tensorflow Neural Network
        3 - Naive Bayes Classifier 
    ''')
    inpt = int(input())

    # Custom Neural Network
    if inpt == 1:
        

        print('''
        Choose mode:
            1 - Train network
            2 - Predict
        ''')
        inpt = int(input())

        if inpt == 1:

            # Creating the dataset, unzip, 
            training_data_df, test_data_df = create_dataset()
            training_data_df, test_data_df = create_feature_pickles(training_data_df, test_data_df)
            inputs_training = build_homemade_network_input_list(training_data_df)
            inputs_test = build_homemade_network_input_list(test_data_df)
            targets, genre_labels = create_labels(training_data_df)

            print('''
            Do you want plots of features?:
                1 - yes
                2 - no
            ''')
            plt = int(input())

            if plt == 1:
                plotting(training_data_df)
                

            network = {
                'weights': train(inputs_training,targets, 10),
                'genre_labels': genre_labels
            }
            
            count = 0
            for i, t in zip(inputs_test, test_data_df['genre']):
                test_res = predict(i, network['weights'])
                # print(test_res)
                maxnum = test_res.index(max(test_res))
                # print(t)
                if genre_labels[maxnum] == t:
                    count = count +1
                    print(f'{test_res} genre = {genre_labels[maxnum]}')
            success_rate = (count/len(inputs_test))*100
            print(f'Success Rate : {success_rate}% of {len(inputs_test)} lyrics')
            print('\nname your weight file to save it')
            inpt = input()
            pd.to_pickle(network, os.path.join('src','models','trained',f'{inpt}.pkl'))
        elif inpt == 2:
            print('''
            specify filename of trained weights to use.
            example: <FILENAME.pkl>
            ''')
            inpt = input()
            network = pd.read_pickle(os.path.join('src','models','trained',inpt))
            print('''
            paste lyrics
            ''')
            lyrics = input()
            
            dic = {'lyrics': [lyrics]}
            df = pd.DataFrame(dic)
            features = predict_features(df)
            print(features)
            print(network['genre_labels'])
            print(predict(features[0], network['weights']))