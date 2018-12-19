from src.data.make_dataset import create_dataset
from src.models.train_model import train
from src.models.predict_model import predict
from src.features.build_features import create_feature_pickles, build_homemade_network_input_list, create_labels, predict_features
import sys
import pandas as pd
import os


from src.visualization.visualize import plotting
if __name__ == "__main__":

    print('''
    Choose mode (Custom Neural Network):
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
            'weights': train(inputs_training,targets, 1000),
            'genre_labels': genre_labels
        }
        
        count = 0
        for i, t in zip(inputs_test, test_data_df['genre']):
            test_res = predict(i, network['weights'])
            maxnum = test_res.index(max(test_res))
            if genre_labels[maxnum] == t:
                count = count +1
                print(f'{test_res} genre = {genre_labels[maxnum]}')
        
        success_rate = (count/len(inputs_test))*100
        print(f'Success Rate : {success_rate}% of {len(inputs_test)} lyrics')
        print('\nSupply filename for weights file :')
        inpt = input()
        
        pd.to_pickle(network, os.path.join('src','models','trained',f'{inpt}.pkl'))
        print('*********************************************************************************************\n')
        print('*        Please run again and choose 2 for predict and use your trained network             *\n')
        print('*********************************************************************************************\n')

    elif inpt == 2:
        print('*********************************************************************************************')
        print('''
        specify filename of trained weights to use.
        example: <FILENAME.pkl>
        (existing files are listed below)
        ''')
        print(str(os.listdir(os.path.join('src','models','trained'))))
        inpt = input()
    
        network = pd.read_pickle(os.path.join('src','models','trained',inpt))
        while True:
            print('*********************************************************************************************')
            print('''
            Paste lyrics to predict (one line of lyrics multiple lines are not supported)
            Press Enter/Return after for result
            ''')
            lyrics = input()
            
            dic = {'lyrics': [lyrics]}
            df = pd.DataFrame(dic)
            features = predict_features(df)
            
            labels = network['genre_labels']
            res = predict(features[0], network['weights'])
            for_print = [f'{l} = {r}' for l, r in zip(labels, res) ]
            
            print('\n\n\n*********************************************************************************************\n')
            print('\n'.join(for_print))
            print(f'\nThis is most likely a {labels[res.index(max(res))]} song')
            print('\n*********************************************************************************************')