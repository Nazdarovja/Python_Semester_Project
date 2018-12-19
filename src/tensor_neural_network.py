# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os

def prepare_data(df, data_cols, label_col, training_size=1000, test_size=250):
    labels = df_cp[label_col].value_counts().keys().tolist()
    train_data, train_labels, test_data, test_labels = [], [], [], []
    
    # shuffle dataset
    df = df.copy().sample(frac=1).reset_index(drop=True)
    
    for label in labels:
        data = df[df[label_col] == label]
        # kun hvis der er nok eksempler, ift. training_size og test_size, ud fra den pågældende label
        if len(data) > training_size + test_size:
            data = data.reset_index(drop=True)
            train_data += data[data_cols][0:training_size].values.tolist()
            train_labels += data[label_col][0:training_size].values.tolist()
            test_data += data[data_cols][training_size:training_size+test_size].values.tolist()
            test_labels += data[label_col][training_size:training_size+test_size].values.tolist()
    
    # da modellen kun kan trænes med numpy arrays, så skal listerne lige konverteres
    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels)
    test_data = np.asarray(test_data)
    test_labels = np.asarray(test_labels)
    
    return (train_data, train_labels), (test_data, test_labels)

def train_and_save(df, checkpoint_path):
    checkpoint_dir = os.path.join('src','models', checkpoint_path)

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                    save_weights_only=True,
                                                    verbose=1)

    model = create_model()

    model.fit(train_images, train_labels,  epochs = 10, 
            validation_data = (test_images,test_labels),
            callbacks = [cp_callback])