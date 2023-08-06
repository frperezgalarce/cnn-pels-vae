
import pandas as pd
import scipy.signal as signal
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score
import seaborn as sns

PATH_FEATURES_TRAIN = '/home/franciscoperez/Documents/GitHub/data/BIASEDFATS/Train_rrlyr-1.csv'
PATH_FEATURES_TEST = '/home/franciscoperez/Documents/GitHub/data/BIASEDFATS/Test_rrlyr-1.csv'
PATH_LIGHT_CURVES_OGLE = '/home/franciscoperez/Desktop/Code/FATS/LCsOGLE/data/'

def plot_training(epochs_range, train_loss_values, val_loss_values,train_accuracy_values,  val_accuracy_values):
    plt.figure(figsize=(12, 4))
    # Plot the loss values
    plt.subplot(1, 2, 1)
    print(epochs_range)
    print(np.unique(train_loss_values))
    print(val_loss_values)
    plt.plot(train_loss_values, label='Training Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot the accuracy values
    plt.subplot(1, 2, 2)
    print(train_accuracy_values)
    print(val_accuracy_values)
    plt.plot(train_accuracy_values, label='Training Accuracy')
    plt.plot(val_accuracy_values, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def read_light_curve_ogle(example_test, example_train, lenght_lc=100):

    numpy_array_lcus_test = np.empty((0, 0, 2)) # initialize the 3D array with zeros
    numpy_array_lcus_train =  np.empty((0, 0, 2)) 
    numpy_y_test =  np.empty((0, ), dtype=object) 
    numpy_y_train = np.empty((0, ), dtype=object)  

    numpy_array_lcus_test, numpy_y_test = insert_lc(example_test, numpy_array_lcus_test,numpy_y_test, lenght_lc=lenght_lc)
    numpy_array_lcus_train, numpy_y_train = insert_lc(example_train, numpy_array_lcus_train,numpy_y_train, lenght_lc=lenght_lc)

    return numpy_array_lcus_train, numpy_array_lcus_test, numpy_y_test, numpy_y_train

def load_light_curve_ogle():
    numpy_array_lcus_train = np.load('src/data/np_array.npy', allow_pickle=True)
    numpy_array_lcus_test = np.load('src/data/np_array.npy', allow_pickle=True)
    numpy_y_test = np.load('src/data/np_array_y.npy', allow_pickle=True) #TODO: modify
    numpy_y_train = np.load('src/data/np_array_y.npy', allow_pickle=True)
    return numpy_array_lcus_train, numpy_array_lcus_test, numpy_y_test, numpy_y_train


def insert_lc(examples, np_array, np_array_y, lenght_lc = 0, signal_noise=3):
    counter = 0
    subclasses = pd.read_csv('/home/franciscoperez/Documents/GitHub/vsbms_multiple_classes/bayesianMLP/src/data/all_subclasses')
    for lc in examples.ID.unique():
        path_lc = PATH_LIGHT_CURVES_OGLE+lc.split('-')[1].lower()+'/'+lc.split('-')[2].lower()+'/phot/I/'+lc
        lcu = pd.read_table(path_lc, sep=" ", names=['time', 'magnitude', 'error'])
        #print(lcu.dtypes)
        lcu['delta_time'] = lcu['time'].diff()
        lcu['delta_mag'] = lcu['magnitude'].diff()
        lcu = lcu[lcu.magnitude/lcu.error>signal_noise] #Delete S/N greater than 3
        lcu = lcu[lcu.delta_time<10] #Delete time between observations greater than 10 
        lcu = lcu[lcu.delta_mag!=0]
        lcu = delete_by_std(lcu)
        lcu['delta_time'] = lcu['time'].diff()
        lcu['delta_mag'] = lcu['magnitude'].diff()
        lcu.dropna(axis=0, inplace=True)

        if lcu.shape[0]> lenght_lc:
            if False:
                lcu[['delta_time', 'delta_mag']].plot.scatter(x='delta_time', y='delta_mag')
                plt.show()

            lcu_data = np.asarray(lcu[['delta_time', 'delta_mag']].head(lenght_lc))

            try:
                new_element = str(subclasses.loc[subclasses.ID==lc.replace('.dat', ''),'sub_clase'].values[0])
                np_array_y = np.append(np_array_y, new_element)
                #print('label added', np_array_y.shape)
                np_array = np.resize(np_array, (np_array.shape[0] + 1, lcu_data.shape[0], 2))
                np_array[-1] = lcu_data
                #print('data added', np_array.shape)
            except Exception as error:
                #print(error)
                #print(lc.split('-')[2])
                new_element = lc.split('-')[2]
                if new_element not in ['LPV', 'RRLYR', 'ECL']:
                    np_array_y = np.append(np_array_y, new_element)
                    #print('label added', np_array_y.shape)
                    np_array = np.resize(np_array, (np_array.shape[0] + 1, lcu_data.shape[0], 2))
                    np_array[-1] = lcu_data
                    #print('data added', np_array.shape)
                #else: 
                #    print('not added since there are subclasses for this class')
            counter = counter + 1

    print('shape: ', np_array.shape, np_array_y.shape)
    print(np_array_y)
    np.save('/home/franciscoperez/Documents/GitHub/vsbms_multiple_classes/bayesianMLP/src/data/np_array_y.npy', np_array_y)
    np.save('/home/franciscoperez/Documents/GitHub/vsbms_multiple_classes/bayesianMLP/src/data/np_array.npy', np_array)
    return np_array, np_array_y

def delete_by_std(df): 

    std_dev = df.std(axis=0)
    # get columns with 0 standard deviation
    cols_to_drop = std_dev[std_dev == 0].index

    # drop columns with 0 standard deviation
    df.drop(cols_to_drop, axis=1, inplace=True)
    return df

def delete_by_magnitude(df):
    df = df.loc[(df.delta_mag != 0).all(axis=1)]
    return df

def delete_low_represented_classes(df, column='class', threshold=100): 
    # Count the number of samples per category
    counts = df.groupby(column).size()
    # Filter out the categories with less than 200 samples
    categories_to_keep = counts[counts >= threshold].index
    df_filtered = df[df[column].isin(categories_to_keep)]
    return df_filtered

def plot_cm(cm, labels, title='Confusion matrix'):
    plt.figure(figsize=(8,8))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Add text annotations
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = int(np.round(cm[i, j],2)*100)  # Convert float to integer
        plt.text(j, i, format(value, 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    # Add labels to the x-axis and y-axis
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    plt.tight_layout()
    plt.show()

def get_ids(n=1): 
    path_train = PATH_FEATURES_TRAIN
    path_test = PATH_FEATURES_TEST
    lc_test = pd.read_table(path_test, sep= ',')
    #lc_test = lc_test[lc_test.label=='ClassA']
    lc_train = pd.read_table(path_train, sep= ',')
    #lc_train = lc_train[lc_train.label=='ClassA']
    example_test  = lc_test[['ID']]
    example_train = lc_train.sample(n)[['ID']]
    print(example_test)


    new_cols_test = example_test.ID.str.split("-", n = 3, expand = True)
    new_cols_test.columns = ['survey', 'field', 'class', 'number']
    # concatenate the new columns with the original DataFrame
    example_test = pd.concat([new_cols_test, example_test], axis=1)
    print(example_test.head())
    example_test = delete_low_represented_classes(example_test, column='class', threshold=100)

    new_cols_train = example_train.ID.str.split("-", n = 3, expand = True)
    new_cols_train.columns = ['survey', 'field', 'class', 'number']

    # concatenate the new columns with the original DataFrame
    print(new_cols_train.shape)
    print(example_train.shape)

    example_train = pd.concat([new_cols_train, example_train], axis=1)
    print(example_train.shape)
    print('clases: ', example_train['class'].unique())

    example_train = delete_low_represented_classes(example_train, column='class', threshold=100)

    print('clases: ', example_train['class'].unique())
    example_test['class'] = pd.factorize(example_test['class'])[0]
    example_train['class'] = pd.factorize(example_train['class'])[0]

    print('clases: ', example_train['class'].unique())

    return example_test, example_train

def get_data(sample_size=50000, mode='load'):
    if mode=='create':
        x_test, x_train  = get_ids(n=sample_size) 
        x_train, x_test, y_test, y_train  = read_light_curve_ogle(x_test, x_train)
    elif mode == 'load':
        x_train, x_test, y_test, y_train  = load_light_curve_ogle()


    # Create a label encoder
    label_encoder = LabelEncoder()

    # Encode the string labels to numerical values
    encoded_labels = label_encoder.fit_transform(y_train)
    encoded_labels_test = label_encoder.fit_transform(y_test)


    # Convert the encoded labels to a PyTorch tensor
    y_train = torch.from_numpy(encoded_labels).long()
    y_test = torch.from_numpy(encoded_labels_test).long()

    # Convert the data to PyTorch tensors
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()

    # Permute the dimensions of the input tensor
    x_train = x_train.permute(0, 2, 1)
    x_test = x_test.permute(0, 2, 1)

    # Print the shapes of the data
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    y_train, y_test = torch.from_numpy(np.asarray(y_train)), torch.from_numpy(np.asarray(y_test))

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

    return x_train, x_test,  y_train, y_test, x_val, y_val, label_encoder


def export_recall_latex(true_labels, predicted_labels, label_encoder): 
    # Calculate the recall for each class
    recall_values = recall_score(true_labels, predicted_labels, average=None)

    # Convert recall values to LaTeX table format
    latex_table = "\\begin{tabular}{|c|c|}\n\\hline\nClass & Recall \\\\\n\\hline\n"

    for i, recall in enumerate(recall_values):
        class_decoded = label_encoder.inverse_transform([i])
        latex_table += f"Class {class_decoded} & {recall:.2f} \\\\\n"

    latex_table += "\\hline\n\\end{tabular}"

    # Print the LaTeX table
    print(latex_table)


