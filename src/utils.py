import os, re, glob
import socket
import yaml
import datetime
import torch
import numpy as np
import pandas as pd
import matplotlib
if socket.gethostname() == 'exalearn':
    matplotlib.use('agg')
from tqdm import tqdm
from collections import OrderedDict
from src.vae.vae_models import *
from src.visualization import plot_cm
from sklearn.model_selection import train_test_split
import pickle
from typing import  Any, Dict, List
import logging
import gzip
from concurrent.futures import ThreadPoolExecutor
import random 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

import time 

logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

with open('src/configuration/paths.yaml', 'r') as file:
    YAML_FILE = yaml.safe_load(file)

PATHS =YAML_FILE['paths']
PATH_LIGHT_CURVES_OGLE = PATHS['PATH_LIGHT_CURVES_OGLE']
PATH_FEATURES_TRAIN = PATHS['PATH_FEATURES_TRAIN']
PATH_FEATURES_TEST = PATHS['PATH_FEATURES_TEST']
PATH_NUMPY_DATA_X_TRAIN = PATHS['PATH_NUMPY_DATA_X_TRAIN']
PATH_NUMPY_DATA_X_TEST = PATHS['PATH_NUMPY_DATA_X_TEST'] 
PATH_NUMPY_DATA_Y_TRAIN = PATHS['PATH_NUMPY_DATA_Y_TRAIN']
PATH_NUMPY_DATA_Y_TEST = PATHS['PATH_NUMPY_DATA_Y_TEST'] 
PATH_SUBCLASSES = PATHS["PATH_SUBCLASSES"]
PATH_DATA_FOLDER = PATHS["PATH_DATA_FOLDER"]
PATH_FIGURES: str = PATHS['PATH_FIGURES']
PATH_MODELS: str = PATHS["PATH_MODELS"]
PATH_ZIP_GAIA:str = PATHS["PATH_ZIP_GAIA"]

# Read configurations from a YAML file
with open('src/configuration/regressor.yaml', 'r') as file:
    reg_conf_file: Dict[str, Any] = yaml.safe_load(file)

with open('src/configuration/nn_config.yaml', 'r') as file:
    nn_config = yaml.safe_load(file)

data_sufix: str =   reg_conf_file['model_parameters']['sufix_path']  

if 'LOG' in data_sufix:
    MIN_PERIOD_VALUE = 0.1
else: 
    MIN_PERIOD_VALUE = np.log(0.1)

def read_light_curve_ogle(example_test, example_train, values_count, lenght_lc=100):
    
    numpy_array_lcus_test = np.empty((0, 0, 2))
    numpy_array_lcus_train =  np.empty((0, 0, 2))
    numpy_y_test =  np.empty((0, ), dtype=object)
    numpy_y_train = np.empty((0, ), dtype=object)
    sn_ratio_test = []
    sn_ratio_train = [] 

    def task1():
        return insert_lc(example_test, numpy_array_lcus_test,
                         numpy_y_test, lenght_lc=lenght_lc, 
                         signal_noise=nn_config['data']['sn_ratio'], 
                         train_set=False, file_name='test')
    
    def task2():
        return insert_lc(example_train, numpy_array_lcus_train,
                         numpy_y_train, values_count=values_count, 
                         signal_noise=nn_config['data']['sn_ratio'], 
                         lenght_lc=lenght_lc, train_set=True, file_name='train')
    
    with ThreadPoolExecutor() as executor:
        future1 = executor.submit(task1)
        future2 = executor.submit(task2)
        
        numpy_array_lcus_test, numpy_y_test, sn_ratio_test = future1.result()
        numpy_array_lcus_train, numpy_y_train, sn_ratio_train = future2.result()

        sn_ratio = sn_ratio_test + sn_ratio_train
        pd.DataFrame(sn_ratio).to_csv('sn_ratio_sample.csv')

    return numpy_array_lcus_train, numpy_array_lcus_test, numpy_y_test, numpy_y_train

def load_light_curve_ogle():
    print('Loading light curves')
    #Implement concurrent process

    numpy_array_lcus_train = np.load(PATH_NUMPY_DATA_X_TRAIN, allow_pickle=True)
    numpy_array_lcus_test = np.load(PATH_NUMPY_DATA_X_TEST, allow_pickle=True)
    numpy_y_train = np.load(PATH_NUMPY_DATA_Y_TRAIN, allow_pickle=True)
    numpy_y_test = np.load(PATH_NUMPY_DATA_Y_TEST, allow_pickle=True)
    print('loaded files')
    print('training data shape: ', numpy_array_lcus_train.shape)
    print('testing data shape: ', numpy_array_lcus_test.shape)
    return numpy_array_lcus_train, numpy_array_lcus_test, numpy_y_test, numpy_y_train

def light_curve_preprocessing(lcu):
    # Drop NaN values
    lcu.dropna(axis=0, inplace=True)
    
    # Calculate delta_time and delta_mag
    lcu['delta_time'] = lcu['time'].diff()
    lcu['delta_mag'] = lcu['magnitude'].diff()

    # Calculate quantiles once for efficiency
    quantile_99 = lcu['delta_mag'].quantile(0.999)
    quantile_001 = lcu['delta_mag'].quantile(0.001)
    delta_time_max = nn_config['data']['delta_time_max']
    
    # Filter by quantiles and delta_time_max
    lcu = lcu[(lcu['delta_mag'] < quantile_99) & (lcu['delta_mag'] > quantile_001)]
    lcu = lcu[lcu['delta_time'] < delta_time_max]
    
    # Filter by delta_time quantile
    quantile_time_001 = lcu['delta_time'].quantile(0.001)
    lcu = lcu[lcu['delta_time'] > quantile_time_001]
    
    # Call delete_by_std function if needed
    lcu = delete_by_std(lcu)
    
    return lcu

def add_ell(new_element, ELL, lc): 
    if (new_element in ['ECL']): 
        type_value = ELL.loc[ELL.ID==lc.replace('.dat', ''), 'Type'].values
        if type_value and type_value=='ELL':
            new_element = 'ELL' 
        else: 
            new_element = 'ECL'
    return new_element

def update_arrays(np_array, np_array_y, new_element, lcu_data, counter, verbose = False): 
    
    if verbose: 
        print('original shape: ', np_array.shape, np_array_y.shape)
    np_array_y = np.append(np_array_y, new_element)
    np_array = np.resize(np_array, (np_array.shape[0] + 1, lcu_data.shape[0], 2))
    np_array[-1] = lcu_data
    counter = counter + 1
    if verbose: 
        print('original shape: ', np_array.shape, np_array_y.shape)

    return np_array, np_array_y, counter 

def insert_lc(examples, np_array, np_array_y, values_count= None, lenght_lc = 0, signal_noise=6, 
            train_set=True, file_name='train', verbose = False, oversampling=False):
    counter = 0
    ELL = pd.read_table(PATH_DATA_FOLDER+'/ELL.txt')
    
    snr_list = []

    for lc in tqdm(examples.ID.unique(), desc='Processing Light Curves'):
        if train_set:
            class_counter = values_count[values_count.Type==lc.split('-')[2]].counter.values[0]
        else: 
            class_counter = np.iinfo(np.int64).max

        path_lc = PATH_LIGHT_CURVES_OGLE+lc.split('-')[1].lower()+'/'+lc.split('-')[2].lower()+'/phot/I/'+lc
        
        lcu = pd.read_table(path_lc, sep=" ", names=['time', 'magnitude', 'error'])
        
        lcu = light_curve_preprocessing(lcu)

        signal = lcu['magnitude'].max() - lcu['magnitude'].min()

        condition1 = (lcu.shape[0]> lenght_lc)
        condition2 = ('error' in lcu.columns)

        if 'error' in lcu.columns and not lcu['error'].isnull().all():
            snr_list.append([(signal)/lcu['error'].mean(), lc])

        if condition2: 
            condition3 = ((signal)/lcu['error'].mean()>signal_noise)
        else: 
            condition3 = False

        if verbose:
            print(lc, condition1, condition2, condition3)

        if condition1 and condition2 and condition3:
            condition4 =  (class_counter > nn_config['data']['limit_to_define_minority_Class'])
            if verbose:
                print(train_set, condition4)
                print(class_counter, nn_config['data']['limit_to_define_minority_Class'])
            
            if (train_set) or condition4:
                lcu_data = np.asarray(lcu[['delta_time', 'delta_mag']].sample(lenght_lc))
                try: 
                    new_element = lc.split('-')[2]
                    if verbose: 
                        print(new_element)
                    new_element = add_ell(new_element, ELL, lc)
                    if new_element in list(nn_config['data']['classes']):
                        np_array, np_array_y, counter = update_arrays(np_array, np_array_y, new_element, lcu_data, counter)
                except Exception as error: 
                    logging.error(f"The light curve {lc} was not loaded: {error}")
            else:
                if oversampling:
                    k_samples_by_object = int(nn_config['data']['upper_limit_majority_classes']
                                            /(class_counter)) 
                else: 
                    k_samples_by_object = 1
                if verbose: 
                    print(k_samples_by_object)

                for _ in range(k_samples_by_object):

                    first_positive_index = lcu[lcu['delta_mag'] > 0].index[0]
                    filtered_df = lcu.loc[:first_positive_index - 1]

                    lcu_data = np.asarray(filtered_df[['delta_time', 'delta_mag']]
                                                    .sample(lenght_lc)
                                                    .sort_index())
                    try: 
                        new_element = lc.split('-')[2]
                        new_element = add_ell(new_element, ELL, lc)
                        if new_element in list(nn_config['data']['classes']):
                            np_array, np_array_y, counter = update_arrays(np_array, 
                                                                          np_array_y, 
                                                                          new_element, 
                                                                          lcu_data, counter)
                    except Exception as error: 
                        logging.error(f"The light curve {lc} was not loaded: {error}")
                        
    print('shape: ', np_array.shape, np_array_y.shape)
    np.save(PATH_DATA_FOLDER+'/'+file_name+'_np_array_y.npy', np_array_y)
    np.save(PATH_DATA_FOLDER+'/'+file_name+'_np_array.npy', np_array)
    return np_array, np_array_y, snr_list

def delete_by_std(df): 

    std_dev = df.std(axis=0)
    # get columns with 0 standard deviation
    cols_to_drop = std_dev[std_dev == 0].index

    # drop columns with 0 standard deviation
    df.drop(cols_to_drop, axis=1, inplace=True)
    return df

def delete_low_represented_classes(df, column='class', threshold=100): 
    # Count the number of samples per category
    counts = df.groupby(column).size()
    # Filter out the categories with less than 200 samples
    categories_to_keep = counts[counts >= threshold].index
    df_filtered = df[df[column].isin(categories_to_keep)]
    return df_filtered

def move_data_to_device(data, device):
    return tuple(torch.tensor(d).to(device) if isinstance(d, np.ndarray) else d.to(device) for d in data)

def custom_sample(df, class_column, threshold, n):
    """
    df: The input dataframe
    class_column: The name of the column which contains the class labels
    threshold: The threshold which decides if a class is low or high represented
    n: Number of samples desired for overrepresented classes
    """
    
    # Count the occurrences of each class
    class_counts = df[class_column].value_counts()
    
    # Identify low and high represented classes
    low_represented = class_counts[class_counts <= threshold].index
    high_represented = class_counts[class_counts > threshold].index
    
    sampled_dataframes = []
    
    # Include all instances for low represented classes
    for class_value in low_represented:
        sampled_dataframes.append(df[df[class_column] == class_value])
    
    # Sample n instances for high represented classes
    for class_value in high_represented:
        sample_size = min(n, class_counts[class_value]) # To handle cases where n is greater than class count
        sampled_dataframes.append(df[df[class_column] == class_value].sample(sample_size))
    
    # Concatenate the results and return
    return pd.concat(sampled_dataframes)

def get_ids(n=1, iteration=0): 
    path_train = PATH_FEATURES_TRAIN
    path_test = PATH_FEATURES_TEST

    lc_test = pd.read_table(path_test, sep= ',')
    lc_train = pd.read_table(path_train, sep= ',')

    example_test  = lc_test[['ID']]
    new_cols_test = example_test.ID.str.split("-", n = 3, expand = True)
    new_cols_test.columns = ['survey', 'field', 'class', 'number']

    example_test = pd.concat([new_cols_test, example_test], axis=1)
    example_test = delete_low_represented_classes(example_test, column='class', 
                                                  threshold=20)

    new_cols_train = lc_train.ID.str.split("-", n = 3, expand = True)
    new_cols_train.columns = ['survey', 'field', 'class', 'number']
    lc_train = pd.concat([new_cols_train, lc_train], axis=1)
    lc_train = delete_low_represented_classes(lc_train, column='class', threshold=20)

    '''
    if n>=lc_train.shape[0]:
        example_train =lc_train[['ID','survey', 'field', 'class', 'number']]
    else:
        frac = n/lc_train.shape[0]
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=iteration)
        for _, sample_idx in strat_split.split(lc_train, lc_train['class']):
            example_train = lc_train.iloc[sample_idx][['ID','survey', 'field', 'class', 'number']]
    ''' 
    print(n, lc_train.shape[0])
    if n >= lc_train.shape[0]:
        example_train = lc_train[['ID', 'survey', 'field', 'class', 'number']]
    else:
        example_train, _ = train_test_split(
            lc_train, 
            train_size=n, 
            stratify=lc_train['class'], 
            random_state=iteration
            )


    values_count = example_train['class'].value_counts().reset_index()
    values_count.columns = ['Type', 'counter']
    
    example_train = example_train[example_train['class'].isin(list(nn_config['data']['classes']))]
    example_test = example_test[example_test['class'].isin(list(nn_config['data']['classes']))]

    return example_test, example_train, values_count

def load_id_period_to_sample(classes: List[str] = [], period: List[float] = []) -> pd.DataFrame:
    """
    Load and sample data based on specified classes and periods.

    This function loads data from a specified path, filters and samples it according to the given classes and period criteria.
    It supports handling of log-transformed period values.

    Parameters:
    classes (List[str]): List of classes to filter the data. If empty, a random sample is returned.
    period (List[float]): List of period values to use for filtering, corresponding to each class.

    Returns:
    pd.DataFrame: A DataFrame containing the sampled data based on the provided criteria.
    """
    PATH_ZIP_LCs = (PATH_DATA_FOLDER + '/time_series/real/OGLE3_lcs_I_meta_snr5_augmented_folded_trim600_' + 
                   data_sufix + '.npy.gz')
    print('Loading from:\n', PATH_ZIP_LCs)
    with gzip.open(PATH_ZIP_LCs, 'rb') as f:
        np_data = np.load(f, allow_pickle=True)

    df = np_data.item()['meta'][['OGLE_id', 'Period', 'Type']]

    black_list = pd.read_csv('data/black_list.csv')

    df = df[~df.OGLE_id.isin(black_list.ID.to_list())]
    if len(classes) == 0: 
        raise('There is not a label for sampling')
    else:
        samples = []
        counter = 0
        for t in classes:
            sample = pd.DataFrame()
            if t == 'ELL':
                sample = df[df['Type'] == 'ECL'].sample(n=1)
            else:
                filtered_df = df[(df['Type'] == t)]
                closest_idx = (filtered_df['Period'] - period[counter]).abs().nsmallest(1).index
                sample = filtered_df.loc[closest_idx]
            samples.append(sample)
            counter += 1
        df = pd.concat(samples, axis=0).reset_index(drop=True)
    return df

def transform_to_consecutive(input_list, label_encoder):
    unique_elements = sorted(set(input_list))
    modified_labelencoder_classes = [element for idx, element in enumerate(label_encoder.classes_) if idx in unique_elements]
    mapping = {value: index for index, value in enumerate(unique_elements)}
    modified_labels = np.array([mapping[element] for element in input_list])
    return modified_labels, modified_labelencoder_classes

def get_data(sample_size, mode, iteration):

    with open('src/configuration/nn_config.yaml', 'r') as file:
        nn_config = yaml.safe_load(file)

    print('-'*50)
    print('MODE DATA: ', mode)

    if mode=='create':
        id_test, id_train, values_count  = get_ids(n=sample_size, iteration = iteration) 
        x_train, x_test, y_test, y_train  = read_light_curve_ogle(id_test, id_train, values_count, 
                                                                 lenght_lc= nn_config['data']['seq_length'])
    elif mode == 'load':
        x_train, x_test, y_test, y_train  = load_light_curve_ogle()
    else: 
        raise('Mode not valid. Please check mode_running in nn_config. Use create or load strings.')
    
    print('Loaded shape training set: ', x_train.shape)
    print('Loaded shape testing set: ', x_test.shape)
    if int(x_train.shape[1])!=2:
        x_train = x_train.transpose(0, 2, 1)
        x_test = x_test.transpose(0, 2, 1)
    print('Modified shape training set: ', x_train.shape)
    print('Modified shape testing set: ', x_test.shape)

    with open(PATH_MODELS+'label_encoder_vae.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
    
    encoded_labels = label_encoder.transform(y_train)
    encoded_labels_test = label_encoder.transform(y_test)
    encoded_labels, modified_labelencoder_classes = transform_to_consecutive(encoded_labels, label_encoder)

    encoded_labels_test, modified_labelencoder_classes = transform_to_consecutive(encoded_labels_test, label_encoder)
    n_values = len(np.unique(encoded_labels))
    y_train = np.eye(n_values)[encoded_labels]
    y_test = np.eye(n_values)[encoded_labels_test]

    with open(PATH_MODELS+'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    print("Label to Number Mapping:")
    for index, label in enumerate(modified_labelencoder_classes):
        print(f"{label}: {index}")

    nn_config['data']['classes'] = modified_labelencoder_classes

    with open('src/configuration/nn_config.yaml', 'w') as file:
        yaml.safe_dump(nn_config, file)
    time.sleep(2)

    y_labels = torch.from_numpy(encoded_labels).long()
    y_labels_test = torch.from_numpy(encoded_labels_test).long()
    y_train_onehot = torch.from_numpy(y_train).long()
    y_test_onehot = torch.from_numpy(y_test).long()

    # Convert the data to PyTorch tensors
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()

    y_train, y_test = torch.from_numpy(np.asarray(y_train_onehot)), torch.from_numpy(np.asarray(y_test_onehot))
 
    x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size=0.3, random_state=42, stratify=y_labels)

    return x_train, x_test,  y_train, y_test, x_val,\
           y_val, modified_labelencoder_classes,\
           y_labels, y_labels_test

path = os.path.dirname(os.getcwd())+'/cnn-pels-vae'

def load_yaml_priors(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def extract_maximum_of_max_periods(configuration):
    maximum_periods = []
    
    # Extract maximum periods for each category
    for _ , properties in configuration.items():
        print(properties)
        if 'Period' in properties:
            print(properties['Period'])
            max_period = max(properties['Period'])
            maximum_periods.append(max_period)
    
    # Find the maximum of all maximum periods
    max_of_max_periods = max(maximum_periods)
    
    return max_of_max_periods

def extract_midpoints(class_data):
    # Initialize the result list
    result = []

    # Iterate through the subclasses and extract midpoints
    for subclass_name, subclass_data in class_data.items():
        if (subclass_name == "CompleteName") or (subclass_name == "max_period") or (subclass_name == "min_period"):
            continue
        #print(subclass_name)
        teff_mid = subclass_data['Midpoints']['teff_val']
        period_mid = subclass_data['Midpoints']['Period']
        admagg_mid = subclass_data['Midpoints']['abs_Gmag']
        feh_95_mid = subclass_data['Midpoints']['FeH_J95']
        radius_val_mid = subclass_data['Midpoints']['radius_val']
        logg_mid = subclass_data['Midpoints']['logg']

        result.append([ period_mid, teff_mid, feh_95_mid, admagg_mid,  radius_val_mid, logg_mid])

    return result    

## return number of trainable parameters in the model
def count_parameters(model):
    """Calculate the number of trainable parameters of a Pytorch moel.

    Parameters
    ----------
    model : pytorh model
        Pytorch model

    Returns
    -------
    int
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

## normalize light curves
def normalize_each(data, norm_time=False, scale_to=[0, 1], n_feat=3):
    """MinMax normalization of all light curves per item.

    Parameters
    ----------
    data      : numpy array
        Light curves to be normalized
    norm_time : bool array, optional
        Wheather to normalize time axis or not, default=False
    scale_to  : list, optional
        Normalize range [min, max]
    n_feat    : int, optional
        numeber of features to be normalized

    Returns
    -------
    normed
        Normalized light curves
    """
    normed = np.zeros_like(data)
    for i, lc in enumerate(data):
        for f in range(n_feat):
            normed[i, :, f] = lc[:, f]
            ## normalize time if asked
            if f == 0 and norm_time:
                normed[i, :, f] = (lc[:, f] - np.min(lc[:, f])) / \
                                  (np.max(lc[:, f]) - np.min(lc[:, f]))
            ## normalize other feature values
            if f == 1:
                normed[i, :, f] = (lc[:, f] - np.min(lc[:, f])) / \
                                  (np.max(lc[:, f]) - np.min(lc[:, f]))
            if f == 2:
                normed[i, :, f] = (lc[:, f]) / \
                                  (np.max(lc[:, f-1]) - np.min(lc[:, f-1]))
            ## scale feature values if asked
            if scale_to != [0, 1]:
                if f == 0 and not norm_time: continue
                if f == 2:
                    normed[i, :, f] = normed[i, :, f] * (scale_to[1] -
                                                         scale_to[0])
                else:
                    normed[i, :, f] = normed[i, :, f] * \
                        (scale_to[1] - scale_to[0]) + scale_to[0]
    return normed

## convert MJD to delta t
def return_dt(data, n_feat=3):
    """Return delta times from a sequence of observation times. 
    Time axis must be first position of last dimension

    Parameters
    ----------
    data    : numpy array
        Light curves to be processed
    n_feats : list, optional
        Number of features

    Returns
    -------
    data
        delta times
    """
    data[:,:,0] = [x-z for x, z in zip(data[:,:,0],
                                       np.min(data[:,:,0], axis=1))]
    return data

def load_model_list(ID='zg3r4orb', device='cuda:0'):
    """Load a Python VAE model from file stored in a W&B archive

    Parameters
    ----------
    ID     : str
        W&B ID of the model to be loaded
    device : str, optional
        device where the model is loaded, cpu or gpu

    Returns
    -------
    vae
        VAE model, Python module
    conf
        Dictionary with model hyperparameters and configuration values
    """
    print('#'*50)
    print('LOADING cVAE: ')
    fname = glob.glob('%s/wandb/run-*-%s/VAE_model_*.pt' % (path, ID))[0]
    print(fname)
    config_f = glob.glob('%s/wandb/run-*-%s/config.yaml' % (path, ID))[0]
    with open(config_f, 'r') as f:
        conf = yaml.safe_load(f)
    conf = {k: v['value'] for k,v in conf.items() if 'wandb' not in k}
    conf['normed'] = True
    conf['folded'] = True
    aux = re.findall('\/run-(\d+\_\d+?)-\S+\/', config_f)
    conf['date']   = aux[0] if len(aux) != 0 else ''
    conf['ID'] = ID
    
    
    if conf['architecture'] == 'tcn':
        vae = VAE_TCN(latent_dim  = conf['latent_dim'],
                      seq_len     = conf['sequence_lenght'], 
                      kernel_size = conf['kernel_size'], 
                      hidden_dim  = conf['hidden_size'], 
                      nlevels     = conf['num_layers'], 
                      n_feats     = conf['n_feats'], 
                      dropout     = conf['dropout'], 
                      return_norm = conf['normed'], 
                      latent_mode = conf['latent_mode'], 
                      lab_dim     = conf['label_dim'], 
                      phy_dim     = conf['physics_dim'],
                      feed_pp     = True if conf['feed_pp'] == 'T' else False)
    elif conf['architecture'] in['lstm', 'gru']:
        vae = VAE_RNN(latent_dim  = conf['latent_dim'], 
                      seq_len     = conf['sequence_lenght'], 
                      hidden_dim  = conf['hidden_size'], 
                      n_layers    = conf['num_layers'],
                      rnn         = conf['architecture'], 
                      n_feats     = conf['n_feats'], 
                      dropout     = conf['dropout'], 
                      return_norm = conf['normed'], 
                      latent_mode = conf['latent_mode'],
                      lab_dim     = conf['label_dim'], 
                      phy_dim     = conf['physics_dim'])
    state_dict = torch.load(fname, map_location=device)
    if list(state_dict.keys())[0].split('.')[0] == 'module':
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    vae.load_state_dict(new_state_dict)
    vae.eval()
    vae.to(device)
    #print('Is model in cuda? ', next(vae.parameters()).is_cuda)
    
    return vae, conf

def evaluate_encoder(model, dataloader, params, force=False, device='cuda:0'):
    """Creates a joint plot of features, used during training, figures
    are W&B ready

    Parameters
    ----------
    model      : pytorch obejct
        model to be evaluated
    dataloader : pytorch object
        dataloader object with data to be evaluated with model
    params     : dictionary
        dictionary of model configuration parameters
    force      : bool, optional
        wheather to force model evaluation or load values from file archive
    device     : str, optional
        device where model runs, gpu or cpu

    Returns
    -------
    mu_df
        Pandas dataframe of mu values, last column are the labels 
    std_df
        Pandas dataframe of std values, last column are the labels 
    """
    
    fname_mu = '%s/wandb/run-%s-%s/latent_space_mu.txt' % (path, 
params['date'], params['ID'])
    fname_std = '%s/wandb/run-%s-%s/latent_space_std.txt' % (path, params['date'], params['ID'])
    fname_lbs = '%s/wandb/run-%s-%s/labels.txt' % (path, params['date'], params['ID'])
    if os.path.exists(fname_mu) & os.path.exists(fname_std) & os.path.exists(fname_lbs) &  ~force:
        print('Loading from files...')
        mu = np.loadtxt(fname_mu)
        std = np.loadtxt(fname_std)
        labels = np.loadtxt(fname_lbs, dtype=np.str)
    else:
        print('Evaluating Encoder...')
        time_start = datetime.datetime.now()
        
        mu, logvar, xhat, labels = [], [], [], []
        for i, (data, label, onehot, pp) in enumerate(dataloader):
            data = data.to(device)
            onehot = onehot.to(device)
            pp = pp.to(device)
            #cc = torch.cat([onehot, pp], dim=1)
            if params['label_dim'] > 0 and params['physics_dim'] > 0:
                mu_, logvar_ = model.encoder(data, label=onehot, phy=pp)
            elif params['label_dim'] > 0 and params['physics_dim'] == 0:
                mu_, logvar_ = model.encoder(data, label=onehot)
            elif params['label_dim'] == 0:
                mu_, logvar_ = model.encoder(data)
            else:
                print('Check conditional dimension...')
            
            mu.extend(mu_.data.cpu().numpy())
            logvar.extend(logvar_.data.cpu().numpy())
            labels.extend(label)
            torch.cuda.empty_cache()
        mu = np.array(mu)
        std = np.exp(0.5 * np.array(logvar))
        elap_time = datetime.datetime.now() - time_start
        print('Elapsed time  : %.2f s' % (elap_time.seconds))
        print('##'*20)
        
    mu_df = pd.DataFrame(mu)
    std_df = pd.DataFrame(std)
        
    mu_df['class'] = labels
    std_df['class'] = labels

    return mu_df, std_df

def flat_and_quantile(tensor, q):
    """
    Manually compute the quantile of a tensor.
    
    Parameters:
    - tensor: A PyTorch tensor.
    - q: Quantile to compute, a float in [0, 1].
    
    Returns:
    - The q-th quantile of the tensor.
    """
    flat_tensor = tensor.view(-1)
    sorted_tensor, _ = torch.sort(flat_tensor)
    index = int(q * (sorted_tensor.numel() - 1))
    return sorted_tensor[index]

def quantile(tensor, q):
    sorted_tensor, _ = torch.sort(tensor)
    length = sorted_tensor.shape[0]
    idx = float(length) * q
    lower_idx = int(torch.floor(torch.tensor(idx)).item())
    upper_idx = int(torch.ceil(torch.tensor(idx)).item())
    
    lower_val = sorted_tensor[lower_idx] if lower_idx < length else sorted_tensor[-1]
    upper_val = sorted_tensor[upper_idx] if upper_idx < length else sorted_tensor[-1]

    return lower_val + (upper_val - lower_val) * (idx - lower_idx)

def revert_light_curve(period, folded_normed_light_curve, 
                        original_sequences, faintness=1.0, 
                        classes = None, noise=True):
    """
    Revert previously folded and normed light curves back to the original light curves.

    Parameters:
        period (float): The period of the variability in the light curves.
        folded_normed_light_curve (numpy.ndarray): A 3D array representing the folded and normed light curves.
        faintness (float, optional): A scaling factor to adjust the magnitude values of the reverted light curves.
                                     Defaults to 1.0, meaning no scaling.

    Returns:
        list: A list of 2D arrays representing the reverted real light curves with time and magnitude values.
    """
    num_sequences = folded_normed_light_curve.shape[0]
    reverted_light_curves = []
    magnitude_min_max = get_time_sequence(n=1, star_class=classes)

    for i in range(num_sequences):
        # Extract the time (period) and magnitude values from the folded and normed light curve
        time = original_sequences[i] #folded_normed_light_curve[i,:,0]
        if np.max(folded_normed_light_curve[i,:,0])<0.95: 
            continue

        normed_magnitudes_min = np.quantile(folded_normed_light_curve[i,:,1], 0.01) 
        normed_magnitudes_max = np.quantile(folded_normed_light_curve[i,:,1], 0.99)
        
        normed_magnitudes = ((folded_normed_light_curve[i,:,1]-normed_magnitudes_min)/
                            (normed_magnitudes_max-normed_magnitudes_min))

        # Generate the time values for the reverted light curve
        [original_min, original_max, error_mean, error_std] = magnitude_min_max[i]

        #real_time =  time #get_time_from_period(period[i], time, example_sequence, sequence_length=600)

        real_time =  get_time_from_period(period[i],  folded_normed_light_curve[i,:,0], 
                                            time, sequence_length=600)

        # Revert the normed magnitudes back to the original magnitudes using min-max scaling and faintness factor
        if noise:
            original_magnitudes = ((normed_magnitudes * (original_max - original_min)) + original_min) * faintness
            random_noise = np.random.normal(error_mean, error_std, original_magnitudes.shape)
            original_magnitudes += random_noise

        else: 
            original_magnitudes = ((normed_magnitudes * (original_max - original_min)) + original_min) * faintness

        if isinstance(real_time, torch.Tensor):
            if real_time.is_cuda:
                real_time = real_time.cpu().numpy()
            else:
                real_time = real_time.numpy()

        # Convert original_magnitudes to NumPy array if it's a PyTorch tensor
        if isinstance(original_magnitudes, torch.Tensor):
            if original_magnitudes.is_cuda:
                original_magnitudes = original_magnitudes.cpu().numpy()
            else:
                original_magnitudes = original_magnitudes.numpy()

        # Now, you can use np.column_stack without issues
        reverted_light_curve = np.column_stack((original_magnitudes, real_time))

        reverted_light_curves.append(reverted_light_curve)
    
    reverted_light_curves = np.stack(reverted_light_curves)

    reverted_light_curves = np.swapaxes(reverted_light_curves, 1, 2)
    # Generate random unique indices along the last dimension
    random_indices = np.random.choice(350, nn_config['data']['seq_length']+1, replace=False)

    # Sort the indices for easier interpretation and debugging (optional)
    random_indices = random_indices.sort()

    # Select 200 random observations
    reverted_light_curves_random = reverted_light_curves[:, :, random_indices]

    reverted_light_curves_random = reverted_light_curves_random.squeeze(2) 

    for i in range(reverted_light_curves_random.shape[0]):
        sort_indices = np.argsort(reverted_light_curves_random[i, 1, :])
        for j in range(reverted_light_curves_random.shape[1]):
            reverted_light_curves_random[i, j, :] = reverted_light_curves_random[i, j, sort_indices]

    for i in range(reverted_light_curves_random.shape[0]):
        reverted_light_curves_random[i] = np.flipud(reverted_light_curves_random[i])

    print('Shape of reverted_light_curves_random[i, :, :]:', reverted_light_curves_random[i, :, :].shape)
    print('Shape of sort_indices:', sort_indices.shape)

    return reverted_light_curves_random

def apply_sensitivity(array, column, a_percentage=20):
    """
    Apply sensitivity analysis on a specified column of a NumPy array.

    Parameters:
        array (numpy.ndarray): The input 2D NumPy array.
        column (int): The column on which sensitivity is applied.
        a_percentage (float): Percentage for linear increment, range will be [-a%, a%].

    Returns:
        numpy.ndarray: A new array after applying sensitivity on the specified column.
    """
    assert len(array.shape) == 2, "Array should be 2-dimensional."
    num_rows = array.shape[0]
    increment_values = np.linspace(-a_percentage/100, a_percentage/100, num_rows)
    # Adjust values in the specified column based on the linear increment
    array[:, column] += array[0, column] * increment_values
    print('array pp in sensitive: ', array)
    return array

def get_time_sequence(n=1, star_class=['RRLYR']):
    """
    Retrieve time sequences from light curves data for 'n' objects.
    Parameters:
        n (int): Number of objects to sample.
    Returns:
        list: A list of lists containing time sequences from the light curves of 'n' objects.
    """
    n = int(n)
    
    path_train = PATH_FEATURES_TRAIN
    lc_train = pd.read_table(path_train, sep=',')


    lc_train[['SURVEY', 'FIELD', 'CLASS', 'NUMBER']] = lc_train['ID'].str.split('-', expand=True)
    lc_train['NUMBER'] = lc_train['NUMBER'].str.replace('.dat', '')

    time_sequences = []

    for star in tqdm(star_class, desc='Getting time sequences'):
        counter = 0
        fail = True

        while(fail):
            infinity_sanity_test = True
            counter = counter + 1

            try: 
                if counter < 50:
                    quantile_series = lc_train[lc_train.CLASS == star].Amplitude.quantile(0.1)
                    quantile_value = float(quantile_series)
                    new_label = (lc_train[(lc_train.CLASS==star) 
                                                                & (lc_train.Amplitude > quantile_value)]
                                                                .sample(1, replace=True)['ID']
                                                                .to_list()[0]
                                )
                    #print(new_label)
                else:
                    quantile_series = lc_train.Amplitude.quantile(0.25)
                    quantile_value = float(quantile_series)
                    new_label = (lc_train[(lc_train.Amplitude > quantile_value)]
                                                                .sample(1, replace=True)['ID']
                                                                .to_list()[0]
                                )
                path_lc = (PATH_LIGHT_CURVES_OGLE + new_label.split('-')[1].lower() +
                   '/' + new_label.split('-')[2].lower() + '/phot/I/' + new_label)
                

                lcu = pd.read_table(path_lc, sep=" ", names=['time', 'magnitude', 'error'])

                if lcu[['time', 'magnitude', 'error']].apply(np.isinf).any().any():
                    infinity_sanity_test = False
                    #print(lcu)

                #print(len(lcu['time'].to_list()), infinity_sanity_test, (lcu['time'].is_monotonic_increasing))
                if (infinity_sanity_test) and (len(lcu['time'].to_list())>nn_config['data']['minimum_lenght_real_curves']) and (lcu['time'].is_monotonic_increasing):
                    time_sequences.append([lcu.magnitude.min(), lcu.magnitude.max(), lcu.error.mean(), lcu.error.std()])
                    #print([lcu.magnitude.min(), lcu.magnitude.max(), lcu.error.mean(), lcu.error.std()])
                    fail = False

            except Exception as error :
                #print(error) 
                fail = True    
                logging.error(f"[get_time_sequence] The light curve {new_label} was not added: {error}")

    return time_sequences

def get_only_time_sequence(n=1, star_class=['RRLYR'], period=[1.0]):
    """
    Retrieve time sequences from light curves data for 'n' objects.
    Parameters:
        n (int): Number of objects to sample.
    Returns:
        list: A list of lists containing time sequences from the light curves of 'n' objects.
    """
    
    n = int(n)
    df_id_period = load_id_period_to_sample(star_class, period=period)
    
    df_id_period[['SURVEY', 'FIELD', 'CLASS', 'NUMBER']] = df_id_period['OGLE_id'].str.split('-', expand=True)
    time_sequences = []
    original_sequences = []

    star_counter = 0
    for star in tqdm(star_class, desc='Selecting light curves'):
        if period[star_counter] < MIN_PERIOD_VALUE: 
            period[star_counter] = MIN_PERIOD_VALUE
        
        closest_idx = (df_id_period['Period'] - period[star_counter]).abs().idxmin()

        new_label = df_id_period.loc[closest_idx]['OGLE_id']

        path_lc = (PATH_LIGHT_CURVES_OGLE + new_label.split('-')[1].lower() +
            '/' + new_label.split('-')[2].lower() + '/phot/I/' + new_label + '.dat')

        lcu = pd.read_table(path_lc, sep=" ", names=['time', 'magnitude', 'error'])
        lcu = lcu.reset_index()

        if 'level_0' in lcu.columns: 
            lcu = lcu.dropna(axis=1)
            lcu.columns = ['time', 'magnitude', 'error']

        lcu = lcu.dropna(axis=0) 
        period_i = df_id_period[df_id_period.OGLE_id==new_label].Period.values[0]
        times = lcu['time'].to_list()
        lc_adapted = ensure_n_elements(times)
        lc_adapted_to_real_sequence = ensure_n_elements(times, n=350)
        lc_phased = ((lc_adapted-np.min(lc_adapted))/period_i)%1
        sorted_lc_phased = np.sort(lc_phased)
        time_sequences.append(sorted_lc_phased)
        original_sequences.append(lc_adapted_to_real_sequence)
        star_counter = star_counter + 1
    return time_sequences, original_sequences

def ensure_n_elements(lst, n=600):
    """
    Ensure that a list has exactly 'n' elements.

    Parameters:
        lst (list): The input list.
        n (int): The desired number of elements.

    Returns:
        array: The modified array with 'n' elements.
    """
    if len(lst) < n:
        differences = [j - i for i, j in zip(lst[:-1], lst[1:])]
        mean_difference = sum(differences) / len(differences)
        while len(lst) < n:
            lst.append(lst[-1]+mean_difference)
    elif len(lst) > n:
        lst = random.sample(lst, n)
        lst.sort()

    lst = np.array(lst, dtype=float)
    return lst

def get_time_from_period(period, phased_time,  example_sequence, sequence_length=600, verbose =False): 
    """
    Generate a time sequence based on the provided period, phased_time (folded times),
    sequence length, and example sequence.

    The function first calculates a range for the 'k' values, based on the minimum and maximum values
    of the example sequence divided by the period. It then generates an array of random 'k' values
    within this range using NumPy's np.random.uniform().

    The time sequence is then calculated by adding the 'phased_time' to the product of '(period + phased_time)'
    and 'k_values', and then adding the minimum value of the example sequence.

    Parameters:
        period (float): The period of the variability in the light curve.
        phased_time (numpy.ndarray): A 1D array representing the folded times (estimated as real_time % period).
        sequence_length (int): The desired length of the time sequence.
        example_sequence (numpy.ndarray): A 1D array representing an example sequence.

    Returns:
        numpy.ndarray: A 1D array representing the generated time sequence with reverted time values.
    """
    # Calculate the range for 'k' values based on the minimum and maximum values of example_sequence
    if len(example_sequence) > 0:
        k_min = int(np.quantile(example_sequence, 0.001) / period)
        k_max = int(np.quantile(example_sequence, 0.999) / period)
        if k_min>k_max: 
            k_min = 1000
            k_max = 4000
        elif k_min < 0: 
            k_min = 1000
            k_max = 4000
    else:
        raise ValueError("example_sequence is empty, cannot perform min/max operations.")

    if isinstance(k_min, np.ndarray):
        k_min = torch.tensor(k_min).to(period.device)

    if isinstance(k_max, np.ndarray):
        k_max = torch.tensor(k_max).to(period.device)

    # Generate an array of random 'k' values within the specified range
    if verbose:
        print('k_min, k_max: ', k_min, k_max)
        
    k_values = torch.randint(low=k_min, high=k_max + 1, size=(sequence_length,)).to(period.device)
    k_values = k_values.float()  # Convert to float32

    if verbose:
        print('k_values: ', k_values)

    # Calculate the minimum value of the example sequence
    min_sequence = np.min(example_sequence)

    if verbose: 
        print('min_sequence: ', min_sequence)

    # Ensure phased_time is a PyTorch tensor and on the same device as 'period'
    if isinstance(phased_time, np.ndarray):
        phased_time = torch.tensor(phased_time).to(period.device)

    # Ensure min_sequence is a PyTorch tensor and on the same device as 'period'
    if isinstance(min_sequence, np.ndarray):
        min_sequence = torch.tensor(min_sequence).to(period.device)

    # Calculate the time sequence using vectorized operations
    period = period.float()  # Convert tensor1 to float32
    phased_time = phased_time.float()  # Convert tensor2 to float32

    print(period)
    if 'LOG' in data_sufix: 
        period = torch.exp(period)

    print(period)
    time_sequence = min_sequence + (period)*(k_values + phased_time)

    # Check if any values in time_sequence are infinity
    if np.isinf(time_sequence.cpu().numpy()).any():
        print("Infinity values detected in time_sequence tensor:")
        print(time_sequence)
    #else:
        #print("No infinity values detected in time_sequence.")

    if verbose: 
        print('time_sequence: ', time_sequence)
    
    return time_sequence

def save_arrays_to_folder(array1, array2, folder_path):
    """
    Save two NumPy arrays to a specified folder.

    Parameters:
        array1 (numpy.ndarray): The first NumPy array to save.
        array2 (numpy.ndarray): The second NumPy array to save.
        folder_path (str): The path to the folder where the arrays will be saved.

    Raises:
        ValueError: If the folder_path is not a valid directory.

    Example:
        array1 = np.array([1, 2, 3])
        array2 = np.array([4, 5, 6])
        save_arrays_to_folder(array1, array2, '/path/to/folder/')
    """
    if not os.path.exists(folder_path):
        raise ValueError("The folder_path is not a valid directory.")


    if isinstance(array1, torch.Tensor):
        if array1.is_cuda:
           array1 = array1.cpu()
    if isinstance(array2, torch.Tensor):
        if array2.is_cuda:
           array2 = array2.cpu()  

    # Create filenames for the two arrays
    array1_filename = os.path.join(folder_path, "x_batch_pelsvae.npy")
    array2_filename = os.path.join(folder_path, "y_batch_pelsvae.npy")

    # Save the arrays
    np.save(array1_filename, array1)
    np.save(array2_filename, array2)

def load_pp_list(vae_model: str) -> List[str]:
    with open(f'wandb/run--{vae_model}/config.yaml', 'r') as file:
        config_vae: Dict[str, Any] = yaml.safe_load(file)
    
    phy_params = config_vae['phys_params']['value'].lower()  # Convert to lowercase

    # Mapping from character to its corresponding physical parameter
    pp_mapping = {
        'p': 'Period',
        't': 'teff_val',
        'm': '[Fe/H]_J95',
        'c': 'bp_rp',
        'a': 'abs_Gmag',
        'r': 'radius_val',
        'l': 'lum_val',
        'g': 'logg'
    }

    # Generate the PP_list
    PP_list = [value for key, value in pp_mapping.items() if key in phy_params]
    #PP_list = ['Period','teff_val', 'abs_Gmag', 'radius_val', '[Fe/H]_J95', 'logg']

    print('FEATURES: ', PP_list)

    return PP_list

def load_metadata():
    print('#'*50)
    print('SETUP')
    print('#'*50)

    with open('src/configuration/paths.yaml', 'r') as file:
        YAML_FILE: Dict[str, Any] = yaml.safe_load(file)

    PATHS: Dict[str, str] = YAML_FILE['paths']
    PATH_PRIOS: str = PATHS['PATH_PRIOS']
    mean_prior_dict: Dict[str, Any] = load_yaml_priors(PATH_PRIOS)

    with open('src/configuration/regressor.yaml', 'r') as file:
        config_file: Dict[str, Any] = yaml.safe_load(file)
    return mean_prior_dict, config_file

'''
def pp_sensitive_test(PP_list):
    print('conducting sensitive test')
    for pp in PP_list:
        objects_by_class = {'ACEP':24}#, 'CEP': 24,  'DSCT': 24,  'ECL':24,  'ELL': 24,  'LPV': 24,  'RRLYR':  24,  'T2CEP':24}
        for key, value in objects_by_class.items():
            temp_dict = {key: value}
            creator.get_synthetic_light_curves(None, None, training_cnn=False, plot=False, save_plot=True,
                                            objects_by_class=temp_dict, sensivity=pp,
                                            column_to_sensivity=pp)
            del temp_dict
'''

def evaluate_and_plot_cm_from_dataloader(model, dataloader, label_encoder, title, wandb_active: bool = True):
    all_y_data = []
    all_predicted = []
    
    model.eval()  # Switch to evaluation mode

    with torch.no_grad():  # No need to calculate gradients in evaluation
        for batch in dataloader:
            x_data, y_data = batch
            x_data, y_data = x_data.float(), y_data.float()
            outputs = model(x_data)
            _, predicted = torch.max(outputs.data, 1)
            
            # Check if y_data is one-hot encoded and convert to label encoding if it is
            if len(y_data.shape) > 1 and y_data.shape[1] > 1:
                y_data = torch.argmax(y_data, dim=1)
            
            # Move data to CPU and append to lists
            all_y_data.extend(y_data.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
    
    # Compute confusion matrix
    try:
        cm = confusion_matrix(all_y_data, all_predicted, normalize='true')
    except ValueError as e:
        print(f"An error occurred: {e}")
        print(f"y_data shape: {len(all_y_data)}, predicted shape: {len(all_predicted)}")
        return None
    
    print('wandb_active: ', wandb_active)
    plot_cm(cm, label_encoder, title=title, wandb_active = wandb_active)
    #export_recall_latex(all_y_data, all_predicted, label_encoder)
    
    return cm

def remove_duplicates(my_list):
    # Using an ordered collection to preserve the order of elements
    no_duplicates = []
    for element in my_list:
        if element not in no_duplicates:
            no_duplicates.append(element)
    return no_duplicates
