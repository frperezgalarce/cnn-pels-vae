import torch
import gzip 
import numpy as np
import pandas as pd

def test_gpus():
    print(torch.cuda.is_available())
    print(torch.__version__)

def load_data_to_train(): 
    data_path = ('data/time_series/real/OGLE3_lcs_I_meta_snr5_augmented_folded_trim600.npy.gz')
    print('Loading from:\n', data_path)
    with gzip.open(data_path, 'rb') as f:
        np_data = np.load(f, allow_pickle=True)
    print(type(np_data))
    print(print("Keys:", np_data.dtype.names))
    print('meta')
    print(np_data.item()['meta'])
    print('lcs')
    print(np_data.item()['lcs'])

def load_new_validated_pp():
    pp_path = 'data/inter/Validated_OGLExGaiaDR3.csv'
    df = pd.read_csv(pp_path)
    print(df.head(50))

test_gpus()
