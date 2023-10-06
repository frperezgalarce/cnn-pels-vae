import numpy as np
import matplotlib.pyplot as plt
import random 
from typing import Union, Tuple, Optional, Any, Dict, List
import yaml
import pickle 

def transform_to_consecutive(input_list, label_encoder):
    unique_elements = sorted(set(input_list))
    modified_labelencoder_classes = [element for idx, element in enumerate(label_encoder.classes_) if idx in unique_elements]
    mapping = {value: index for index, value in enumerate(unique_elements)}
    modified_labels = np.array([mapping[element] for element in input_list])
    return modified_labels, modified_labelencoder_classes


with open('../src/paths.yaml', 'r') as file:
    YAML_FILE: Dict[str, Any] = yaml.safe_load(file)

PATHS: Dict[str, str] = YAML_FILE['paths']
PATH_MODELS: str = PATHS['PATH_MODELS']

df1 = np.load('x_batch_pelsvae.npy', allow_pickle=True)
df2 = np.load('train_np_array.npy', allow_pickle=True)
df1y = np.load('y_batch_pelsvae.npy', allow_pickle=True)
df2y = np.load('train_np_array_y.npy', allow_pickle=True)

print(df1.shape)
print(df2.shape)

if int(df2.shape[1])!=2:
    df2 = df2.transpose(0, 2, 1)

print(df1.shape)
print(df2.shape)
with open(PATH_MODELS+'label_encoder_vae.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
# Create filenames for the two arrays

print(df1y.shape)
print(df1y)

print(df2y.shape)
df2y = label_encoder.transform(df2y)
df2y, _ = transform_to_consecutive(df2y, label_encoder)

print(np.max(df2y))


print(np.unique(df1y), np.unique(df2y))
for j in range(50):
    lc = random.randint(1, df1.shape[0]-1)
    lc2 = random.randint(1, df2.shape[0]-1)
    print(lc, lc2)

    while np.argmax(df1y[lc]) != df2y[lc2]: 
        lc = random.randint(1, df1.shape[0]-1)
        lc2 = random.randint(1, df2.shape[0]-1)
        print(lc, lc2)
        print(np.argmax(df1y[lc]), df2y[lc2])
    
    label1 = np.argmax(df1y[lc])
    label1 = label_encoder.inverse_transform([label1])
    label2 = label_encoder.inverse_transform([df2y[lc2]])

    plt.figure()
    plt.scatter(df1[lc][1,:], df1[lc][0,:], alpha=0.2, label = label1)
    plt.scatter(df2[lc2][1,:], df2[lc2][0,:], color='red', alpha=0.2, label = label2)
    plt.legend()
    plt.show()