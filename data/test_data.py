import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

print(df1y.shape)
print(np.max(df2y))
print(df2y.shape)
# Assuming 6 classes in y, for each class plot the 2D density
# For each class, plot the 2D density
for i in range(np.max(df2y) + 1):  # Ensure all classes are covered
    
    plt.figure(figsize=(12, 8))
    
    # Dataset 1
    mask1 = df1y[:, i] == 1
    delta_time1 = df1[mask1][:, 0, :].ravel()
    delta_magnitude1 = df1[mask1][:, 1, :].ravel()
    
    # Dataset 2
    mask2 = df2y == i
    delta_time2 = df2[mask2][:, 0, :].ravel()
    delta_magnitude2 = df2[mask2][:, 1, :].ravel()

    # Class name
    class_name = label_encoder.inverse_transform([i])[0]
    

    # Plot 2D scatter for Dataset 2
    sns.scatterplot(x=delta_time2, y=delta_magnitude2, color="b", label="Real light curves", alpha=0.1)

        # Plot 2D scatter for Dataset 1
    sns.scatterplot(x=delta_time1, y=delta_magnitude1, color="r", label="Synthetic Light curves", alpha=0.2)
    
    
    plt.title(f"2D Density for Class {class_name}")
    plt.legend()
    plt.savefig(f"2D_Density_Class_{class_name}.png")
    plt.close()

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