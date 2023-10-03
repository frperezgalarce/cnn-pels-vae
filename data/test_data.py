import numpy as np
import matplotlib.pyplot as plt

df1 = np.load('x_batch_pelsvae.npy', allow_pickle=True)
df2 = np.load('train_np_array.npy', allow_pickle=True)

print(df1.shape)
print(df2.shape)

if int(df2.shape[1])!=2:
    df2 = df2.transpose(0, 2, 1)

print(df1.shape)
print(df2.shape)

# Create filenames for the two arrays


plt.figure()
plt.scatter(df2[0][1,:], df2[0][0,:], color='red', alpha=0.2)
plt.scatter(df1[0][1,:], df1[0][0,:], alpha=0.2)
plt.show()