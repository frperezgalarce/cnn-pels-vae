import numpy as np
import torch

class LightCurveRandomSampler:
    def __init__(self, lc_reverted, labels, new_length, m):
        """
        Initialize the LightCurveRandomSampler.
        
        Parameters:
        - lc_reverted: NumPy array of shape (k, 2, n) where k is the number of light curves, and n is the number of observations.
        - labels: One-hot encoded labels (PyTorch tensor) for the light curves, of shape (k, label_dim)
        - new_length: Length of the randomly sampled segment from each light curve.
        - m: Number of samples to be extracted from each light curve.
        """
        self.lc_reverted = lc_reverted
        self.labels = labels
        self.new_length = new_length
        self.m = m
        self.k, _, self.n = lc_reverted.shape

    def sample(self):
        """Sample m segments of length new_length from each light curve in lc_reverted and replicate corresponding labels."""
        all_samples = []
        all_labels = []

        for i, curve in enumerate(self.lc_reverted):
            for _ in range(self.m):
                # Randomly select a start index
                #start_index = np.random.randint(0, self.n - self.new_length + 1)
                random_indexes = np.sort(np.random.choice(self.n, self.new_length, replace=False))
                # Extract the sequence starting from the start index
                sample = curve[:, random_indexes]
                
                all_samples.append(sample[np.newaxis, :])  # Add an extra dimension to concatenate later
                all_labels.append(self.labels[i][np.newaxis, :])

        # Concatenate all samples and labels along the first dimension
        return np.concatenate(all_samples, axis=0), np.concatenate(all_labels, axis=0)