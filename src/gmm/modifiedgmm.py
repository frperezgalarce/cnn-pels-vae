import pickle
from scipy.stats import multivariate_normal
from scipy.integrate import nquad
import numpy as np

class ModifiedGaussianSampler:
    def __init__(self, b=0.5, components=3):
        self.model = None
        self.b = b
        self.Z = None
        self.q = None
        self.components = components

    def load_p(self, filename):
        with open(filename, 'rb') as file:
            params = pickle.load(file)
            self.model = params['model']

    def p(self,x):
        return np.exp(self.model.score_samples(x.reshape(1, -1)))

    def metropolis_hasting(self, n_samples=5, iterations=1000, burn_in=500):
        # Metropolis-Hastings to sample from q(x)
        samples = []
        current_x = np.random.randn(self.components)  # Start from a random point
        for i in range(iterations):
            proposal_x = current_x + np.random.normal(0, 0.5, size=self.components) # Random walk proposal
            acceptance_ratio = (self.p(proposal_x)**self.b) / (self.p(current_x)**self.b)
            if np.random.rand() < acceptance_ratio:
                current_x = proposal_x
            if i > burn_in and (i-burn_in) % ((iterations-burn_in)//n_samples) == 0:
                samples.append(current_x)

        samples = np.array(samples)

        return samples

    def modify_and_sample(self, path): 
        self.load_p(path)
        samples = self.metropolis_hasting(n_samples=5, iterations=1000, burn_in=500)
        return samples
    