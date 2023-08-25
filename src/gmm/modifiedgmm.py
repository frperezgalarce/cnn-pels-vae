import pickle
import numpy as np
from typing import Any, List

class ModifiedGaussianSampler:
    def __init__(self, b: float = 0.5, components: int = 3, features: int = 3) -> None:
        self.model: Any = None
        self.b: float = b
        self.Z: Any = None
        self.q: Any = None
        self.components: int = components
        self.features: int = features

    def load_p(self, filename: str) -> None:
        with open(filename, 'rb') as file:
            params = pickle.load(file)
            self.model = params['model']
            print(self.model)

    def p(self, x: np.ndarray) -> np.float64:
        return np.exp(self.model.score_samples(x.reshape(1, -1)))

    def metropolis_hasting(self, n_samples: int = 5, iterations: int = 1000, burn_in: int = 500) -> np.ndarray:
        # Metropolis-Hastings to sample from q(x)
        samples: List[np.ndarray] = []
        current_x = np.random.randn(self.features)  # Start from a random point
        print(current_x)
        for i in range(iterations):
            proposal_x = current_x + np.random.normal(0, 0.5, size=self.features) # Random walk proposal
            acceptance_ratio = (self.p(proposal_x)**self.b) / (self.p(current_x)**self.b)
            if np.random.rand() < acceptance_ratio:
                current_x = proposal_x
            if i > burn_in and (i-burn_in) % ((iterations-burn_in)//n_samples) == 0:
                samples.append(current_x)

        samples = np.array(samples)
        return samples

    def modify_and_sample(self, path: str) -> np.ndarray:
        self.load_p(path)
        samples = self.metropolis_hasting(n_samples=5, iterations=1000, burn_in=500 )
        return samples
