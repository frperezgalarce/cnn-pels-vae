import pickle
import numpy as np
from typing import Any, List
import yaml


np.set_printoptions(precision=4)

with open('src/configuration/nn_config.yaml', 'r') as file:
    nn_config = yaml.safe_load(file)

ITERATIONS_MH: int = nn_config['sampling']['iterations']
BURNIN_MH: int = nn_config['sampling']['burnin']

class ModifiedGaussianSampler:
    def __init__(self, b: float = 0.5, components: int = 3, features: int = 3) -> None:
        self.model: Any = None
        self.b: float = b
        self.Z: Any = None
        self.q: Any = None
        self.components: int = components
        self.features: int = features

    def load_p(self, filename: str) -> None:
        print('loading model: ', filename)
        with open(filename, 'rb') as file:
            params = pickle.load(file)
            self.model = params['model']
            print(self.model)

    def p(self, x: np.ndarray) -> np.float64:
        return np.exp(self.model.score_samples(x.reshape(1, -1)))

    def randomly_select_component(self) -> int:
        """Randomly select a component based on weights and return its index."""
        weights = self.model.weights_
        selected_component_index = np.random.choice(np.arange(len(weights)), p=weights)
        return selected_component_index
    

    def metropolis_hasting(self, n_samples: int = 5) -> np.ndarray:
        """
        Metropolis-Hastings algorithm for sampling from a distribution q(x).
    
        Parameters:
        n_samples (int): The number of samples to return.

        Returns:
        np.ndarray: An array of samples from the distribution.
        """
        iterations: int = ITERATIONS_MH
        burn_in: int = BURNIN_MH
        samples: List[np.ndarray] = []

        selected_component_index = self.randomly_select_component()
        selected_mean = self.model.means_[selected_component_index] + np.random.normal(0, 0.01, size=len(self.features))
        current_x = selected_mean

        sample_interval = (iterations - burn_in) // n_samples

        for i in range(1, iterations+100):
            proposal_x = current_x + np.random.normal(0, 0.001, size=len(self.features))*current_x 
            acceptance_ratio = (self.p(proposal_x) ** self.b) / (self.p(current_x) ** self.b)           
            if np.random.rand() < acceptance_ratio:
                current_x = proposal_x
            if i > burn_in and ((i - burn_in) % sample_interval) == 0:
                samples.append(current_x.copy())
            if len(samples) == n_samples: 
                break


        print('#'*50)
        print('Samples: ')
        print(len(samples))

        return np.array(samples)

    def metropolis_hasting_all_components(self, n_samples: int = 5) -> np.ndarray:
        """
        Metropolis-Hastings algorithm for sampling from a distribution q(x).
    
        Parameters:
        n_samples (int): The number of samples to return.

        Returns:
        np.ndarray: An array of samples from the distribution.
        """
        iterations: int = ITERATIONS_MH
        burn_in: int = BURNIN_MH 
        samples: List[np.ndarray] = []
        weights = self.model.weights_

        for selected_component_index in range(len(weights)):
            weight = self.model.weights_[selected_component_index]
            iterations_component = int(iterations*weight)
            selected_mean = self.model.means_[selected_component_index] + np.random.normal(0, 0.01, size=len(self.features))
            current_x = selected_mean
            sample_interval = (iterations_component - burn_in) // np.ceil(n_samples*weight)
        
            for i in range(1, iterations_component+100):
                proposal_x = current_x + np.random.normal(0, 0.001, size=len(self.features))*current_x 
                acceptance_ratio = (self.p(proposal_x) ** self.b) / (self.p(current_x) ** self.b)           
                if np.random.rand() < acceptance_ratio:
                    current_x = proposal_x
                if i > burn_in and (i - burn_in) % sample_interval == 0:
                    samples.append(current_x.copy())
                if len(samples) == n_samples: 
                    break
        print('#'*50)
        print('Samples: ')
        print(len(samples))
        if len(samples)<n_samples: 
            replicate_n = n_samples-len(samples)
            for j in range(replicate_n): 
                random_index = np.random.randint(0,len(samples))
                samples.append(samples[random_index])
        return np.array(samples)


    def modify_and_sample(self, path: str, n_samples=5, mode='allcomponents') -> np.ndarray:
        np.set_printoptions(suppress=True)
        self.load_p(path)
        if mode == 'onecomponent':
            samples = self.metropolis_hasting(n_samples=n_samples)
        elif mode == 'allcomponents': 
            print('Sampling from all components')
            samples = self.metropolis_hasting_all_components(n_samples=n_samples)
        else: 
            raise('the mmmmm')
        return samples

