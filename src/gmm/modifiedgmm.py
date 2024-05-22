import numpy as np
from typing import Any, List
import pickle
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.set_printoptions(precision=4)

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
            print(self.model.weights_)

    def p(self, x: np.ndarray) -> np.float64:
        return np.exp(self.model.score_samples(x.reshape(1, -1)))

    def randomly_select_component(self) -> int:
        """Randomly select a component based on weights and return its index."""
        weights = self.model.weights_
        selected_component_index = np.random.choice(np.arange(len(weights)), p=weights)
        return selected_component_index
    
    def autocorrelation(self, samples, lag=1):
        """Compute the autocorrelation of the samples."""
        n = len(samples)
        mean = np.mean(samples, axis=0)
        var = np.var(samples, axis=0)
        autocorr = np.sum((samples[:-lag] - mean) * (samples[lag:] - mean), axis=0) / (var * (n - lag))
        return autocorr

    def check_convergence(self, samples, max_lag=10):
        """Check the convergence of the algorithm by assessing the autocorrelation of the samples."""
        for lag in range(1, max_lag + 1):
            autocorr = self.autocorrelation(samples, lag)
            print(f"Autocorrelation at lag {lag}: {autocorr}")
            if np.all(np.abs(autocorr) < 0.05):
                print("Convergence likely achieved.")
                return True
        print("Convergence not yet achieved.")
        return False

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
        print('selected_component_index: ', selected_component_index)
        
        current_x = self.model.means_[selected_component_index] + np.random.normal(0, 0.01, size=len(self.features))
        print('current_x: ', current_x)
        
        
        sample_interval = (iterations - burn_in) // n_samples
        print('sample_interval: ', sample_interval) 
        
        for i in range(1, iterations+100):
            proposal_x = current_x + np.random.normal(0, 0.01, size=len(self.features))*current_x 
            
            acceptance_ratio = (self.p(proposal_x) ** self.b) / (self.p(current_x) ** self.b)
            r = np.random.rand()            
            if r < acceptance_ratio:
                current_x = proposal_x
                
            if i > burn_in and ((i - burn_in) % sample_interval) == 0:
                print('append: ',  current_x)
                samples.append(current_x.copy())
                
            if len(samples) == n_samples: 
                break

        print('#'*50)
        print('Samples: ')
        print(len(samples))
        self.check_convergence(np.array(samples))


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
            current_x = self.model.means_[selected_component_index] + np.random.normal(0, 0.1, size=len(self.features))
            sample_interval = (iterations_component - burn_in) // np.ceil(n_samples*weight)
        
        
            print(self.model)
            print(self.model.sample(n_samples=5))
            print(type(self.p()))
            print(type(self.p()), self.p())
            print(self.b, type(self.b))
            
            fail
            for i in range(1, iterations_component+100):
                proposal_x = current_x + np.random.normal(0, 0.075, size=len(self.features))*current_x 
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
        self.check_convergence(np.array(samples))

        return np.array(samples)

    def two_step_sample(self, n_samples: int = 5, first_sample: int = 1000) -> np.ndarray:
        # Sample from the GMM
        samples, _ = self.model.sample(n_samples = first_sample)
        print("Original Samples:")
        print(samples)

        # Calculate the probabilities of the samples
        log_probs = self.model.score_samples(samples)
        probs = np.exp(log_probs)
        print("\nOriginal Probabilities:")
        print(probs)

        # Modify the probabilities according to p^b
        modified_probs = probs ** self.b
        modified_probs /= np.sum(modified_probs)  # normalize to create a probability distribution
        print("\nModified Probabilities:")
        print(modified_probs)

        # Resample a subsample according to the new density
        n_resamples = n_samples  # number of samples to resample
        resample_indices = np.random.choice(range(len(samples)), size=n_resamples, p=modified_probs)
        resampled_samples = samples[resample_indices]
        print("\nResampled Samples:")
        print(resampled_samples)

        return np.array(resampled_samples)

    def plot_mixture_of_bivariate_gaussians(self, ax, i, j, PP):
        # Means and Covariances
        means = self.model.means_
        covariances = self.model.covariances_
        x, y = np.mgrid[means[:, i].min()-3:means[:, i].max()+3:.01,
                        means[:, j].min()-3:means[:, j].max()+3:.01]
        pos = np.dstack((x, y))
        pdf = np.zeros(pos.shape[:2])

        # Summing the probability densities of each Gaussian
        for k in range(means.shape[0]):
            mean_ij = means[k, [i, j]]
            covariance_ij = covariances[k, [i, j], :][:, [i, j]]
            rv = multivariate_normal(mean_ij, covariance_ij)
            pdf += rv.pdf(pos)

        ax.contourf(x, y, pdf, cmap='viridis')
        ax.set_xlabel(PP[i])
        ax.set_ylabel(PP[j])

    def plot_samples_densities(self, PP, new_data_points, star_class):
        fig, axs = plt.subplots(len(PP), len(PP), figsize=(15, 15))
        for i in range(len(PP)):
            for j in range(len(PP)):
                if i < j:
                    self.plot_mixture_of_bivariate_gaussians(axs[i, j], i, j, PP)
                    axs[i, j].scatter(new_data_points[:, i], new_data_points[:, j], color='red')
                else:
                    axs[i, j].axis('off')

        plt.tight_layout()
        plt.savefig(star_class + '_density.png')
        plt.show()
    
    def modify_and_sample(self, path: str, n_samples=5, mode='allcomponents') -> np.ndarray:
        np.set_printoptions(suppress=True)

        print(n_samples)
        self.load_p(path)
        print('Model loaded: ', self.model)
        print(self.model.means_)
        print('mode: ', mode)
        if mode == 'onecomponent':
            samples = self.metropolis_hasting(n_samples=n_samples)
        elif mode == 'allcomponents': 
            print('Sampling from all components')
            samples = self.metropolis_hasting_all_components(n_samples=n_samples)
        elif mode == 'two_steps': 
            samples = self.two_step_sample(n_samples=n_samples)
        else: 
            raise('The mode ' + mode + 'is not implemented.')
        return samples