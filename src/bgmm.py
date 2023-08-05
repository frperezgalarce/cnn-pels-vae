import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import pickle
import pandas as pd
import matplotlib.pyplot as plt 

PATH_PP = '/home/franciscoperez/Documents/GitHub/CNN-PELSVAE/src/PELSVAE/data/inter/definite_matches.csv'

mean_prior_dict = {'RRLYR':{'components':2,'mean_priors':[[1.0, 3.0, 4.9],[1.0, 3.0, 4.9]]}, 
                   'CEP':{'components':2,'mean_priors':[[1.0, 3.0, 4.9],[1.0, 3.0, 4.9]]}}
                #TODO: complete with expert knowledge this dictionary

class BayesianGaussianMixtureModel:
    def __init__(self, n_components=2, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.bgm = BayesianGaussianMixture(n_components=self.n_components, random_state=self.random_state)
        self.object = None
        self.mean_prior = None

    def train(self, X, mean_prior=None):
        self.mean_prior = mean_prior
        if mean_prior is not None:
            self.bgm.mean_prior_ = mean_prior
        self.bgm.fit(X)
        self.object = self.bgm  # Saving the trained model as the object attribute

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump({'n_components': self.n_components,
                         'random_state': self.random_state,
                         'mean_prior': self.mean_prior,
                         'model': self.bgm},
                        file)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as file:
            model_data = pickle.load(file)
        instance = cls(n_components=model_data['n_components'],
                       random_state=model_data['random_state'])
        instance.mean_prior = model_data['mean_prior']
        instance.bgm = model_data['model']
        instance.object = instance.bgm  # Save the loaded model as the object attribute
        return instance

    def plot_2d_bgmm(self, bgmm, X, starClass, feature1 = 'abs_Imag', feature2='teff_val'):
        # Plotting the data points
        plt.scatter(X[feature1], X[feature2], c='blue', alpha=0.5, label='Data Points')
        plt.xlabel(feature1)
        plt.ylabel(feature2)

        # Getting the data limits
        x_min, x_max = X[feature1].min(), X[feature1].max()
        y_min, y_max = X[feature2].min(), X[feature2].max()

        # Create a meshgrid to cover the data range
        x_mesh, y_mesh = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        # Create a 3D meshgrid using np.dstack
        xy = np.dstack((x_mesh, y_mesh, np.zeros_like(x_mesh)))

        # Calculate the log-probability of the points on the meshgrid
        z = -bgmm.bgm.score_samples(xy.reshape(-1, 3))
        z = z.reshape(x_mesh.shape)

        # Plotting the contour plots of the Gaussian components
        plt.contourf(x_mesh, y_mesh, z, levels=np.logspace(0, 3, 10), cmap=plt.cm.Blues_r, alpha=0.8)

        plt.title('Gaussian Mixture Model Components - ' + str(starClass))
        #plt.legend()
        plt.colorbar(label='Log Probability')
        plt.show()

    def generate_samples(self, n_samples=1):
        samples, _ = self.bgm.sample(n_samples)
        return samples

def train_and_save(components=3):
    data = pd.read_csv(PATH_PP)
    df_selected_columns = data[['Type','teff_val','Period','abs_Imag']]
    classes = df_selected_columns.Type.unique()

    for star_class in classes:
        print(star_class)
        df_filtered_by_class = df_selected_columns[df_selected_columns.Type==star_class]
        X = df_filtered_by_class[['teff_val','Period','abs_Imag']]
        X = X.dropna()
        if X.shape[0] > 30:
            bgmm = BayesianGaussianMixtureModel(n_components=components, random_state=42)
            bgmm.train(X, mean_prior=None)
            bgmm.save_model('bgm_model_'+str(star_class)+'.pkl')
            bgmm.plot_2d_bgmm(bgmm, X, star_class, feature1 = 'teff_val', feature2='Period')

def get_load_and_sample(star_class='RRLYR'):
    # Load the model and generate samples
    train_and_save(components=2)
    loaded_bgmm = BayesianGaussianMixtureModel.load_model('bgm_model_'+str(star_class)+'.pkl')
    generated_samples = loaded_bgmm.generate_samples(n_samples=5)
    print(generated_samples)
