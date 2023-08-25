import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from typing import Optional, Dict, Union, Tuple, ClassVar
from src.utils import load_yaml_priors, extract_midpoints
import yaml
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

with open('src/paths.yaml', 'r') as file:
    YAML_FILE: str = yaml.safe_load(file)

PATHS: str = YAML_FILE['paths']
PATH_PRIOS: str = PATHS['PATH_PRIOS']
PATH_PP: str = PATHS['PATH_PP']
PATH_FIGURES: str = PATHS['PATH_FIGURES']

class BayesianGaussianMixtureModel:
    def __init__(self, n_components: int = 2, random_state: Optional[int] = None):
        self.n_components = n_components
        self.random_state = random_state
        self.bgm = BayesianGaussianMixture(n_components=self.n_components, random_state=self.random_state)
        self.object = None
        self.mean_prior = None

    def train(self, X: pd.DataFrame, mean_prior: Optional[np.array] = None) -> None:
        self.mean_prior = mean_prior
        if mean_prior is not None:
            self.bgm.mean_prior_ = mean_prior
        self.bgm.fit(X)
        self.object = self.bgm  # Saving the trained model as the object attribute

    def save_model(self, filename: str) -> None:
        with open(filename, 'wb') as file:
            pickle.dump({
                'n_components': self.n_components,
                'random_state': self.random_state,
                'mean_prior': self.mean_prior,
                'model': self.bgm
            }, file)

    @classmethod
    def load_model(cls, filename: str) -> "BayesianGaussianMixtureModel":
        with open(filename, 'rb') as file:
            model_data: Dict = pickle.load(file)
        instance = cls(n_components=model_data['n_components'], random_state=model_data['random_state'])
        instance.mean_prior = model_data['mean_prior']
        instance.bgm = model_data['model']
        instance.object = instance.bgm  # Save the loaded model as the object attribute
        return instance

    def plot_2d_bgmm(self, bgmm: "BayesianGaussianMixtureModel", 
                           X: pd.DataFrame, starClass: str, feature1: str = 'abs_Imag', 
                           feature2: str = 'teff_val', 
                           save=True, priors: bool =False) -> None:
        
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
        if save:
            fig_name = PATH_FIGURES+feature1+'_'+feature2+'_priors_'+str(priors)+'_PP_'+str(X.shape[1])
            plt.savefig(fig_name+'.png', dpi=300)
            plt.savefig(fig_name+'.svg')  # saves the figure in SVG format
            plt.savefig(fig_name+'.pdf')  # saves the figure in PDF format
            #plt.savefig(fig_name+'.eps')  # saves the figure in EPS format
        else: 
            plt.show()

    def generate_samples(self, n_samples: int = 1) -> np.array:
        samples, _ = self.bgm.sample(n_samples)
        return samples

def train_and_save(priors: bool = True, columns=['Type','teff_val','Period','abs_Imag']) -> None:
    data = pd.read_csv(PATH_PP)
    print(data.columns)
    df_selected_columns = data[columns]
    classes = df_selected_columns.Type.unique()
    print(classes)
    columns.remove('Type')
    mean_prior_dict = load_yaml_priors(PATH_PRIOS)
    for star_class in classes:
        print(star_class)
        star_type_data = mean_prior_dict['StarTypes'][star_class]
        components = len([key for key in star_type_data.keys() if key != 'CompleteName'])
        df_filtered_by_class = df_selected_columns[df_selected_columns.Type==star_class]
        X = df_filtered_by_class[columns]
        X = X.dropna()
        if X.shape[0] > 30:
            bgmm = BayesianGaussianMixtureModel(n_components=components, random_state=42)
            if priors:
                array_midpoints = extract_midpoints(mean_prior_dict['StarTypes'][star_class])
                try:
                    array_midpoints = extract_midpoints(mean_prior_dict['StarTypes'][star_class])
                    print('array_midpoints: ', array_midpoints)
                    bgmm.train(X, mean_prior=array_midpoints)
                except Exception as error:
                    print(error)
                    bgmm.train(X, mean_prior=None)
            else: 
                bgmm.train(X, mean_prior=None)
            bgmm.save_model('models/bgm_model_'+str(star_class)+'_priors_'+str(priors)+'_PP_'+str(len(columns))+'.pkl')

            for col1, col2 in combinations(columns, 2):
                bgmm.plot_2d_bgmm(bgmm, X, star_class, feature1 = col1, feature2= col2, priors=priors)


def fit_gausians(priors_dict):
    #TODO:refactor code in order to manage priors here and only fit in train and save
    train_and_save(priors = True, columns=['Type','teff_val','Period','abs_Imag'])

def get_load_and_sample(star_class: str = 'RRLYR') -> None:
    # Load the model and generate samples
    train_and_save(components=2)
    loaded_bgmm = BayesianGaussianMixtureModel.load_model('bgm_model_'+str(star_class)+'.pkl')
    generated_samples = loaded_bgmm.generate_samples(n_samples=5)
