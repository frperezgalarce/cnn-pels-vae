import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from typing import Optional, Dict, Union, Tuple, ClassVar, Any
from src.utils import load_yaml_priors, extract_midpoints, extract_maximum_of_max_periods
from sklearn.neighbors import NearestNeighbors
import yaml
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

with open('src/configuration/paths.yaml', 'r') as file:
    YAML_FILE: str = yaml.safe_load(file)
PATHS: str = YAML_FILE['paths']
PATH_PRIOS: str = PATHS['PATH_PRIOS']
PATH_PP: str = PATHS['PATH_PP']

with open('src/configuration/regressor.yaml', 'r') as file:
    config_file: Dict[str, Any] = yaml.safe_load(file)

sufix_path: str = config_file['model_parameters']['sufix_path']


class BayesianGaussianMixtureModel:
    def __init__(self, n_components: int = 2, random_state: Optional[int] = None, 
                covariance_type = 'full', max_iter = 5000, mean_prior=None):
        self.n_components = n_components
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.object = None
        self.mean_prior = mean_prior
        self.bgm = BayesianGaussianMixture(n_components=self.n_components, 
                                            random_state=self.random_state, 
                                            covariance_type=self.covariance_type, 
                                            max_iter = self.max_iter, 
                                            mean_prior = self.mean_prior)

    def train(self, X: pd.DataFrame) -> None:
        self.bgm.fit(X)
        self.object = self.bgm  # Saving the trained model as the object attribute
        np.set_printoptions(precision=4, suppress=True)

        print("Hyperparameters:")
        print(self.bgm)  # This prints the hyperparameters

        print("\nFitted Means:")
        print(self.bgm.means_)

        print("\nFitted Covariances:")
        print(self.bgm.covariances_)


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

    def generate_samples(self, n_samples: int = 1) -> np.array:
        samples, _ = self.bgm.sample(n_samples)
        return samples

def train_and_save(priors: bool = True, 
                   columns=['Type','Period', 'teff_val', '[Fe/H]_J95', 'abs_Gmag', 'radius_val', 'logg']) -> None:
    
    data = pd.read_csv(PATH_PP)
    df_selected_columns = data[columns]
    classes = df_selected_columns.Type.unique()
    columns.remove('Type')

    mean_prior_dict = load_yaml_priors(PATH_PRIOS)

    for star_class in classes:
        print(star_class)
        star_type_data = mean_prior_dict['StarTypes'][star_class]
        components = len([key for key in star_type_data.keys()])-3
        df_filtered_by_class = df_selected_columns[df_selected_columns.Type==star_class]
        X = df_filtered_by_class[columns]
        X = X.dropna()
        if 'LOG' in sufix_path:
            X['Period']=np.log(X['Period']) 
            X['teff_val']=np.log(X['teff_val']) 
            X['radius_val']=np.log(X['radius_val']) 
            period_upper_limit = np.log(mean_prior_dict['StarTypes'][star_class]['max_period'])
            period_lower_limit = np.log(mean_prior_dict['StarTypes'][star_class]['min_period'])
        else: 
            period_upper_limit = mean_prior_dict['StarTypes'][star_class]['max_period']
            period_lower_limit = mean_prior_dict['StarTypes'][star_class]['min_period']
        X = X[X.Period<period_upper_limit]
        X = X[X.Period>period_lower_limit]

        if X.shape[0] > 30:
            
            if priors:
                try:
                    print(mean_prior_dict['StarTypes'][star_class])
                    array_midpoints = extract_midpoints(mean_prior_dict['StarTypes'][star_class])
                    array_midpoints = np.array(array_midpoints)
                    if 'LOG' in sufix_path: 
                        array_midpoints[:, [0, 1, 4]] = np.log(array_midpoints[:, [0, 1, 4]])
                    print(array_midpoints)

                    print(components)
                    bgmm = BayesianGaussianMixtureModel(n_components=components, random_state=42, mean_prior=np.mean(array_midpoints, axis=0))
                    print(bgmm)
                    bgmm.train(X)
                except Exception as error:
                    raise('The model was not trained')
            else:
                bgmm = BayesianGaussianMixtureModel(n_components=components, random_state=42, mean_prior=None)
                bgmm.train(X)

            bgmm.save_model('models/bgm_model_'+str(star_class)+'_priors_'+str(priors)+'_PP_'+str(len(columns))+'.pkl')

def fit_gaussians(priors=True, columns = ['Type','Period', 'teff_val', '[Fe/H]_J95', 'abs_Gmag', 'radius_val', 'logg']):
    train_and_save(priors = priors, columns= columns)