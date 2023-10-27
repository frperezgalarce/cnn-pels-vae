import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from typing import Optional, Dict, Union, Tuple, ClassVar
from src.utils import load_yaml_priors, extract_midpoints, extract_maximum_of_max_periods
from sklearn.neighbors import NearestNeighbors
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
    def __init__(self, n_components: int = 2, random_state: Optional[int] = None, 
                covariance_type = 'full', max_iter = 500):
        self.n_components = n_components
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.bgm = BayesianGaussianMixture(n_components=self.n_components, 
                                            random_state=self.random_state, 
                                            covariance_type=self.covariance_type, 
                                            max_iter = self.max_iter)
        self.object = None
        self.mean_prior = None

    def train(self, X: pd.DataFrame, mean_prior: Optional[np.array] = None) -> None:
        self.mean_prior = mean_prior
        if mean_prior is not None:
            self.bgm.mean_prior_ = mean_prior
        self.bgm.fit(X)
        self.object = self.bgm  # Saving the trained model as the object attribute
        # Accessing and printing specific attributes
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

    def plot_2d_bgmm(self, bgmm: "BayesianGaussianMixtureModel", 
                    X: pd.DataFrame, starClass: str, feature1: str = 'abs_Gmag', 
                    feature2: str = 'teff_val', save=True, priors: bool = False, 
                    number_of_features: int = 6, save_path: str = None) -> None:
            """
            Plots a 2D Bayesian Gaussian Mixture Model.

            Parameters:
                bgmm: BayesianGaussianMixtureModel object
                X: pd.DataFrame containing the data points
                starClass: String representing the star class
                feature1: First feature to plot (Default 'abs_Gmag')
                feature2: Second feature to plot (Default 'teff_val')
                save: Boolean to save the plot (Default True)
                priors: Boolean to indicate the use of priors (Default False)
                number_of_features: Number of features in the model (Default 6)
                save_path: Path where to save the figures (Default None)
            """
            
            # Check if the feature columns exist in the DataFrame
            if feature1 not in X.columns or feature2 not in X.columns:
                print(f"Error: Feature columns {feature1} or {feature2} not found in the DataFrame.")
                return
            plt.scatter(X[feature1], X[feature2], c='blue', alpha=0.5, label='Data Points')
            plt.xlim(0.9*X[feature1].min(), 1.1*X[feature1].max())
            plt.ylim(0.9*X[feature2].min(), 1.1*X[feature2].max())
            plt.xlabel(feature1)
            plt.ylabel(feature2)

            # Create a meshgrid to cover the data range
            x_mesh, y_mesh = np.meshgrid(np.linspace(X[feature1].min(), X[feature1].max(), 100),
                                        np.linspace(X[feature2].min(), X[feature2].max(), 100))
            
            # Prepare meshgrid for scoring. Here we're considering only feature1 and feature2
            # for visualization, but we want to score using all features.
            mesh_df = pd.DataFrame({feature1: x_mesh.ravel(), feature2: y_mesh.ravel()})
            
            X_values = X[[feature1, feature2]].values
            mesh_values = mesh_df[[feature1, feature2]].values

            # Fit k-NN model
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(X_values)

            # Find the indices of the nearest points in X for each point in mesh_df
            _, indices = knn.kneighbors(mesh_values)

            # Use these indices to fill in the missing columns in mesh_df
            for col in X.columns:
                if col not in [feature1, feature2]:
                    mesh_df[col] = X.iloc[indices.flatten()][col].values
            # Calculate the log-probability of the points on the meshgrid

            log_prob = bgmm.bgm.score_samples(mesh_df)
            # Make them numerically stable (Shift to near zero)
            max_log_prob = np.max(log_prob)
            z = np.exp(log_prob - max_log_prob)
            # Check shapes to make sure they match
            if z.shape[0] == np.prod(x_mesh.shape):
                # Reshape z to match x_mesh and y_mesh
                z = z.reshape(x_mesh.shape)
            else:
                print(f"Shape mismatch: z has shape {z.shape[0]} but x_mesh and y_mesh require {np.prod(x_mesh.shape)}")
                # Here you could either skip plotting or perform some other action to handle the mismatch

            # Check if z.min() and z.max() are distinct
            if np.isclose(np.quantile(z, 0.5),  np.quantile(z, 0.95)):
                print("Warning: Minimum and maximum probabilities are too close. Adjusting levels for contour plot.")
                levels = np.linspace(0, 1, 10000)  # or some other range of levels you're interested in
            else:
                levels = np.linspace(np.quantile(z, 0.5), np.quantile(z, 0.95), 10000)

            # Plotting the contour plots of the Gaussian components, assuming z's shape is correct
            if z.shape == x_mesh.shape:
                plt.contourf(x_mesh, y_mesh, z, levels=levels, cmap=plt.cm.Blues, alpha=0.9)
            else:
                print("Skipping contour plot due to shape mismatch.")

            plt.title('Gaussian Mixture Model Components - ' + str(starClass))
            #plt.legend()
            plt.colorbar(label='Log Probability')
            if save:
                fig_name = starClass+'_'+ feature1+'_'+feature2+'_priors_'+str(priors)+'_PP_'+str(X.shape[1])
                fig_name = PATH_FIGURES+'Gaussians/'+str(fig_name).replace('[', '').replace(']','').replace('_','').replace('/','')
                plt.savefig(fig_name+'.png', dpi=300)
                plt.savefig(fig_name+'.svg')  # saves the figure in SVG format
                plt.savefig(fig_name+'.pdf')  # saves the figure in PDF format
                plt.close()
                #plt.savefig(fig_name+'.eps')  # saves the figure in EPS format
            else: 
                plt.show()

    def generate_samples(self, n_samples: int = 1) -> np.array:
        samples, _ = self.bgm.sample(n_samples)
        return samples

def train_and_save(priors: bool = True, columns=['Type','Period', 'teff_val', '[Fe/H]_J95', 'abs_Gmag', 'radius_val', 'logg'], plot_or_save_figs=False) -> None:
    data = pd.read_csv(PATH_PP)
    df_selected_columns = data[columns]
    classes = df_selected_columns.Type.unique()
    columns.remove('Type')

    mean_prior_dict = load_yaml_priors(PATH_PRIOS)
    print(mean_prior_dict)
    for star_class in classes:
        print(star_class)
        star_type_data = mean_prior_dict['StarTypes'][star_class]
        components = len([key for key in star_type_data.keys() if (key != 'CompleteName') and (key!='max_period')])
        df_filtered_by_class = df_selected_columns[df_selected_columns.Type==star_class]
        X = df_filtered_by_class[columns]
        X = X.dropna()
        print(mean_prior_dict['StarTypes'][star_class]['max_period'])
        period_upper_limit = mean_prior_dict['StarTypes'][star_class]['max_period']
        X = X[X.Period<period_upper_limit]
        print(X)
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

            if plot_or_save_figs:
                for col1, col2 in combinations(columns, 2):
                    bgmm.plot_2d_bgmm(bgmm, X, star_class, feature1 = col1, 
                                    feature2= col2, priors=priors, 
                                    number_of_features=len(columns), save=False)


def fit_gausians(priors_dict, columns = ['Type','Period', 'teff_val', '[Fe/H]_J95', 'abs_Gmag', 'radius_val', 'logg']):
    train_and_save(priors = False, columns= columns)