# Import necessary libraries
import socket
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
import pickle
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import yaml
import os
from typing import Tuple, Any, Dict, Type, Union, List
from sklearn import preprocessing

#sys.path.append('./')
from src.vae.datasets import Astro_lightcurves
from src.utils import evaluate_encoder, load_model_list
warnings.filterwarnings('ignore')

# Read configurations from a YAML file
with open('src/regressor.yaml', 'r') as file:
    reg_conf_file: Dict[str, Any] = yaml.safe_load(file)

# Extracting path configurations
PATH_DATA: str = reg_conf_file['PATH_DATA']
save_plots: bool = reg_conf_file['save_plots']
save_tables: bool = reg_conf_file['save_tables']



import pickle
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Type, Any, List, Union, Optional

def process_regressors(reg_conf_file: Dict[str, Any],
                       meta_: Optional[Dict[str, Any]] = None,
                       phys2: Optional[List[str]] = None,
                       mu_: Optional[List[float]] = None,
                       samples: Optional[List[float]] = None,
                       assess_regressor: bool = True,
                       train_rf: bool = False) -> None:
    """
    Process the regressors.

    Parameters:
        reg_conf_file (Dict[str, Any]): Configuration file.
        meta_ (Dict[str, Any], optional): Metadata.
        phys2 (List[str], optional): Physical parameters.
        mu_ (List[float], optional): mu values.
        samples (List[float], optional): Sample data.
        assess_regressor (bool): Whether to assess the regressor. Default is True.
        train_rf (bool): Whether to train random forest. Default is False.

    Returns:
        None
    """

    regressors = {'RFR': RandomForestRegressor}

    config_dict = dict(reg_conf_file['regressors'])

    if assess_regressor:
        if (meta_ is None) or (phys2 is None) or (mu_ is None):
            raise ValueError("Meta data, phys2, and mu_ must be provided if assess_regressor is True.")
        mm_scaler = preprocessing.MinMaxScaler()
        mm_scaler.fit(meta_[phys2].values.astype(np.float32))
        meta_ = mm_scaler.transform(meta_.loc[:,phys2].values.astype(np.float32))
        p = meta_.copy()
        z = mu_.copy()
    else:
        if samples is None:
            raise ValueError("Samples must be provided if assess_regressor is False.")
        print('pp: ')
        print(samples)
        p = samples

    for name, reg in regressors.items():
        filename = 'models/' + reg_conf_file['model_parameters']['ID']+'_'+name + '.pkl'
        # Check if the model file exists
        if train_rf:
            print(f"Training new model {name}")
            model = train_model(reg, config_dict, name, p, z)
            save_model(model, filename=filename)
        else:
            print(f"Loading existing model from {filename}")
            model = pickle.load(open(filename, 'rb'))
        z_hat = model.predict(p)  # Directly using the model for prediction
    print(z_hat)
    return z_hat


# Function to set up environment and download model weights if not available
def setup_environment(ID: str, gpu: bool = False) -> Tuple[Any, Dict[str, Any], torch.device]:
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    vae, config = load_model_list(ID=ID)
    return vae, config, device

# Function to prepare the dataset
def prepare_dataset(config: Dict[str, Any]) -> Astro_lightcurves:
    dataset = Astro_lightcurves(survey=config['data'],
                            band='I' if config['data'] else 'B',
                            use_time=True,
                            use_err=True,
                            norm=config['normed'],
                            folded=config['folded'],
                            machine=socket.gethostname(),
                            seq_len=config['sequence_lenght'],
                            phy_params=config['phys_params'])
    if config['classes'].split('_')[0] == 'drop':
        dataset.drop_class(config['classes'].split('_')[1])
    elif config['classes'].split('_')[0] == 'only':
        dataset.only_class(config['classes'].split('_')[1])
    dataset.remove_nan()

    return dataset

# Function to print regression metrics
def print_metrics_regression(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    print('Mean Squared Error (MSE): ', mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error (RMSE): ', sqrt(mean_squared_error(y_test, y_pred)))
    print('Mean Absolute Error (MAE): ', mean_absolute_error(y_test, y_pred))
    print('R^2 Score: ', r2_score(y_test, y_pred))

# Functions to plot various diagnostic plots
def scatter_plot(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    plt.scatter(y_test, y_pred, alpha=reg_conf_file['plotting']['scatter_plot_alpha'])
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=reg_conf_file['plotting']['scatter_plot_line_width'])
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('True vs Predicted Values')
    plt.show()

def residual_plot(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=reg_conf_file['plotting']['residual_plot_alpha'])
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='k', linestyles='solid')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

def prediction_error_plot(y_test: np.ndarray, y_pred: np.ndarray) -> None:

    residuals = y_test - y_pred
    sns.distplot(residuals)
    plt.title('Prediction Error Plot')
    plt.xlabel('Prediction Error')
    plt.show()

def plot_figures(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    scatter_plot(y_test, y_pred)
    residual_plot(y_test, y_pred)
    prediction_error_plot(y_test, y_pred)

# Function to load predictions from a saved model
def load_predict(pp: np.ndarray, filename: str = 'file.pkl') -> np.ndarray: 
    loaded_model = pickle.load(open(filename, 'rb'))
    z = loaded_model.predict(pp)
    return z

# Function to save trained model to disk
def save_model(model: Any, filename: str = 'filename_model.pkl') -> None:
    pickle.dump(model, open(filename, 'wb'))

# Function to train the model using specified configurations
def train_model(reg: Type, config_dic: Dict[str, Any], name: str, p: np.ndarray, z: np.ndarray) -> Any:
    model = reg(**config_dic[name])
    try:
        print('p: ', p.shape)
        print('z: ', z.shape)
        model.fit(p, z)
    except MemoryError:
        print('Fail') 
    return model

# Main function to set up the model and training process
def main(samples: Union[np.ndarray, List], vae_model, assess_regressor: bool = False, train_rf: bool = True, 
        phys2 = ['abs_Gmag', 'teff_val', 'Period', 'radius_val', '[Fe/H]_J95']) -> None:

    if assess_regressor:
        gpu = reg_conf_file['model_parameters']['gpu'] #True 
        print(vae_model)
        vae, config, _ = setup_environment(vae_model, gpu)
        dataset = prepare_dataset(config)
        train_dataloader, test_dataloader = dataset.get_dataloader(batch_size=100, test_split=0.0, shuffle=False)
        num_cls = dataset.labels_onehot.shape[1]
        mu, _ = evaluate_encoder(vae, train_dataloader, config, force=False)
        meta_ = dataset.meta.dropna(subset=phys2)
        mu_ = mu.iloc[:, :-1].values
        mu_ = mu_.astype(np.float64)
        print('meta: ', meta_.shape)
        print('mu: ', mu_.shape)
        unique_idx = meta_.reset_index().drop_duplicates(subset=['OGLE_id']).index
        meta_u = meta_.iloc[unique_idx]
        mu_u = mu_[unique_idx]
        meta_ = meta_u.copy()
        mu_ = mu_u.copy()
        z_hat = process_regressors(reg_conf_file, phys2=dataset.phy_names, meta_= meta_, mu_=mu_,
                                  samples= None, assess_regressor=assess_regressor, 
                                  train_rf=True)
    else: 
        z_hat = process_regressors(reg_conf_file=reg_conf_file, meta_=None,
                                phys2=phys2, mu_=None, samples=samples, 
                                assess_regressor=assess_regressor, train_rf=False)
    return z_hat
        
if __name__ == "__main__":
    main()