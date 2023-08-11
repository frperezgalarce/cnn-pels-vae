# Import necessary libraries
import socket
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
import pickle
import seaborn as sns
import wandb
import warnings
import matplotlib.pyplot as plt
import yaml
import sys, os

#sys.path.append('./')
from src.datasets import Astro_lightcurves
from src.utils import evaluate_encoder, load_model_list
warnings.filterwarnings('ignore')

# Read configurations from a YAML file
with open('src/regressor.yaml', 'r') as file:
    config_file = yaml.safe_load(file)

# Extracting path configurations
PATH_DATA = config_file['PATH_DATA']
save_plots = config_file['save_plots']
save_tables = config_file['save_tables']

# Function to set up environment and download model weights if not available
def setup_environment(ID, gpu=False):
    main_path = os.path.dirname(os.getcwd())
    print(main_path)
    if not os.path.exists('%s/wandb/src/run--%s/VAE_model_None.pt' % (main_path, ID)):
        print('Downloading files from Weight & Biases')
        api = wandb.Api()
        print(ID)
        run = api.run('jorgemarpa/Phy-VAE/%s' % (ID))
        run.file('VAE_model_None.pt').download(replace=True, root='%s/wandb/run--%s/' % (main_path, ID))
        run.file('config.yaml').download(replace=True, root='%s/wandb/run--%s/' % (main_path, ID))
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    vae, config = load_model_list(ID=ID)
    return vae, config, device

# Function to prepare the dataset
def prepare_dataset(config):
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
def print_metrics_regression(y_test, y_pred):
    print('Mean Squared Error (MSE): ', mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error (RMSE): ', sqrt(mean_squared_error(y_test, y_pred)))
    print('Mean Absolute Error (MAE): ', mean_absolute_error(y_test, y_pred))
    print('R^2 Score: ', r2_score(y_test, y_pred))

# Functions to plot various diagnostic plots
def scatter_plot(y_test, y_pred):
    plt.scatter(y_test, y_pred, alpha=config_file['plotting']['scatter_plot_alpha'])
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=config_file['plotting']['scatter_plot_line_width'])
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('True vs Predicted Values')
    plt.show()

def residual_plot(y_test, y_pred):
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=config_file['plotting']['residual_plot_alpha'])
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='k', linestyles='solid')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

def prediction_error_plot(y_test, y_pred):

    residuals = y_test - y_pred
    sns.distplot(residuals)
    plt.title('Prediction Error Plot')
    plt.xlabel('Prediction Error')
    plt.show()

def plot_figures(y_test, y_pred):
    scatter_plot(y_test, y_pred)
    residual_plot(y_test, y_pred)
    prediction_error_plot(y_test, y_pred)

# Function to load predictions from a saved model
def load_predict(pp, filename = 'file.pkl'): 
    loaded_model = pickle.load(open(filename, 'rb'))
    z = loaded_model.predict(pp)
    return z

# Function to save trained model to disk
def save_model(model, filename='filename_model.pkl'): 
    pickle.dump(model, open(filename, 'wb'))

# Function to train the model using specified configurations
def train_model(reg, config_dic, name, p, z):
    model = reg(**config_dic[name])
    
    try:
        model.fit(p, z)
    except MemoryError:
        print('Fail') 
    return model

# Main function to set up the model and training process
def main():

    phys2 = ['abs_Gmag', 'teff_val', 'Period']
    ID = config_file['model_parameters']['ID'] #'b68z1hwo'#'b68z1hwo' #'7q2bduwv'
    gpu = config_file['model_parameters']['gpu'] #True 
    vae, config, _ = setup_environment(ID, gpu)
    dataset = prepare_dataset(config)
    dataloader, _ = dataset.get_dataloader(batch_size=100, test_split=0., shuffle=False)
    num_cls = dataset.labels_onehot.shape[1]
    
    mu, _ = evaluate_encoder(vae, dataloader, config, 
                           n_classes=num_cls, force=False)

    meta_ = dataset.meta.dropna(subset=phys2)
    if len(mu) > 30000:
        mu_ = mu.loc[meta_.index].values[:,:-1]
    else:
        mu_ = mu.iloc[:, :-1].values
    mu_ = mu_.astype(np.float64)

    unique_idx = meta_.reset_index().drop_duplicates(subset=['OGLE_id']).index
    meta_u = meta_.iloc[unique_idx]
    mu_u = mu_[unique_idx]

    meta_ = meta_u.copy()
    mu_ = mu_u.copy()

    regressors = {'RFR': RandomForestRegressor}

    config_dict = dict(config_file['regressors'])
    p = meta_.loc[:, phys2].values
    z = mu_.copy()

    for name, reg in regressors.items():
        filename = name + '.pkl'
        
        # Check if the model file exists
        if os.path.exists(filename):
            print(f"Loading existing model from {filename}")
            model = pickle.load(open(filename, 'rb'))
        else:
            print(f"Training new model {name}")
            model = train_model(reg, config_dict, name, p, z)
            save_model(model, filename=filename)

        z_hat = model.predict(p) # Directly using the model for prediction

        print_metrics_regression(z, z_hat)
        plot_figures(z, z_hat)
        
if __name__ == "__main__":
    main()