import os, glob, re, sys
import socket
import torch
#import wandb
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb
import pandas as pd
#import umap
from sklearn.manifold import TSNE
from tqdm import tqdm_notebook
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

sys.path.append('../')
from src.vae.vae_models import *
from src.vae.datasets import Astro_lightcurves
from src.utils import *

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from math import sqrt

def print_metrics_regression(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    print('Mean Squared Error (MSE): ', mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error (RMSE): ', sqrt(mean_squared_error(y_test, y_pred)))
    print('Mean Absolute Error (MAE): ', mean_absolute_error(y_test, y_pred))
    print('R^2 Score: ', r2_score(y_test, y_pred))


def train_rf_with_gs(p: np.ndarray, z: np.ndarray):

    # Split the data into training and test sets (70% train, 30% test)
    p_train, p_test, z_train, z_test = train_test_split(p, z, test_size=0.1, random_state=42)

    # Define the hyperparameters and their possible values for RandomForestRegressor
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Setup the GridSearch with 5-fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=kf, verbose=1, scoring='neg_mean_absolute_error', n_jobs=-1)
    # Fit the data to perform the grid search on training set
    grid_search.fit(p_train, z_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Train the RandomForestRegressor using the best hyperparameters on the full training set
    rf_regressor = RandomForestRegressor(**best_params)

    rf_regressor.fit(p_train, z_train)

    # Evaluate the trained model on test set
    z_pred = rf_regressor.predict(p_test)

    print_metrics_regression(z_test, z_pred)

    return rf_regressor, best_params

main_path = os.path.dirname(os.getcwd())

save_plots = False
save_tables = False

import gzip
data_path = "/home/franciscoperez/Documents/GitHub/CNN-PELSVAE2/cnn-pels-vae/data/time_series/real/OGLE3_lcs_I_meta_snr5_augmented_folded_trim600_GAIA3_6PP.npy.gz"
with gzip.open(data_path, 'rb') as f:
    test = np.load(f, allow_pickle=True)

ID = '3v5kanq1'
gpu = True

device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")

vae, config = load_model_list(ID=ID)

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
print('Using physical parameters: ', dataset.phy_names)
dataset.remove_nan()
print(dataset.class_value_counts())
print('Total: ', len(dataset))
num_cls = dataset.labels_onehot.shape[1]

dataloader, _ = dataset.get_dataloader(batch_size=100, 
                                       test_split=0., shuffle=False)


mu, std = evaluate_encoder(vae, dataloader, config, force=False)

examples = []
meta_aux = dataset.meta.reset_index()
for i, cls in enumerate(dataset.label_onehot_enc.categories_[0]):
    aux = meta_aux.query('Type == "%s"' % (cls)).sample(3)
    examples.append(aux)
examples = pd.concat(examples, axis=0)


data, lb, onehot, pp = dataset[examples.index]
data = torch.from_numpy(data).to(device)
onehot = torch.from_numpy(onehot).to(device)
pp = torch.from_numpy(pp).to(device)

if config['label_dim'] > 0 and config['physics_dim'] > 0:
    xhat_z, mu_, logvar_, z_, cop_loss = vae(data, label=onehot, phy=pp)
    xhat_mu = vae.decoder(mu_, data[:,:,0], label=onehot, phy=pp)
elif config['label_dim'] > 0 and config['physics_dim'] == 0:
    xhat_z, mu_, logvar_, z_, cop_loss = vae(data, label=onehot)
    xhat_mu = vae.decoder(mu_, data[:,:,0], label=onehot)
elif config['label_dim'] == 0:
    xhat_z, mu_, logvar_, z_, cop_loss = vae(data)
    xhat_mu = vae.decoder(mu_, data[:,:,0])
else:
    print('Check conditional dimension...')

xhat_z = torch.cat([data[:,:,0].unsqueeze(-1), xhat_z], dim=-1).cpu().detach().numpy()

xhat_mu = torch.cat([data[:,:,0].unsqueeze(-1), xhat_mu], dim=-1).cpu().detach().numpy()
data = data.cpu().detach().numpy()

phys2 = ['Period','teff_val', 'abs_Gmag', 'radius_val', '[Fe/H]_J95', 'logg']

meta_ = dataset.meta.dropna(subset=phys2)
mu_ = mu.iloc[:, :-1].values
std_ = std.iloc[:, :-1].values
mu_ = mu_.astype(np.float64)
std_ = std_.astype(np.float64)

unique_idx = meta_.reset_index().drop_duplicates(subset=['OGLE_id']).index
meta_u = meta_.iloc[unique_idx]
mu_u = mu_[unique_idx]
std_u = std_[unique_idx]
meta_u.shape, mu_u.shape, std_u.shape

meta__ = meta_.copy()
mu__ = mu_.copy()
std__ = std_.copy()

meta_ = meta_u.copy()
mu_ = mu_u.copy()
std_ = std_u.copy()

regressors = {'RFR': RandomForestRegressor}

config_dict = dict(RFR=dict(n_estimators=100,
                            criterion='mse',
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_features=len(phys2),
                            n_jobs=-1))

p = meta_.loc[:,phys2].values
z = mu_.copy()

all_lcs = []

for name, reg in regressors.items():
    if name == 'GPR': continue
    
    model = reg(**config_dict[name])
    try:
        model.fit(p, z)
        #model, _ = train_rf_with_gs(p, z)
    except MemoryError:
        continue
    mse = metrics.mean_squared_error(z, model.predict(p))
    print_metrics_regression(z, model.predict(p))
    filename = 'models/RF_2808.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print('mse  : ', mse)

