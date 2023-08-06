import os, sys
import socket
import torch
import random 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import pandas as pd
import seaborn as sns
sys.path.append('../')
from src.vae_models import *
from src.datasets import Astro_lightcurves
from src.utils import *
import warnings
warnings.filterwarnings('ignore')

save_plots = False
save_tables = False

def setup_environment(ID, gpu=False):
    main_path = os.path.dirname(os.getcwd())
    PATH_DATA = "/home/franciscoperez/Documents/GitHub/CNN-PELSVAE/src/data"
    if not os.path.exists('%s/wandb/run--%s/VAE_model_None.pt' % (main_path, ID)):
        print('Downloading files from Weight & Biases')
        api = wandb.Api()
        run = api.run('jorgemarpa/Phy-VAE/%s' % (ID))
        run.file('VAE_model_None.pt').download(replace=True, root='%s/wandb/run--%s/' % (main_path, ID))
        run.file('config.yaml').download(replace=True, root='%s/wandb/run--%s/' % (main_path, ID))
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    vae, config = load_model_list(ID=ID)
    return vae, config, device

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

def evaluate_model(vae, dataloader, config, num_cls):
    mu, std = evaluate_encoder(vae, dataloader, config, n_classes=num_cls, force=False)
    print('mu shape: ', mu.shape)       
    columns = list(mu.columns)
    columns.remove('class')
    mu_values = mu[columns].values
    std_values = std[columns].values
    z = mu_values + std_values * np.random.normal(0, 1, (mu.shape[0], len(columns)))
    z = pd.DataFrame(z)
    z['label'] = mu['class']
    print('z shape: ', z.shape)
    #z = pd.get_dummies(z, columns=['label'], prefix = ['category'])
    return z

def train_and_test_models(z, examples, dataset):
    features = examples[['Type','teff_val','Period','abs_Imag']] #include class
    features = pd.get_dummies(features, columns=['Type'], prefix = ['category'])
    features = features.fillna(features.mean()) #TODO: impute by class
    for z_i in z.columns: 
        X = features
        y = z[z_i]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(random_state=42)
        parameters = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15]}
        clf = GridSearchCV(model, parameters)
        clf.fit(X_train, y_train)
        filename = f'model_{z_i}.sav'
        pickle.dump(clf, open(filename, 'wb'))
        loaded_model = pickle.load(open(filename, 'rb'))
        y_pred = loaded_model.predict(X_test)
        print_metrics_regression(y_test, y_pred)
        plot_figures(y_test, y_pred)

def print_metrics_regression(y_test, y_pred):
    print('Mean Squared Error (MSE): ', mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error (RMSE): ', sqrt(mean_squared_error(y_test, y_pred)))
    print('Mean Absolute Error (MAE): ', mean_absolute_error(y_test, y_pred))
    print('R^2 Score: ', r2_score(y_test, y_pred))

def scatter_plot(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('True vs Predicted Values')
    plt.show()

def residual_plot(y_test, y_pred):
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals)
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

def main():
    ID = 'b68z1hwo'
    gpu = False 
    vae, config, device = setup_environment(ID, gpu)
    dataset = prepare_dataset(config)
    dataloader, _ = dataset.get_dataloader(batch_size=100, test_split=0., shuffle=False)
    num_cls = dataset.labels_onehot.shape[1]
    
    z = evaluate_model(vae, dataloader, config, num_cls)
    
    examples = []
    meta_aux = dataset.meta.reset_index()
    #objects_by_class = {'ACEP':1000, 'CEP': 1000,  'DSCT': 1000,  'ECL':1000,  'ELL': 1000,  'LPV': 1000,  'RRLYR':  1000,  'T2CEP':1000}
    
    for i, cls in enumerate(dataset.label_onehot_enc.categories_[0]):
        aux = meta_aux.query('Type == "%s"' % (cls))#.sample(objects_by_class[cls])
        z_cl = z[z.label==cls] 
        examples.append(aux)
    
        examples = pd.concat(examples, axis=0)
        print(z_cl.shape)
        print(examples.shape)
        train_and_test_models(z_cl, examples, dataset)

if __name__ == "__main__":
    main()
