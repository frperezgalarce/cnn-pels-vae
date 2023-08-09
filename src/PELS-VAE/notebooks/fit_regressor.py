import socket
import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
import pickle
import seaborn as sns
import wandb
import warnings
import matplotlib.pyplot as plt

sys.path.append('../')
from src.datasets import Astro_lightcurves
from src.utils import evaluate_encoder, load_model_list
warnings.filterwarnings('ignore')
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# You can now access your settings as:
PATH_DATA = config['PATH_DATA']

#PATH_DATA = "/home/franciscoperez/Documents/GitHub/CNN-PELSVAE/src/data"

save_plots = False
save_tables = False

def setup_environment(ID, gpu=False):
    main_path = os.path.dirname(os.getcwd())
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

def print_metrics_regression(y_test, y_pred):
    print('Mean Squared Error (MSE): ', mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error (RMSE): ', sqrt(mean_squared_error(y_test, y_pred)))
    print('Mean Absolute Error (MAE): ', mean_absolute_error(y_test, y_pred))
    print('R^2 Score: ', r2_score(y_test, y_pred))

def scatter_plot(y_test, y_pred):
    plt.scatter(y_test, y_pred, alpha=0.05)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('True vs Predicted Values')
    plt.show()

def residual_plot(y_test, y_pred):
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.05)
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

def load_predict(pp, filename = 'file.pkl'): 
    loaded_model = pickle.load(open(filename, 'rb'))
    z = loaded_model.predict(pp)
    return z

def save_model(model, filename='filename_model.pkl'): 
    pickle.dump(model, open(filename, 'wb'))

def train_model(reg, config_dic, name):
    model = reg(**config_dic[name])
    
    try:
        model.fit(p, z)
    except MemoryError:
        print('Fail')
        continue

    return model

def main():

    phys2 = ['abs_Gmag', 'teff_val', 'Period']
    ID = 'b68z1hwo'#'b68z1hwo' #'7q2bduwv'
    gpu = True 
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

    for name, reg in regressors.items():
        
        model = train_model(reg, config_dict, name)

        save_model(model, filename=name+'.pkl')
        
        z_hat = load_predict(p, filename=name+'.pkl')

        print_metrics_regression(z, z_hat)

        plot_figures(z, z_hat)

if __name__ == "__main__":
    main()