import os, sys
import socket
import torch
import pandas as pd
sys.path.append('../')
from src.vae_models import *
from src.datasets import Astro_lightcurves
from src.utils import *
import warnings
warnings.filterwarnings('ignore')
main_path = os.path.dirname(os.getcwd())
PATH_DATA = "/home/franciscoperez/Documents/GitHub/CNN-PELSVAE/src/data"

save_plots = False
save_tables = False

ID = 'b68z1hwo'
gpu = False # fail when true is selected

if not os.path.exists('%s/wandb/run--%s/VAE_model_None.pt' % 
                      (main_path, ID)):
    print('Downloading files from Weight & Biases')
    
    api = wandb.Api()
    run = api.run('jorgemarpa/Phy-VAE/%s' % (ID))
    run.file('VAE_model_None.pt').download(replace=True, 
                                           root='%s/wandb/run--%s/' % 
                                           (main_path, ID))
    run.file('config.yaml').download(replace=True, 
                                     root='%s/wandb/run--%s/' % 
                                     (main_path, ID))

device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")

vae, config = load_model_list(ID=ID)
config['phys_params'] = 'PTA'
config['physics_dim'] = 3

print(config)

pp = sample_pp(epoch_information, criteria='s1')

z = get_latent_space(pp)

lcs = decoder(z)

lc_reverted = revert_light_curve(sample_period, xhat_mu, classes = lb)

compare_folded_crude_lc(xhat_mu, lc_reverted, cls=lb, period=sample_period)

save_arrays_to_folder(lc_reverted, lb, PATH_DATA)