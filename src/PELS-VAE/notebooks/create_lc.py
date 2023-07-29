import os, glob, re, sys
import socket
import torch
#import wandb
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb
import pandas as pd
#import umap
from sklearn.manifold import TSNE
from tqdm import tqdm_notebook

sys.path.append('../')
from src.vae_models import *
from src.datasets import Astro_lightcurves
from src.utils import *

import warnings
warnings.filterwarnings('ignore')

main_path = os.path.dirname(os.getcwd())

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
#config['normed'] = False
print(config)

dataset = Astro_lightcurves(survey=config['data'],
                            band='I' if config['data'] else 'B',
                            use_time=True,
                            use_err=True,
                            norm=config['normed'],
                            folded=config['folded'],
                            machine=socket.gethostname(),
                            seq_len=config['sequence_lenght'],
                            phy_params=config['phys_params'])

print('here')

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


mu, std = evaluate_encoder(vae, dataloader, config, 
                           n_classes=num_cls, force=False)       

print('mu: ', mu.shape)

print('std: ', std.shape)

examples = []
meta_aux = dataset.meta.reset_index()
for i, cls in enumerate(dataset.label_onehot_enc.categories_[0]):
    aux = meta_aux.query('Type == "%s"' % (cls)).sample(3)
    examples.append(aux)
examples = pd.concat(examples, axis=0)
print(examples.index)

print('examples: ', examples)

data, lb, onehot, pp = dataset[examples.index]


print('data: {}, lb: {}, onehot: {}, pp: {}'.format(data, lb, onehot, pp))

data = add_perturbation(data, scale=0.0)

#print('data: ', data)
#print('data2: ', data2)

#this should be modified to generate new light curves

data = torch.from_numpy(data).to(device)
onehot = torch.from_numpy(onehot).to(device)
pp = torch.from_numpy(pp).to(device)

if config['label_dim'] > 0 and config['physics_dim'] > 0:
    xhat_z, mu_, logvar_, z_ = vae(data, label=onehot, phy=pp)
    xhat_mu = vae.decoder(mu_, data[:,:,0], label=onehot, phy=pp)
elif config['label_dim'] > 0 and config['physics_dim'] == 0:
    xhat_z, mu_, logvar_, z_ = vae(data, label=onehot)
    xhat_mu = vae.decoder(mu_, data[:,:,0], label=onehot)
elif config['label_dim'] == 0:
    xhat_z, mu_, logvar_, z_ = vae(data)
    xhat_mu = vae.decoder(mu_, data[:,:,0])
else:
    print('Check conditional dimension...')

xhat_z = torch.cat([data[:,:,0].unsqueeze(-1), xhat_z], dim=-1).detach().numpy()
xhat_mu = torch.cat([data[:,:,0].unsqueeze(-1), xhat_mu], dim=-1).detach().numpy()
data = data.detach().numpy()

plot_wall_lcs(xhat_mu, data, cls=lb, save=save_plots) #data is real_lc

plot_wall_synthetic_lcs(xhat_mu,  cls=lb,  save=save_plots)

lc_reverted = revert_light_curve(2, xhat_mu[0,:,:])



plt.show()
plt.figure(figsize=(10, 5))

# Scatter plot 1
plt.subplot(1, 2, 1)
plt.scatter(xhat_mu[0,:,0], xhat_mu[0,:,1], label='Folded light curve')
plt.gca().invert_yaxis()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Folded light curve from PELS-VAE')
plt.legend()

# Scatter plot 2
plt.subplot(1, 2, 2)
plt.scatter(lc_reverted[:,0], lc_reverted[:,1], label='Recover light curve')
plt.gca().invert_yaxis()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Recovered light curve')
plt.legend()


# Adjust spacing between plots
plt.tight_layout()

# Show the figure
plt.show()
