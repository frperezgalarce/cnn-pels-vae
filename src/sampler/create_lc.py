import os, sys
import socket
import torch
import pandas as pd
from typing import List, Optional, Union, Dict, Tuple, Any
sys.path.append('../')
from src.vae.vae_models import *
from src.utils import *
from src.vae.datasets import Astro_lightcurves
import warnings
warnings.filterwarnings('ignore')
import yaml

main_path: str = os.path.dirname(os.getcwd())

with open('src/paths.yaml', 'r') as file:
    YAML_FILE: Dict[str, Any] = yaml.safe_load(file)

PATHS: Dict[str, str] = YAML_FILE['paths']
PATH_DATA: str = PATHS["PATH_DATA_FOLDER"]
save_plots: bool = False
save_tables: bool = False

with open('src/regressor.yaml', 'r') as file:
    config_file: Dict[str, Any] = yaml.safe_load(file)

ID: str = config_file['model_parameters']['ID']

#ID: str = '1pjeearx' #'20twxmei' #'b68z1hwo'  #'7q2bduwv' #'b68z1hwo''yp4qdw1r'
gpu: bool = True # fail when true is selected

device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")

def prepare_dataset(config: Dict[str, Union[str, bool, int]]) -> Astro_lightcurves:
    dataset: Astro_lightcurves = Astro_lightcurves(survey=config['data'],
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
    return dataset

#Incorporate samples for pp and latent space to generate light curves
def main(samples: np.ndarray, z_hat: np.ndarray) -> None:
    print('cuda: ', torch.cuda.is_available())
    print('create main 1')
    vae, config = load_model_list(ID=ID, device=device)
    config['phys_params'] = 'PTAMR'
    config['physics_dim'] = 5
    print('create main 2')

    dataset = prepare_dataset(config)
    print('create main 3')

    num_cls = dataset.labels_onehot.shape[1]

    dataloader, _ = dataset.get_dataloader(batch_size=100, 
                                        test_split=0., shuffle=False)

    #mu, std = evaluate_encoder(vae, dataloader, config, 
    #                        n_classes=num_cls, force=False)       

    examples = []
    meta_aux = dataset.meta.reset_index()

    objects_by_class = {'ACEP':0, 'CEP': 0,  'DSCT': 0,  'ECL':0,  'ELL': 0,  'LPV': 0,  'RRLYR':  24,  'T2CEP':0}
    for i, cls in enumerate(dataset.label_onehot_enc.categories_[0]):
        aux = meta_aux.query('Type == "%s"' % (cls)).sample(objects_by_class[cls])
        examples.append(aux)

    examples = pd.concat(examples, axis=0)
    sample_period = np.round(examples['Period'].to_list(),2)

    data, lb, onehot, pp = dataset[examples.index]

    pp2 = add_perturbation(pp, scale=5.0)

    data = torch.from_numpy(data).to(device)
    onehot = torch.from_numpy(onehot).to(device)
    pp = torch.from_numpy(pp).to(device)
    pp2 = torch.from_numpy(pp2).to(device)

    print('device: ', device)

    print('torch assigned')
    print('labels: ', config['label_dim'])
    print('physics dim: ', config['physics_dim'])
    if config['label_dim'] > 0 and config['physics_dim'] > 0:
        xhat_z, mu_, logvar_, z_ = vae(data, label=onehot, phy=pp)
        xhat_mu = vae.decoder(mu_, data[:,:,0], label=onehot, phy=pp)
        xhat_z2, mu_2, logvar_2, z_2 = vae(data, label=onehot, phy=pp2)
        xhat_mu2 = vae.decoder(mu_2, data[:,:,0], label=onehot, phy=pp2)
    elif config['label_dim'] > 0 and config['physics_dim'] == 0:
        print('path 2')
        xhat_z, mu_, logvar_, z_ = vae(data, label=onehot)
        xhat_mu = vae.decoder(mu_, data[:,:,0], label=onehot)
    elif config['label_dim'] == 0:
        print('path 3')
        xhat_z, mu_, logvar_, z_ = vae(data)
        xhat_mu = vae.decoder(mu_, data[:,:,0])
    else:
        print('Check conditional dimension...')

    print('z hat extracted')
    xhat_z = torch.cat([data[:,:,0].unsqueeze(-1), xhat_z], dim=-1).cpu().detach().numpy()
    xhat_mu = torch.cat([data[:,:,0].unsqueeze(-1), xhat_mu], dim=-1).cpu().detach().numpy()
    xhat_mu2 = torch.cat([data[:,:,0].unsqueeze(-1), xhat_mu2], dim=-1).cpu().detach().numpy()

    data = data.cpu().detach().numpy()

    plot_wall_lcs(xhat_mu, data, cls=lb, save=save_plots) #data is real_lc

    plot_wall_lcs(xhat_mu, xhat_mu2, cls=lb, save=save_plots) #data is real_lc

    plot_wall_synthetic_lcs(xhat_mu,  cls=lb,  save=save_plots)

    lc_reverted = revert_light_curve(sample_period, xhat_mu, classes = lb)

    print('lc_reverted: ', lc_reverted.shape)
    print('xhat_mu: ', xhat_mu.shape)

    compare_folded_crude_lc(xhat_mu, lc_reverted, cls=lb, period=sample_period)

    save_arrays_to_folder(lc_reverted, lb, PATH_DATA)

if __name__ == "__main__":
    main()