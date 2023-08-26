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
import src.sampler.fit_regressor as reg

main_path: str = os.path.dirname(os.getcwd())

with open('src/paths.yaml', 'r') as file:
    YAML_FILE: Dict[str, Any] = yaml.safe_load(file)

PATHS: Dict[str, str] = YAML_FILE['paths']
PATH_DATA: str = PATHS["PATH_DATA_FOLDER"]
save_tables: bool = False

with open('src/regressor.yaml', 'r') as file:
    reg_conf_file: Dict[str, Any] = yaml.safe_load(file)

ID: str = reg_conf_file['model_parameters']['ID']

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
def get_synthetic_light_curves(samples: np.ndarray, z_hat: np.ndarray, training_cnn: bool = False, 
        plot:bool = False, save_plot:bool = False, 
        objects_by_class: Dict = {'ACEP':12,  'RRLYR':  12}, 
        plot_reverted: bool = False,
        sensivity: Any = None, 
        column_to_sensivity = None) -> None:
    
    print('cuda: ', torch.cuda.is_available())
    print('model: ', ID)
    vae, config = load_model_list(ID=ID, device=device)
    
    if training_cnn:
        pass
    else:
        dataset = prepare_dataset(config)
        #num_cls = dataset.labels_onehot.shape[1]
        #dataloader, _ = dataset.get_dataloader(batch_size=100, 
        #                                   test_split=0., shuffle=False)

    #mu, std = evaluate_encoder(vae, dataloader, config, 
    #                        n_classes=num_cls, force=False)       
    if training_cnn: 
        pass
    else:
        examples = []
        meta_aux = dataset.meta.reset_index()
        first_key = list(objects_by_class.keys())[0]
        first_value = objects_by_class[first_key]
        if sensivity is None:
            aux = meta_aux.query('Type == "%s"' % first_key).sample(first_value)        
        else: 
            aux = meta_aux.query('Type == "%s"' % first_key).sample(1)

        examples.append(aux)
        examples = pd.concat(examples, axis=0)
        sample_period = np.round(examples['Period'].to_list(),2)

        data, lb, onehot, pp = dataset[examples.index]

        if sensivity is None: 
            pass
        else:
            data = np.tile(data[0], (24, 1, 1))
            lb = np.tile(lb[0], (24,))
            onehot = np.tile(onehot[0], (24, 1))
            pp = np.tile(pp[0], (24, 1))
    
        if sensivity is None:
            pass
        else:
            print('conducting sensivity on: ' + sensivity)
            print(dataset.phy_names)
            column_to_sensivity = dataset.phy_names.index(sensivity)
            print(column_to_sensivity)
            pp = apply_sensitivity(pp, column = column_to_sensivity, a_percentage=30)
            mu_ = reg.process_regressors(reg_conf_file, phys2=dataset.phy_names, samples= pp, 
                                        assess_regressor=False, train_rf=False)
            mu_ = torch.from_numpy(mu_).to(device)
        data = torch.from_numpy(data).to(device)
        onehot = torch.from_numpy(onehot).to(device)
        pp = torch.from_numpy(pp).to(device)

    print('device: ', device)
    print('torch assigned')
    print('labels: ', config['label_dim'])
    print('physics dim: ', config['physics_dim'])
    
    print('VAE with  {} labels and {} pp '.format( config['label_dim'],config['physics_dim']))
    if config['label_dim'] > 0 and config['physics_dim'] > 0:
        if sensivity is None:
            _, mu_, _, _ = vae(data, label=onehot, phy=pp)
        xhat_mu = vae.decoder(mu_, data[:,:,0], label=onehot, phy=pp)
    elif config['label_dim'] > 0 and config['physics_dim'] == 0:
        _, mu_, _, _ = vae(data, label=onehot)
        xhat_mu = vae.decoder(mu_, data[:,:,0], label=onehot)
    elif config['label_dim'] == 0:
        _, mu_, _, _ = vae(data)
        xhat_mu = vae.decoder(mu_, data[:,:,0])
    else:
        print('Check conditional dimension...')

    print('z hat extracted')
    xhat_mu = torch.cat([data[:,:,0].unsqueeze(-1), xhat_mu], dim=-1).cpu().detach().numpy()
    
    #if plot:
    #    xhat_mu2 = torch.cat([data[:,:,0].unsqueeze(-1), xhat_mu2], dim=-1).cpu().detach().numpy()

    data = data.cpu().detach().numpy()

    if plot:
        plot_wall_lcs(xhat_mu, data, cls=lb, save=save_plot, to_title = pp[column_to_sensivity], sensivity = sensivity) #data is real_lc

        #plot_wall_lcs(xhat_mu, xhat_mu2, cls=lb, save=save_plot) #data is real_lc

        #plot_wall_synthetic_lcs(xhat_mu,  cls=lb,  save=save_plot)

    #lc_reverted = revert_light_curve(sample_period, xhat_mu, classes = lb)

    #print('lc_reverted: ', lc_reverted.shape)
    #print('xhat_mu: ', xhat_mu.shape)

    #if plot_reverted: 
    #    compare_folded_crude_lc(xhat_mu, lc_reverted, cls=lb, period=sample_period)

    #save_arrays_to_folder(lc_reverted, lb, PATH_DATA)

if __name__ == "__main__":
    main()