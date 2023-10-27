import src.cnn as cnn
import src.gmm.bgmm as bgmm 
import src.sampler.create_lc as creator
import src.sampler.fit_regressor as reg
from src.utils import load_yaml_priors, load_pp_list, load_id_period_to_sample
import torch
import argparse
import yaml
from typing import List, Optional, Any, Dict
import wandb

with open('src/paths.yaml', 'r') as file:
    YAML_FILE: Dict[str, Any] = yaml.safe_load(file)

PATHS: Dict[str, str] = YAML_FILE['paths']
PATH_PRIOS: str = PATHS['PATH_PRIOS']
PATH_MODELS: str = PATHS['PATH_MODELS']
mean_prior_dict: Dict[str, Any] = load_yaml_priors(PATH_PRIOS)

with open('src/regressor.yaml', 'r') as file:
    config_file: Dict[str, Any] = yaml.safe_load(file)


print('#'*50)
print('SETUP')
print('#'*50)




def main(train_gmm: Optional[bool] = True, create_samples: Optional[bool] = True, 
         train_classifier: Optional[bool]=True, sensitive_test: bool = False, 
         train_regressor: bool = True, wandb_active = True) -> None:
    torch.cuda.empty_cache()

    if wandb_active:
        wandb.init(project='train-classsifier', entity='fjperez10')
        torch.cuda.empty_cache()
        vae_model = wandb.config.vae_model
        sufix_path = wandb.config.sufix_path
    else: 
        vae_model: str =   config_file['model_parameters']['ID']   # '1pjeearx'#'20twxmei' trained using TPM using GAIA3 ... using 5 PP 1pjeearx
        print('Using vae model: '+ vae_model)
        sufix_path: str =   config_file['model_parameters']['sufix_path']
        print('sufix path: '+ sufix_path)

    PP_list = load_pp_list(vae_model)
    print('FEATURES: ', PP_list)
    if train_regressor:
        print('training regressor using :', PP_list)
        reg.apply_regression(vae_model, from_vae= True, train_rf= True, phys2 = PP_list)

    if sensitive_test:
        print('conducting sensitive test')
        for pp in PP_list:
            objects_by_class = {'ACEP':24}#, 'CEP': 24,  'DSCT': 24,  'ECL':24,  'ELL': 24,  'LPV': 24,  'RRLYR':  24,  'T2CEP':24}
            for key, value in objects_by_class.items():
                temp_dict = {key: value}
                creator.get_synthetic_light_curves(None, None, training_cnn=False, plot=False, save_plot=True,
                                                objects_by_class=temp_dict, sensivity=pp,
                                                column_to_sensivity=pp)
                del temp_dict
    
    if train_gmm:
        print('Fitting Gaussian mixture models') 
        bgmm.fit_gausians(mean_prior_dict, columns= ['Type','Period', 'teff_val', '[Fe/H]_J95', 'abs_Gmag', 'radius_val', 'logg'])
        print('Gaussian were fitted')

    if train_classifier: 
        cnn.run_cnn(create_samples, mean_prior_dict=mean_prior_dict, vae_model=vae_model, PP=PP_list)
    
if __name__ == "__main__": 

    wandb_active = True
    if wandb_active: 
        sweep_config = {
            'method': 'bayes',
            'metric': {'goal': 'maximize', 'name': 'f1_val'},
            'parameters': {
                'learning_rate': {'min': 0.001, 'max': 0.01},
                'batch_size': {'values': [32, 64, 128]},
                'patience':{'values': [20, 30, 50, 100]},
                'repetitions': {'values': [1, 3, 5, 10, 20]},
                'sinthetic_samples_by_class': {'values': [8, 16, 32]},
                'threshold_acc_synthetic': {'min': 0.85, 'max': 0.95},
                'beta_decay_factor': {'min': 0.9, 'max': 0.99}, 
                'EPS': {'min': 0.01, 'max': 0.05},
                'scaling_factor': {'min': 0.1, 'max': 1.0}, 
                'vae_model': {'values': ['1ojzq1t5','16f09v2s', '1j9gn236', '2b0tvacd', '2uioeni3',
                                        '3iyiphkn', '3pbpvynz', '16f09v2s', '22my5dmi', '39snao1w', 
                                        'gn42liaz', 'hu69iv0r']}, #iqf fail
                'sufix_path': {'values': ['GAIA3_6PP', 'GAIA3_IMPUTED_6PP','GAIA3_LOG_6PP','GAIA3_LOG_IMPUTED_6PP', 'GAIA3_LOG_IMPUTED_BY_CLASS_6PP']}, 
                'layers': {'values': [2, 3, 4]},
                'loss': {'values': ['CrossEntropyLoss']},
            }
        }
        with open("sweep.yaml", "w") as sweep_file:
            yaml.safe_dump(sweep_config, sweep_file)
        sweep_id = wandb.sweep(sweep_config, project="train-classsifier")
        wandb.agent(sweep_id, function=main)
    else: 
        main(train_gmm = True, create_samples = False, 
            train_classifier = False, sensitive_test= False, train_regressor=False)
    # create_samples activate samples generation in cnn training
    

    