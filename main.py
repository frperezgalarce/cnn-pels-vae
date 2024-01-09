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
         train_regressor: bool = True, wandb_active = True, prior=True) -> None:
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
        cnn.run_cnn(create_samples, mean_prior_dict=mean_prior_dict, 
                    vae_model=vae_model, PP=PP_list, 
                    wandb_active = wandb_active, prior=prior)
    
if __name__ == "__main__": 

    wandb_active = True
    if wandb_active: 
        sweep_config = {
            'method': 'bayes',
            'metric': {'goal': 'maximize', 'name': 'weighted_f1'},
            'parameters': {
                'learning_rate': {'min': 0.002, 'max': 0.003},
                'batch_size': {'values': [32]},
                'patience':{'min': 20, 'max': 40},
                'repetitions': {'values': [1, 2]},
                'sinthetic_samples_by_class': {'values': [8,16,32]},
                'threshold_acc_synthetic': {'min': 0.75, 'max': 0.90},
                'beta_decay_factor': {'min': 0.96, 'max': 0.98}, 
                'EPS': {'min': 0.2, 'max': 0.3},
                'scaling_factor': {'min': 0.3, 'max': 0.5}, 
                'vae_model': {'values': ['1ojzq1t5', '1qfpknvn', '3iyiphkn', 'gn42liaz']}, 
                'sufix_path': {'values': ['GAIA3_LOG_IMPUTED_BY_CLASS_6PP', 'GAIA3_6PP', 'GAIA3_LOG_6PP']}, 
                'layers': {'values': [2, 3, 4]},
                'loss': {'values': ['focalLoss']},
                'alpha': {'min': 0.2, 'max': 0.4},
                'focal_loss_scale': {'min': 1.5, 'max': 4.0},
                'n_oversampling': {'min': 1, 'max': 32},
                'ranking_method': {'values': ['CCR', 'max_pairwise_confusion', 'max_confusion']},
            }
        }
        

        with open("sweep.yaml", "w") as sweep_file:
            yaml.safe_dump(sweep_config, sweep_file)
        sweep_id = wandb.sweep(sweep_config, project="train-classsifier")
        wandb.agent(sweep_id, function=main, count=100, project="train-classsifier")
    else: 
        main(train_gmm = True, create_samples = True, 
            train_classifier = True, sensitive_test= False, train_regressor=True,
             wandb_active = wandb_active, prior=True)
    # create_samples activate samples generation in cnn training
    

    