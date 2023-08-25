import src.cnn as cnn
import src.gmm.bgmm as bgmm 
import src.sampler.create_lc as creator
from src.utils import load_yaml_priors
import yaml
from typing import List, Optional, Any, Dict


with open('src/paths.yaml', 'r') as file:
    YAML_FILE: Dict[str, Any] = yaml.safe_load(file)

PATHS: Dict[str, str] = YAML_FILE['paths']
PATH_PRIOS: str = PATHS['PATH_PRIOS']
PATH_MODELS: str = PATHS['PATH_MODELS']
mean_prior_dict: Dict[str, Any] = load_yaml_priors(PATH_PRIOS)

with open('src/regressor.yaml', 'r') as file:
    config_file: Dict[str, Any] = yaml.safe_load(file)

vae_model: str =   config_file['model_parameters']['ID']   # '1pjeearx'#'20twxmei' trained using TPM using GAIA3 ... using 5 PP 1pjeearx
print('Using vae model: '+ vae_model)
#TODO: read from file configuration the PP number
PP_list_6 = ['[Fe/H]_J95', 'abs_Gmag', 'teff_val', 'Period','radius_val', 'logg']
PP_list_5 = ['[Fe/H]_J95', 'abs_Gmag', 'teff_val', 'Period','radius_val']
PP_list_3 = ['abs_Gmag', 'teff_val', 'Period']


def main(train_gmm: Optional[bool] = False, create_samples: Optional[bool] = True, train_classifier: Optional[bool]=True) -> None:
    
    objects_by_class = {'ACEP':24, 'CEP': 24,  'DSCT': 24,  'ECL':24,  'ELL': 24,  'LPV': 24,  'RRLYR':  24,  'T2CEP':24}
    for key, value in objects_by_class.items():
        temp_dict = {key: value}
        print(temp_dict)
        creator.main(None, None, training_cnn=False, plot=True, save_plot=True, objects_by_class=temp_dict)
        del temp_dict
    raise

    if train_gmm:
        print('Fitting Gaussian mixture models') 
        bgmm.fit_gausians(mean_prior_dict, columns=PP_list_6)
        print('Gaussian were fitted')
    if train_classifier: 
        #create a dict to link model with set of parameters
        cnn.run_cnn(create_samples, mean_prior_dict=mean_prior_dict, 
                    mode_running='load', vae_model=vae_model, PP=PP_list_6)
    
if __name__ == "__main__":
    main()
