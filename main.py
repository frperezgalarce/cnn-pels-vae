import src.cnn as cnn
import src.gmm.bgmm as bgmm 
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
PP_list_6 = ['Type','[Fe/H]_J95', 'abs_Gmag', 'teff_val', 'Period','radius_val', 'logg']
PP_list_5 = ['Type','[Fe/H]_J95', 'abs_Gmag', 'teff_val', 'Period','radius_val']
PP_list_3 = ['Type','abs_Gmag', 'teff_val', 'Period']


def main(train_gmm: Optional[bool] = True, create_samples: Optional[bool] = True, train_classifier: Optional[bool]=True) -> None:
    if train_gmm:
        print('Fitting Gaussian mixture models') 
        bgmm.fit_gausians(mean_prior_dict, columns=PP_list_3)
        print('Gaussian were fitted')
    if train_classifier: 
        #create a dict to link model with set of parameters
        cnn.run_cnn(create_samples, mean_prior_dict=mean_prior_dict, 
                    mode_running='load', vae_model=vae_model, PP=PP_list_3)
    
if __name__ == "__main__":
    main()
