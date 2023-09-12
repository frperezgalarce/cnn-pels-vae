import src.cnn as cnn
import src.gmm.bgmm as bgmm 
import src.sampler.create_lc as creator
import src.sampler.fit_regressor as reg
from src.utils import load_yaml_priors, load_pp_list

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


print('#'*50)
print('SETUP')
print('#'*50)

vae_model: str =   config_file['model_parameters']['ID']   # '1pjeearx'#'20twxmei' trained using TPM using GAIA3 ... using 5 PP 1pjeearx
print('Using vae model: '+ vae_model)
sufix_path: str =   config_file['model_parameters']['sufix_path']
print('sufix path: '+ sufix_path)

def main(train_gmm: Optional[bool] = False, create_samples: Optional[bool] = True, 
         train_classifier: Optional[bool]=True, sensitive_test: bool = False, 
         train_regressor: bool = False) -> None:
    
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
        #TODO: adapt to consider different lenth of features
        bgmm.fit_gausians(mean_prior_dict, columns=PP_list+['Type'])
        print('Gaussian were fitted')

    if train_classifier: 
        #TODO  create a dict to link model with set of parameters
        cnn.run_cnn(create_samples, mean_prior_dict=mean_prior_dict, 
                    mode_running='load', vae_model=vae_model, PP=PP_list)
    
if __name__ == "__main__": 
    main(train_gmm = False, create_samples = True, 
         train_classifier = True, sensitive_test= False, train_regressor=False)
        # create_samples activate samples generation in cnn training
    