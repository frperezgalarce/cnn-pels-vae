import src.cnn as cnn
import src.gmm.bgmm as bgmm 
from src.utils import load_yaml_priors
import yaml
from typing import List, Optional, Any, Dict

CLASSES: List[str] = ['CEP']

with open('src/paths.yaml', 'r') as file:
    YAML_FILE: Dict[str, Any] = yaml.safe_load(file)

PATHS: Dict[str, str] = YAML_FILE['paths']
PATH_PRIOS: str = PATHS['PATH_PRIOS']
PATH_MODELS: str = PATHS['PATH_MODELS']
mean_prior_dict: Dict[str, Any] = load_yaml_priors(PATH_PRIOS)
vae_model: str = '1pjeearx'#'20twxmei' trained using TPM using GAIA3 ... using 5 PP 1pjeearx

def main(train_gmm: Optional[bool] = False, create_samples: Optional[bool] = True, train_classifier: Optional[bool]=False) -> None:
    if train_gmm: 
        bgmm.fit_gausians(mean_prior_dict)

    if train_classifier: 
        cnn.run_cnn(create_samples, mode_running='load')
    
if __name__ == "__main__":
    main()
