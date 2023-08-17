import src.cnn as cnn
import src.gmm.bgmm as bgmm 
import src.sampler.fit_regressor as reg
import src.sampler.create_lc as creator
import src.gmm.modifiedgmm as mgmm
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

def main(train_gmm: Optional[bool] = False, create_samples: Optional[bool] = False) -> None:
    if train_gmm: 
        bgmm.train_and_save()

    if create_samples:
        print(len(mean_prior_dict['StarTypes'][CLASSES[0]].keys())-1)
        components: int = 3 # len(mean_prior_dict['StarTypes'][CLASSES[0]].keys())-1 TODO: check number of components
        sampler: mgmm.ModifiedGaussianSampler = mgmm.ModifiedGaussianSampler(b=0.5, components=components)
        model_name: str = PATH_MODELS+'bgm_model_'+str(CLASSES[0])+'.pkl'
        samples: np.ndarray = sampler.modify_and_sample(model_name)
        z_hat: Any = reg.main(samples)
        samples, z_hat = None, None
        creator.main(samples, z_hat) #TODO: check error

    cnn.run_cnn(create_samples, mode_running='load')
    
if __name__ == "__main__":
    main()
