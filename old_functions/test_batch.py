from src.sampler.getbatch import SyntheticDataBatcher
from src.visualization import plot_batch 
import yaml 
from typing import List, Optional, Any, Dict
from src.utils import load_pp_list
import pickle

with open('src/configuration/regressor.yaml', 'r') as file:
    config_file: Dict[str, Any] = yaml.safe_load(file)
with open('src/configuration/paths.yaml', 'r') as file:
    YAML_FILE: Dict[str, Any] = yaml.safe_load(file)

PATHS: Dict[str, str] = YAML_FILE['paths']
PATH_MODELS: str = PATHS['PATH_MODELS']

vae_model: str =   config_file['model_parameters']['ID']   # '1pjeearx'#'20twxmei' trained using TPM using GAIA3 ... using 5 PP 1pjeearx

print('Using vae model: '+ vae_model)

sufix_path: str =   config_file['model_parameters']['sufix_path']

print('sufix path: '+ sufix_path)

synthetic_samples_by_class = 8

PP_list = load_pp_list(vae_model)

seq_length = 300

batcher = SyntheticDataBatcher(PP = PP_list, vae_model=vae_model, n_samples=synthetic_samples_by_class, 
                                    seq_length = seq_length)

synthetic_data_loader = batcher.create_synthetic_batch(b=1)

print(batcher.x_array)

with open(PATH_MODELS+'label_encoder_vae.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

plot_batch(batcher.x_array, batcher.y_array, label_encoder)