import src.cnn as cnn
import src.gmm.bgmm as bgmm 
import src.sampler.create_lc as creator
import src.sampler.fit_regressor as reg
from src.utils import load_yaml_priors, load_pp_list, load_id_period_to_sample
import torch
import argparse
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


def config_wandb(): 
    ## Config ##
    parser = argparse.ArgumentParser(description='Masked CNN to mitigate biases using synthetic samples')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                        default=False,
                        help='Load data and initialize models [False]')

    parser.add_argument('--machine', dest='machine', type=str, default='karimsala6s',
                        help='were to is running (Jorges-MBP, karimsala6s, [exalearn])')

    parser.add_argument('--data', dest='data', type=str, default='OGLE3_lcs_I_meta_snr5_augmented_folded_trim600_GAIA3_6PP.npy',
                        help='data used for training ([OGLE3], EROS2)')
    parser.add_argument('--use-err', dest='use_err', type=str, default='T',
                        help='use magnitude errors ([T],F)')
    parser.add_argument('--cls', dest='cls', type=str, default='all',
                        help='drop or select ony one class '+
                        '([all],drop_"vartype",only_"vartype")')

    parser.add_argument('--lr', dest='lr', type=float, default=1e-3,
                        help='learning rate [1e-4]')
    parser.add_argument('--lr-sch', dest='lr_sch', type=str, default='cos',
                        help='learning rate shceduler '+
                        '([None], step, exp,cosine, plateau)')
    parser.add_argument('--beta', dest='beta', type=str, default='4',
                        help='beta factor for latent KL div ([1],step)')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=128,
                        help='batch size [128]')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=200,
                        help='total number of training epochs [150]') 

    parser.add_argument('--cond', dest='cond', type=str, default='T',
                        help='label conditional VAE (F,[T])')
    parser.add_argument('--phy', dest='phy', type=str, default='PTARMG',
                        help='physical parameters to use for conditioning ([],[tm])')
    parser.add_argument('--latent-dim', dest='latent_dim', type=int, default=6,
                        help='dimension of latent space [6]')
    parser.add_argument('--latent-mode', dest='latent_mode', type=str,
                        default='repeat',
                        help='wheather to sample from a 3d or 2d tensor '+
                        '([repeat],linear,convt)')
    parser.add_argument('--arch', dest='arch', type=str, default='tcn',
                        help='architecture for Enc & Dec ([tcn],lstm,gru)')
    parser.add_argument('--transpose', dest='transpose', type=str, default='F',
                        help='use tranpose convolution in Dec ([F],T)')
    parser.add_argument('--units', dest='units', type=int, default=48,
                        help='number of hidden units [32]')
    parser.add_argument('--layers', dest='layers', type=int, default=9,
                        help='number of layers/levels for lstm/tcn [5]')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.2,
                        help='dropout for lstm/tcn layers [0.2]')
    parser.add_argument('--kernel-size', dest='kernel_size', type=int, default=4,
                        help='kernel size for tcn conv, use odd ints [5]')

    parser.add_argument('--comment', dest='comment', type=str, default='GAIA3_LOG_IMPUTED_BY_CLASS_6PP - log_cosh_loss',
                        help='extra comments')
    args = parser.parse_args()


def main(train_gmm: Optional[bool] = False, create_samples: Optional[bool] = True, 
         train_classifier: Optional[bool]=True, sensitive_test: bool = False, 
         train_regressor: bool = False) -> None:
    torch.cuda.empty_cache()

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
        bgmm.fit_gausians(mean_prior_dict, columns= ['Type','Period', 'teff_val', '[Fe/H]_J95', 'abs_Gmag', 'radius_val', 'logg'])
        print('Gaussian were fitted')

    if train_classifier: 
        cnn.run_cnn(create_samples, mean_prior_dict=mean_prior_dict, vae_model=vae_model, PP=PP_list, opt_method='twolosses')
    
if __name__ == "__main__": 
    main(train_gmm = False, create_samples = True, 
         train_classifier = True, sensitive_test= False, train_regressor=False)
        # create_samples activate samples generation in cnn training
    