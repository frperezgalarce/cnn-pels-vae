"""
This module orchestrates the entire data processing and model training pipeline, including
training Gaussian Mixture Models (GMM), classifiers, and regressors, as well as creating
samples and conducting sensitivity testing.

Functions:
    main(train_gmm: Optional[bool], create_samples: Optional[bool], train_classifier: Optional[bool],
         sensitive_test: Optional[bool], train_regressor: Optional[bool], wandb_active: Optional[bool],
         prior: Optional[bool]) -> None:
        The main entry function that coordinates the execution of different phases of the data processing
        and model training pipeline based on the provided flags.

The module leverages various components of the system such as VAE models, GMM, and classification,
regression models to conduct a comprehensive analysis and training process. Integration with Weights &
Biases (wandb) is also provided for hyperparameter optimization and tracking experiments.

Usage:
    The module is designed to be run as a script. It reads configuration from YAML files, prepares the
    data and models, and then executes training, evaluation, and testing processes as specified. 
    It supports flexible execution controlled by command-line arguments or through configuration files
    for different parts of the process like GMM fitting, sample generation, and model training.
"""

from typing import Optional
import torch
from tqdm import tqdm
import yaml
import src.cnn.cnn as cnn
import src.gmm.bgmm as bgmm
import src.sampler.fit_regressor as reg
from src.utils import load_pp_list, pp_sensitive_test, load_metadata
import src.wandb_setup as wsetup
import time 

def main(train_gmm: Optional[bool] = True, create_samples: Optional[bool] = True,
         train_classifier: Optional[bool] = True, sensitive_test: Optional[bool] = False,
         train_regressor: Optional[bool] = True, wandb_active: Optional[bool] = True,
         prior: Optional[bool] = True) -> None:
    """
    Main function to run the data processing and model training pipeline.

    :param train_gmm: Flag to train Gaussian Mixture Models (GMM).
    :param create_samples: Flag to create samples (used in classifier training).
    :param train_classifier: Flag to train the classifier.
    :param sensitive_test: Flag to perform sensitivity testing.
    :param train_regressor: Flag to train the regressor.
    :param wandb_active: Flag to activate Weights & Biases integration.
    :param prior: Flag to use prior knowledge in training.
    """

    # Clearing the GPU cache to ensure maximum available memory
    torch.cuda.empty_cache()

    # Loading metadata and configuration settings
    _, config_file = load_metadata()

    # Setting up the variational autoencoder (VAE) model using the configuration
    vae_model, _ = wsetup.set_cvae(wandb_active, config_file)

    # Loading physical parameter (pp) list used in the training process
    pp_list = load_pp_list(vae_model)

    # Training the regressor if enabled
    if train_regressor:
        reg.apply_regression(vae_model, from_vae=True, train_rf=True, phys2=pp_list)

    # Running sensitivity testing if enabled
    if sensitive_test:
        pp_sensitive_test(pp_list)

    # Training the Gaussian Mixture Model if enabled
    if train_gmm:
        bgmm.fit_gaussians(prior, columns=['Type', 'Period', 'teff_val', '[Fe/H]_J95',
                                           'abs_Gmag', 'radius_val', 'logg'])

    # Training the classifier if enabled
    if train_classifier:
        cnn.run_cnn(create_samples, vae_model=vae_model,
                    pp=pp_list, wandb_active=wandb_active,
                    prior=True)

# Entry point of the script
if __name__ == "__main__":

    # Flag to control the activation of Weights & Biases integration
    wandb_active = True
    #method = "twolosses"

    with open('src/configuration/nn_config.yaml', 'r') as file:
        nn_config = yaml.safe_load(file)

    # Setup hyperparameter optimization if Weights & Biases is active
    if wandb_active:
        sample_sizes = [400000]
        sn_ratios = [6]
        seq_lengths = [300]

        # Create a total progress bar for all iterations
        total_iterations = len(sample_sizes) * len(sn_ratios) * len(seq_lengths)
        with tqdm(total=total_iterations) as pbar:
            for sample_size in sample_sizes: 
                for sn_ratio in sn_ratios:
                    for seq_length in seq_lengths:
                        nn_config['data']['mode_running'] = "create"
                        for method in ['twolosses', 'oneloss']:#,
                            # Clearing the GPU cache to ensure maximum available memory
                            torch.cuda.empty_cache()
                            nn_config['data']['sample_size'] = sample_size
                            nn_config['data']['sn_ratio'] = sn_ratio
                            nn_config['data']['seq_length'] = seq_length
                            nn_config['seq_length'] = seq_length
                            nn_config['data']['opt_method'] = method
                            nn_config['opt_method'] = method
                            nn_config['training']['opt_method'] = method
                            try:
                                with open('src/configuration/nn_config.yaml', 'w') as file:
                                    yaml.dump(nn_config, file)
                                    file.flush()  # Force flushing the file buffer to disk
                            except Exception as e:
                                print(f"Error writing to file: {e}")

                            print(nn_config)
                            time.sleep(2)
                            wsetup.setup_hyper_opt(main, nn_config)
                            pbar.update(1)

    else:
        for sample_size in [10000, 20000, 40000, 80000, 160000, 320000]:
            for ranking_method in ['CCR', 'max_confusion', 'proportion', 'max_pairwise_confusion', 'no_priority']:
                nn_config['data']['sample_size'] = sample_size
                with open('src/configuration/nn_config.yaml', 'w') as file:
                        yaml.dump(nn_config, file)

                # Directly run the main function with specified configurations if W&B is not active
                main(train_gmm=True, create_samples=True,
                    train_classifier=True, sensitive_test=False,
                    train_regressor=True, wandb_active=wandb_active,
                    prior=True)