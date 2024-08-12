
import wandb
import yaml 
import torch 

def save_metrics(wandb_active, opt_method, results_dict,
                 weight_f1_score_hyperparameter_search, 
                 accuracy_test, f1_test, roc_test_ovo, roc_test_ovr):
    """
    Save the metrics to the Weights and Biases (wandb) service if active, and compute the weighted F1 score.

    :param wandb_active: A boolean flag to determine if wandb logging should be activated.
    :param opt_method: A string representing the optimization method used, affects weighted F1 calculation.
    :param results_dict: A dictionary containing the results metrics to be logged, including F1 scores.
    :param weight_f1_score_hyperparameter_search: A float representing the weight for synthetic F1 score in the combined metric.
    :param accuracy_test: A float representing the test set accuracy.
    :param f1_test: A float representing the test set F1 score.
    """
    if wandb_active:
        # Update wandb configuration for test metrics
        wandb.config.acc_test = accuracy_test
        wandb.config.f1_test = f1_test
        wandb.config.roc_test_ovo = roc_test_ovo
        wandb.config.roc_test_ovr = roc_test_ovr

        # Calculate weighted F1 score based on the optimization method
        if opt_method == 'twolosses':
            weighted_f1 = (results_dict['f1_synthetic'] * weight_f1_score_hyperparameter_search +
                           results_dict['f1_val'] * (1 - weight_f1_score_hyperparameter_search))
        else:
            weighted_f1 = results_dict['f1_val']

        # Update results dictionary with the calculated weighted F1 score
        results_dict['weighted_f1'] = weighted_f1

        # Log the results to wandb
        wandb.log(results_dict)

def setup_hyper_opt(main, nn_config): 
    """
    Set up hyperparameter optimization using Weight & Biases sweeps based on neural network configuration.

    :param main: The main function to be optimized, which will be called by wandb agent.
    :param nn_config: A dictionary containing the neural network configuration, including optimization method.
    """

    if nn_config['training']['opt_method'] == 'twolosses':
        # Configuration for the 'twolosses' optimization method
        sweep_config = {
            'method': 'grid',
            #'name':'test_roc',
            'name': f"exp_s_{nn_config['data']['sample_size']}_l_{nn_config['data']['seq_length']}_sn_{nn_config['data']['sn_ratio']}- test new biases",
            'metric': {'goal': 'maximize', 'name': 'weighted_f1'},
            'parameters': {
                'learning_rate': {'values': [0.09]},
                'batch_size': {'values': [64]},
                'patience': {'values': [7]},
                'repetitions': {'values': [11]},
                'synthetic_samples_by_class':  {'values': [32]},  
                'threshold_acc_synthetic': {'values': [0.88]},
                'beta_decay_factor': {'values': [1]},
                'EPS': {'values': [0.35]},  
                'scaling_factor':{'values': [1.32]},
                'vae_model': {'values': ['gn42liaz']},
                'sufix_path': {'values': ['GAIA3_LOG_IMPUTED_BY_CLASS_6PP']},
                'layers': {'values': [4]},
                'loss': {'values': ['focalLoss']}, 
                'focal_loss_scale': {'values': [2]},  
                'n_oversampling': {'values': [16]},
                'decay_parameter_1': {'values': [0.69]},
                'decay_parameter_2': {'values': [0.62]},
                'ranking_method': {'values': ['proportion']},
                'iteration':{'values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
            }
        }
    elif nn_config['training']['opt_method'] == 'oneloss':
        # Configuration for the 'oneloss' optimization method
        sweep_config = {
            'method': 'grid',
            #'name':'test_roc',
            'name': f"exp_s_{nn_config['data']['sample_size']}_l_{nn_config['data']['seq_length']}_sn_{nn_config['data']['sn_ratio']}_oneloss - test new biases",
            'metric': {'goal': 'maximize', 'name': 'weighted_f1'},
            'parameters': {
                'learning_rate': {'values': [0.09]},
                'batch_size': {'values': [64]},
                'patience': {'values': [7]},
                'repetitions': {'values': [11]},
                'synthetic_samples_by_class':  {'values': [32]},  
                'threshold_acc_synthetic': {'values': [0.88]},
                'beta_decay_factor': {'values': [1]},
                'EPS': {'values': [0.35]},  
                'scaling_factor':{'values': [1.32]},
                'vae_model': {'values': ['gn42liaz']},
                'sufix_path': {'values': ['GAIA3_LOG_IMPUTED_BY_CLASS_6PP']},
                'layers': {'values': [4]},
                'loss': {'values': ['focalLoss']}, 
                'focal_loss_scale': {'values': [2]},  
                'n_oversampling': {'values': [16]},
                'decay_parameter_1': {'values': [0.69]},
                'decay_parameter_2': {'values': [0.62]},
                'ranking_method': {'values': ['proportion']},
                'iteration':{'values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
            }
        }
    else:
        raise ValueError('This opt_method is unavailable, please select: oneloss or twolosses')

    # Write the sweep configuration to a YAML file
    with open("sweep.yaml", "w") as sweep_file:
        yaml.safe_dump(sweep_config, sweep_file)

    # Initialize the sweep and run the agent
    sweep_id = wandb.sweep(sweep_config, project="train-classsifier")
    wandb.agent(sweep_id, function=main, project="train-classsifier")

def set_cvae(wandb_active, config_file):
    """
    Initialize the CVAE model by setting its model ID and suffix path based on Weights & Biases configuration or a local config file.

    :param wandb_active: A boolean indicating whether to use Weights & Biases for configuration.
    :param config_file: A configuration file to use if Weights & Biases is not active, containing model parameters.
    :return: A tuple containing the VAE model ID and the suffix path for the model.
    """
    if wandb_active:
        # Initialize wandb and set parameters based on its configuration
        wandb.init(project='train-classsifier', entity='fjperez10')
        torch.cuda.empty_cache()
        vae_model = wandb.config.vae_model
        sufix_path = wandb.config.sufix_path
    else:
        # Set parameters based on the local configuration file
        vae_model = config_file['model_parameters']['ID']
        sufix_path = config_file['model_parameters']['sufix_path']
        print(f'Using VAE model: {vae_model}')
        print(f'Suffix path: {sufix_path}')

    return vae_model, sufix_path

def setup_gradients(wandb_active, model):
    """
    Setup gradient logging for the given model using Weights & Biases if wandb is active.

    :param wandb_active: A boolean indicating whether Weights & Biases logging is active.
    :param model: The model for which to log the gradients.
    """
    if wandb_active:
        # Start watching the model on Weights & Biases with gradient logging
        wandb.watch(model, log='gradients', log_freq=10)
        print("Gradient logging is enabled in Weights & Biases.")
    else:
        print("Weights & Biases logging is not active. No gradient logging will be performed.")

def cnn_hyperparameters(wandb_active, hyperparam_opt, nn_config, config_file):
    """
    Configure CNN hyperparameters and integrate with Weights & Biases for logging and optimization.

    :param wandb_active: A boolean indicating if wandb logging is active.
    :param hyperparam_opt: A boolean indicating if hyperparameter optimization is active.
    :param nn_config: A dictionary containing the neural network training configuration.
    :param config_file: A dictionary containing model parameters configuration.
    :return: A tuple of updated nn_config and config_file dictionaries.
    """
    if wandb_active:
        if hyperparam_opt:
            nn_config['training']['base_learning_rate'] = wandb.config.learning_rate
            nn_config['training']['batch_size'] = wandb.config.batch_size
            nn_config['training']['patience'] =  wandb.config.patience
            nn_config['training']['repetitions'] =  wandb.config.repetitions
            nn_config['training']['synthetic_samples_by_class'] = wandb.config.synthetic_samples_by_class
            nn_config['training']['threshold_acc_synthetic'] = wandb.config.threshold_acc_synthetic
            nn_config['training']['EPS'] = wandb.config.EPS
            nn_config['training']['scaling_factor'] = wandb.config.scaling_factor
            vae_model = wandb.config.vae_model
            sufix_path = wandb.config.sufix_path            
            nn_config['training']['layers'] = wandb.config.layers
            nn_config['training']['loss'] = wandb.config.loss
            config_file['model_parameters']['ID'] =  wandb.config.vae_model
            config_file['model_parameters']['sufix_path'] = wandb.config.sufix_path
            nn_config['training']['focal_loss_scale'] = wandb.config.focal_loss_scale
            nn_config['training']['n_oversampling'] = wandb.config.n_oversampling
            nn_config['training']['ranking_method'] = wandb.config.ranking_method
            nn_config['training']['decay_parameter_1'] = wandb.config.decay_parameter_1
            nn_config['training']['decay_parameter_2'] = wandb.config.decay_parameter_2
            

            with open('src/configuration/regressor.yaml', 'w') as file:
                yaml.dump(config_file, file)

        wandb.config.epochs = nn_config['training']['epochs']
        wandb.config.patience = nn_config['training']['patience']
        wandb.config.batch_size = nn_config['training']['batch_size']
        wandb.config.repetitions = nn_config['training']['repetitions']
        wandb.config.synthetic_samples_by_class = nn_config['training']['synthetic_samples_by_class']
        wandb.config.threshold_acc_synthetic = nn_config['training']['threshold_acc_synthetic']
        #wandb.config.beta_decay_factor = nn_config['training']['beta_decay_factor']
        wandb.config.beta_initial = nn_config['training']['beta_initial']
        wandb.config.EPS = nn_config['training']['EPS']
        wandb.config.base_learning_rate = nn_config['training']['base_learning_rate']
        wandb.config.scaling_factor = nn_config['training']['scaling_factor']
        wandb.config.opt_method = nn_config['training']['opt_method']
        wandb.config.vae_model = vae_model
        wandb.config.sufix_path = sufix_path
        wandb.config.mode_running =  nn_config['data']['mode_running']
        wandb.config.sample_size =  nn_config['data']['sample_size']
        wandb.config.seq_length =  nn_config['data']['seq_length']
        wandb.config.sn_ratio =  nn_config['data']['sn_ratio']
        wandb.config.upper_limit_majority_classes =  nn_config['data']['upper_limit_majority_classes']
        wandb.config.limit_to_define_minority_Class =  nn_config['data']['limit_to_define_minority_Class']
        wandb.config.layers =  nn_config['training']['layers']
        wandb.config.loss =  nn_config['training']['loss']
        wandb.config.focal_loss_scale = nn_config['training']['focal_loss_scale']
        wandb.config.n_oversampling = nn_config['training']['n_oversampling']
        wandb.config.ranking_method = nn_config['training']['ranking_method']

    return nn_config, config_file