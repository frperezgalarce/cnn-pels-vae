'''This module creates and run the 1D-cnn classifier'''
from typing import Any
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import yaml
from torch.nn import init
import wandb
from src.sampler.getbatch import SyntheticDataBatcher
from src.cnn.focalloss import FocalLossMultiClass as focal_loss
import src.utils as utils
import src.wandb_setup as wset
import time
from src.cnn.training_cnn import (get_dict_class_priorization,
                                 train_one_epoch_alternative, create_dataloader,
                                 evaluate_dataloader,
                                 initialize_optimizers)

gpu = True
device = torch.device("cuda:0" if torch.cuda.is_available()
                                    and gpu else "cpu")

class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) for processing light curves.
    
    Attributes:
        layers (int): Number of convolutional layers in the network.
        conv1, conv2, ..., conv4 (nn.Conv1d): Convolutional layers of the network.
        bn1, bn2, ..., bn4 (nn.BatchNorm1d): Batch normalization layers.
        pool1, pool2, ..., pool4 (nn.MaxPool1d): Pooling layers to reduce spatial dimensions.
        fc1 (nn.Linear): Fully connected layer to map features to intermediate representation.
        fc2 (nn.Linear): Final fully connected layer to map intermediate representation to class scores.
    
    Parameters:
        num_classes (int): Number of classes in the output prediction. Default is 2.
        layers (int): Number of convolutional layers to use (2 to 4). Default is 2.
        kernel_size (int): Size of the convolutional kernel. Default is 6.
        stride (int): Stride of the convolution operation. Default is 1.
    
    Methods:
        forward(x): Defines the forward pass of the CNN.
    """

    def __init__(self, seq_length = 300, num_classes: int = 2, layers = 2, 
                 kernel_size = 6, stride = 1, loss_function='focalLoss') -> None:
        """
        Initialize the CNN model with the given parameters.
        """
        super(CNN, self).__init__()

        self.layers = layers
        self.seq_length = seq_length
        self.loss_function = loss_function
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=kernel_size, 
                               stride=stride, padding=int(kernel_size/2), 
                               padding_mode='replicate', groups=2)

        init.xavier_uniform_(self.conv1.weight)  

        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(3)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size,
                               stride=stride, padding=int(kernel_size/2), 
                               padding_mode='replicate', groups=2)

        init.xavier_uniform_(self.conv2.weight)  

        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(3)

        if self.layers > 2: 
            self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, 
                                   kernel_size=kernel_size,
                                   stride=stride, padding=int(kernel_size/2), 
                                   padding_mode='replicate', groups=2)

            init.xavier_uniform_(self.conv3.weight)  
            self.bn3 = nn.BatchNorm1d(64)
            self.pool3 = nn.MaxPool1d(3)  
        
        if self.layers > 3: 
            self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, 
                                   kernel_size=kernel_size, 
                                   stride=stride, padding=int(kernel_size/2), 
                                   padding_mode='replicate', groups=2)

            init.xavier_uniform_(self.conv4.weight)  
            self.bn4 = nn.BatchNorm1d(128)
            self.pool4 = nn.MaxPool1d(3)  


        self.fc1_input_dim = self._calculate_fc1_input_dim()
        
        self.fc1 = nn.Linear(self.fc1_input_dim, 200)
        '''
        print(self.layers)
        if self.layers == 2:
            self.fc1 = nn.Linear(1056, 200)
            init.xavier_uniform_(self.fc1.weight)  
        elif self.layers == 3:
            self.fc1 = nn.Linear(704, 200)
            init.xavier_uniform_(self.fc1.weight) 
        elif self.layers == 4:
            self.fc1 = nn.Linear(512, 200)
            init.xavier_uniform_(self.fc1.weight) 
        '''

        self.fc2 = nn.Linear(200, num_classes)
        init.xavier_uniform_(self.fc2.weight)  


    def _calculate_fc1_input_dim(self):
        """
        Calculate the input dimension for the fully connected layer based on the sequence length
        and the number of layers.
        """
        length = self.seq_length
        length = (length + 2 * int(6/2) - 6) // 1 + 1
        length = length // 3  # After pool1

        length = (length + 2 * int(6/2) - 6) // 1 + 1
        length = length // 3  # After pool2

        if self.layers > 2:
            length = (length + 2 * int(6/2) - 6) // 1 + 1
            length = length // 3  # After pool3

        if self.layers > 3:
            length = (length + 2 * int(6/2) - 6) // 1 + 1
            length = length // 3  # After pool4

        out_channels = 16 * 2**(self.layers - 1)
        return length * out_channels

    def forward(self, x):
        """
        Forward pass of the CNN.

        Parameters:
            x (Tensor): The input data tensor with shape (batch_size, channels, length).

        Returns:
            Tensor: The output tensor with shape (batch_size, num_classes).
        """
        #print("new CNN was created with seq_length: "+ str(self.seq_length)+" layers: "+ str(self.layers))

        #print(f'Input shape: {x.shape}')
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)        
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        if self.layers == 3: 
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.pool3(x)
            
        if self.layers == 4: 
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.pool3(x)
            
            x = self.conv4(x)
            x = self.bn4(x)
            x = F.relu(x)
            x = self.pool4(x)

        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        if (self.loss_function=='NLLLoss') or (self.loss_function=='focalLoss'):
            return F.log_softmax(x, dim=1)  
        else: 
            return x

def setup_model(num_classes: int, show_architecture: bool = True) -> nn.Module:
    """
    Setup and initialize the CNN model with the specified number of output classes and 
    configuration.

    Parameters:
        num_classes (int): Number of classes for the final output layer of the CNN.
        device (torch.device): The device (CPU or GPU) where the model 
        should be allocated. show_architecture (bool): If True, print the 
        architecture of the model. Default is True.

    Returns:
        nn.Module: The initialized CNN model, potentially wrapped in a nn.DataParallel 
        module if multiple GPUs are available.

    This function loads configuration from a YAML file, initializes a CNN model accordin
    to this configuration, and moves the model to the specified device. If multiple GPUs 
    are available, it wraps the model in a nn.DataParallel module to enable parallel 
    processing.
    """
    # Load neural network configuration from YAML files
    nn_config = load_yaml_files(nn_config=True, regressor=False)

    print('----- model setup --------')
    # Initialize the CNN model with parameters from the configuration file

    model = CNN(num_classes=num_classes, seq_length = nn_config['data']['seq_length'], 
                layers=nn_config['training']['layers'],
                loss_function=nn_config['training']['loss'])

    # Move the model to the specified device (CPU or GPU)
    if torch.cuda.is_available():
        model = model.to(device)

    # If more than one GPU is available, use DataParallel for parallel processing
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    # Optionally print the model architecture
    if show_architecture:
        print("Model Architecture:")
        print(model)

    return model

def get_criterion(nn_config, class_weights):
    """
    Returns the loss criterion based on the configuration.

    :param nn_config: Dictionary containing neural network configuration, including the loss type.
    :param class_weights: Tensor representing the weights for each class, used in weighted loss functions.
    :param nn_config['training']['focal_loss_scale']: The gamma parameter for the focal loss, if used.
    :return: criterion, criterion_synthetic_samples - the loss functions for the model and synthetic samples.
    """
    loss_type = nn_config['training']['loss']

    if loss_type == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'non_weighted_CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'NLLLoss':
        criterion = nn.NLLLoss(weight=class_weights)
    elif loss_type == 'focalLoss':
        # Assuming focal_loss is a custom loss function defined elsewhere
        criterion = focal_loss(alpha=class_weights, 
                                gamma=nn_config['training']['focal_loss_scale'])
    else:
        raise ValueError(f'The required loss {loss_type} is not supported.')

    # Assuming the same criterion is used for synthetic samples in this context
    criterion_synthetic_samples = criterion

    return criterion, criterion_synthetic_samples

def load_yaml_files(nn_config: bool = True, regressor: bool = True):
    """
    Load configuration data from YAML files based on the specified options.

    Parameters:
        nn_config (bool): Flag indicating whether to load the neural network 
                          configuration file. Default is True.
        regressor (bool): Flag indicating whether to load the regressor configuration 
                          file. This flag is only considered if `nn_config` is also True. 
                          Default is True.

    Returns:
        A tuple containing the loaded configurations as dictionaries.
        - If both `nn_config` and `regressor` are True, returns a tuple 
          with both configurations.
        - If only `nn_config` is True, returns a single-element tuple with the neural 
          network configuration.
        - Returns None if `nn_config` is False.

    This function reads configuration settings from 'src/configuration/nn_config.yaml' and optionally from
    'src/configuration/regressor.yaml'. The returned configurations are used to set up and 
    customize the behavior of neural network models and training processes.
    """
    if nn_config and regressor:
        with open('src/configuration/nn_config.yaml', 'r') as file:
            nn_config_dict = yaml.safe_load(file)

        with open('src/configuration/regressor.yaml', 'r') as file:
            regressor_dict = yaml.safe_load(file)
        
        print('------ Data loading -------------------')
        print('mode: ', nn_config_dict['data']['mode_running'], nn_config_dict['data']['sample_size'])

        return nn_config_dict, regressor_dict

    elif nn_config:
        with open('src/configuration/nn_config.yaml', 'r') as file:
            nn_config_dict = yaml.safe_load(file)

        return nn_config_dict
    else: 
        raise Exception("Files were not loaded, please check function arguments")

def get_counts_and_weights_by_class(y_train_labeled: torch.Tensor,
                                    y_test_labeled: torch.Tensor,
                                    x_train: torch.Tensor):
    """
    Calculate the class weights based on the labels in the training set and count occurrences
    of each class in both training and testing sets.

    Parameters:
        y_train_labeled (torch.Tensor): Tensor of labeled classes for the training set.
        y_test_labeled (torch.Tensor): Tensor of labeled classes for the testing set.
        x_train (torch.Tensor): Training data to determine the data type for the weights tensor.

    Returns:
        tuple: A tuple containing the computed class weights tensor, the number of unique classes,
               and a zip object of unique classes and their counts in the training set.

    This function computes balanced class weights using 'compute_class_weight' from scikit-learn,
    square roots these weights, and converts them into a tensor. It also prints the number of
    occurrences of each class in the training and testing sets.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available()
                                    and gpu else "cpu")

    print('Training set')
    classes = np.unique(y_train_labeled.numpy())
    num_classes = len(classes)
    print('num_classes:', num_classes)

    # Count the occurrences of each unique value in training set
    unique_classes, counts = np.unique(y_train_labeled.numpy(), return_counts=True)

    # Display the counts for each class in training set
    for cls, count in zip(unique_classes, counts):
        print(f"Class {cls}: {count} occurrences")

    print('Testing set')
    unique_classes_test, counts_test = np.unique(y_test_labeled.numpy(), 
                                                return_counts=True)

    # Display the counts for each class in testing set
    for cls, count in zip(unique_classes_test, counts_test):
        print(f"Class {cls}: {count} occurrences")

    class_weights = compute_class_weight('balanced', classes=unique_classes, 
                                        y=y_train_labeled.numpy())

    class_weights = np.sqrt(class_weights)
    class_weights = torch.tensor(class_weights, device=device, dtype=x_train.dtype)

    return class_weights, num_classes, zip(unique_classes, counts)   

def run_cnn(create_samples: Any, vae_model=None, pp = None,
            wandb_active = True, prior=False,
            hyperparam_opt = True) -> None:
    """
    Main function to run a Convolutional Neural Network (CNN) for classification tasks.

    Parameters:
    - create_samples: The function or flag to generate synthetic samples.
    - mode_running: Flag to indicate mode of operation ('load' for loading data, etc.)
    - mean_prior_dict: Dictionary containing prior distributions for synthetic data.
    - vae_model: Variational AutoEncoder model for synthetic data.
    - pp: A list of Posterior Predictive checks.
    - nn_config['training']['opt_method']: Optimization method ('twolosses' or 'oneloss').

    Returns:
    None
    """
    nn_config, config_file = load_yaml_files(nn_config=True, regressor=True)

    nn_config, config_file = wset.cnn_hyperparameters(wandb_active, hyperparam_opt, 
                                                      nn_config, config_file)

    vae_model: str = config_file['model_parameters']['ID']
    
    x_train, x_test, y_train, y_test, x_val, y_val, \
    label_encoder, y_train_labeled, y_test_labeled = utils.get_data(nn_config['data']['sample_size'], 
                                                              nn_config['data']['mode_running'])
 
    class_weights, num_classes, _  = get_counts_and_weights_by_class(y_train_labeled, 
                                                        y_test_labeled, x_train)
    


    training_data = utils.move_data_to_device((x_train, y_train), device)
    val_data = utils.move_data_to_device((x_val, y_val), device)
    testing_data = utils.move_data_to_device((x_test, y_test), device)

    best_val = np.iinfo(np.int64).max
    harder_samples = True
    no_improvement_count, counter, weight_f1_score_hyperparameter_search  = 0, 0, 0.15
    train_loss_values, val_loss_values, train_accuracy_values, \
                                        val_accuracy_values  = [], [], [], []
    


    model = setup_model(num_classes, device)

    wset.setup_gradients(wandb_active, model)

    train_dataloader = create_dataloader(training_data, nn_config['training']['batch_size'])
    val_dataloader = create_dataloader(val_data, nn_config['training']['batch_size'])
    test_dataloader = create_dataloader(testing_data, nn_config['training']['batch_size'])

    criterion, criterion_synthetic_samples = get_criterion(nn_config, class_weights)

    beta_actual = nn_config['training']['beta_initial']

    optimizer1, optimizer2, locked_masks, \
                locked_masks2 = initialize_optimizers(model, nn_config_dict = nn_config)

    batcher = SyntheticDataBatcher(pp = pp, vae_model=vae_model, 
                                  n_samples=nn_config['training']['synthetic_samples_by_class'],
                                seq_length = x_train.size(-1), prior=prior)

    for epoch in range(nn_config['training']['epochs']):
        print(nn_config['training']['opt_method'], create_samples, harder_samples, 
            counter, nn_config['training']['ranking_method'])

        if (nn_config['training']['opt_method']=='twolosses' 
            and create_samples and harder_samples):
            dict_priorization = {}

            if (nn_config['training']['ranking_method']=='no_priority') or (epoch < 2):
                synthetic_data_loader = batcher.create_synthetic_batch(b=beta_actual, 
                                                wandb_active=wandb_active, 
                                                n_oversampling=nn_config['training']['n_oversampling'])

            elif nn_config['training']['ranking_method']=='proportion':
                ranking, proportions = get_dict_class_priorization(model, 
                                                        train_dataloader, 
                                                        ranking_method = 
                                                        nn_config['training']['ranking_method'])
                
                
                proportions = ((proportions - np.min(proportions))/
                              (np.max(proportions) - np.min(proportions))*16 + 8)

                counter2 = 0

                for o in ranking:
                    dict_priorization[label_encoder[o]] =  int(proportions[counter2])
                    counter2 = counter2 + 1

                synthetic_data_loader = batcher.create_synthetic_batch(b=beta_actual, 
                                                wandb_active=wandb_active, 
                                                samples_dict = dict_priorization, 
                                                n_oversampling=nn_config['training']['n_oversampling'])            
            else:
                ranking, _ = get_dict_class_priorization(model, train_dataloader, 
                                                        ranking_method = 
                                                        nn_config['training']['ranking_method'])
                
                ranking_penalization = 1.25
                for o in ranking:
                    objects = nn_config['training']['synthetic_samples_by_class']*ranking_penalization
                    dict_priorization[label_encoder[o]] =  int(objects)
                    if ranking_penalization>0.5:
                        ranking_penalization = ranking_penalization/1.25

                synthetic_data_loader = batcher.create_synthetic_batch(b=beta_actual, 
                                                wandb_active=wandb_active, 
                                                samples_dict = dict_priorization,
                                                n_oversampling = nn_config['training']['n_oversampling'])

            print("The Batch was created sucessfully...")
            decay_parameter1 = nn_config['training']['decay_parameter_1']
            decay_parameter2= nn_config['training']['decay_parameter_2']
            beta_actual = decay_parameter1 + (1-decay_parameter1) * np.exp(-decay_parameter2 * epoch)
            harder_samples = False

        elif  nn_config['training']['opt_method']=='twolosses' and create_samples: 
            print("Using available synthetic data")
        else:
            print("Skipping synthetic sample creation")
            synthetic_data_loader = None

        running_loss, model, val_loss = train_one_epoch_alternative(model, criterion, 
                                        optimizer1, train_dataloader, val_dataloader, device,
                                        mode = nn_config['training']['opt_method'], 
                                        criterion_2= criterion_synthetic_samples, 
                                        dataloader_2 = synthetic_data_loader,
                                        optimizer_2 = optimizer2, locked_masks2 = locked_masks2,
                                        locked_masks = locked_masks, 
                                        repetitions = nn_config['training']['repetitions'])
                                 
        _, accuracy_val, f1_val, roc_val_ovo, roc_val_ovr = evaluate_dataloader(model, val_dataloader, 
                                                                                criterion, device)
        _, accuracy_train, f1_train, roc_train_ovo, roc_train_ovr = evaluate_dataloader(model, train_dataloader,
                                                                                 criterion, device)


        if nn_config['training']['opt_method']=='twolosses':
            synthetic_loss, accuracy_train_synthetic,\
            f1_synthetic, roc_syn_ovo, roc_syn_ovr =  evaluate_dataloader(model, synthetic_data_loader, 
                                                criterion_synthetic_samples, device)

        elif nn_config['training']['opt_method']=='oneloss':
            synthetic_loss, accuracy_train_synthetic, f1_synthetic, roc_syn_ovo, roc_syn_ovr = 0, 0, 0, 0, 0
        else: 
            raise('This method is unavailable, please use: oneloss or twolosses')
        
        condition1 = (accuracy_train_synthetic>nn_config['training']['threshold_acc_synthetic'])
        condition2 = (counter > 5) 
        
        if condition1 or condition2:
            harder_samples = True
            counter = 0
        else:
            counter = counter + 1


        train_loss_values.append(running_loss)
        val_loss_values.append(val_loss)
        train_accuracy_values.append(accuracy_train)
        val_accuracy_values.append(accuracy_val)
        
        if nn_config['training']['opt_method']=='twolosses':
            weighted_f1 = (f1_synthetic*weight_f1_score_hyperparameter_search +
                          f1_val*(1-weight_f1_score_hyperparameter_search))
        else: 
            weighted_f1 = f1_val

        if  (val_loss < best_val):
            best_val = val_loss
            best_model = model.state_dict()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count >=  nn_config['training']['patience']:
            print(f"Stopping early after {epoch + 1} nn_config['training']['epochs']")
            break

        if wandb_active:
            wandb.log({'epoch': epoch, 'loss': running_loss, 
                      'accuracy_train':accuracy_train, 
                      'synthetic_loss':synthetic_loss, 
                      'acc_synthetic_samples': accuracy_train_synthetic, 
                      'val_loss': val_loss, 'val_accu': accuracy_val, 
                      'f1_val': f1_val, 
                      'f1_train': f1_train, 
                      'f1_synthetic': f1_synthetic, 
                      'weighted_f1': weighted_f1,
                      'patience': no_improvement_count, 
                      'best_val': best_val, 
                      'roc_syn_ovo': roc_syn_ovo,
                      'roc_val_ovo': roc_val_ovo, 
                      'roc_train_ovo' : roc_train_ovo,
                      'roc_syn_ovr': roc_syn_ovr,
                      'roc_val_ovr': roc_val_ovr, 
                      'roc_train_ovr' : roc_train_ovr})

        print(f'epoch: {epoch} loss: {running_loss} acc_train: {accuracy_train} '
             f' f1_train: {f1_train} '
             f'synth_loss: {synthetic_loss} acc_synth: {accuracy_train_synthetic} '
             f'f1_synthetic: {f1_synthetic} val_loss: {val_loss} '
             f'val_accu: {accuracy_val}'
             f'roc_syn_ovo: {roc_syn_ovo} roc_val_ovo: {roc_val_ovo}'
             f'roc_train_ovo: {roc_train_ovo} roc_syn_ovr: {roc_syn_ovr} '
             f'roc_val_ovr: {roc_val_ovr} roc_train_ovr: {roc_train_ovr} '
             f'val_f1: {f1_val} patience: {no_improvement_count} best_val: {best_val}')


    
    # Post-training tasks
    model.load_state_dict(best_model)
    _, accuracy_test, f1_test, roc_test_ovo, roc_test_ovr = evaluate_dataloader(model, test_dataloader, 
                                                    criterion, device)

    _, accuracy_train, f1_train, roc_train_ovo, roc_train_ovr = evaluate_dataloader(model, train_dataloader,
                                                     criterion, device)

    _, accuracy_val, f1_val, roc_val_ovo, roc_val_ovr = evaluate_dataloader(model, val_dataloader, 
                                                criterion, device)
    
    if nn_config['training']['opt_method'] == 'twolosses':
        _, accuracy_train_synthetic, f1_synthetic, roc_syn_ovo, roc_syn_ovr = evaluate_dataloader(model, 
                                                                        synthetic_data_loader, 
                                                                        criterion, device)

    results_dict = {'epoch': epoch, 'loss': running_loss, 
                    'accuracy_train':accuracy_train, 
                    'synthetic_loss':synthetic_loss, 
                    'acc_synthetic_samples': accuracy_train_synthetic, 
                    'val_loss': val_loss, 'val_accu': accuracy_val, 
                    'f1_val': f1_val, 'f1_train': f1_train,
                    'f1_synthetic': f1_synthetic, 'roc_syn_ovo': roc_syn_ovo,
                     'roc_val_ovo': roc_val_ovo, 'roc_train_ovo' : roc_train_ovo,
                     'roc_syn_ovr': roc_syn_ovr,
                     'roc_val_ovr': roc_val_ovr, 'roc_train_ovr' : roc_train_ovr,
                    'patience': no_improvement_count, 'best_val': best_val}

    wset.save_metrics(wandb_active, nn_config['training']['opt_method'], 
                        results_dict, weight_f1_score_hyperparameter_search,
                         accuracy_test, f1_test, roc_test_ovo, roc_test_ovr)

    data_loaders = {
        'Training set': train_dataloader,
        'Validation set': val_dataloader,
        'Testing set': test_dataloader
    }

    # Add synthetic data loader if the optimization method is 'twolosses'
    if nn_config['training']['opt_method'] == 'twolosses':
        data_loaders['Synthetic'] = synthetic_data_loader

    # Evaluate and plot confusion matrix for each data loader
    for title_suffix, data_loader in data_loaders.items():
        _ = utils.evaluate_and_plot_cm_from_dataloader(
            model, data_loader, label_encoder, 
            f'Confusion Matrix - {title_suffix}', wandb_active=wandb_active
        )

    del x_train, y_train, x_val, y_val, x_test, y_test
    del train_dataloader, val_dataloader, test_dataloader
    torch.save(model, 'self_regulated_cnn_model.pt')
    del model
    del optimizer1, optimizer2, locked_masks, locked_masks2 
    gc.collect() 

    #Create a new data set only in the first execution (from ten)
    #nn_config['data']['mode_running'] = "load" 
    with open('src/configuration/nn_config.yaml', 'w') as file:
        yaml.dump(nn_config, file)
        file.flush()  # Force flushing the file buffer to disk
    time.sleep(2)