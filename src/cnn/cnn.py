
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from src.utils import *
from sklearn.utils.class_weight import compute_class_weight
import wandb
from typing import Any, Dict
import yaml 
from src.sampler.getbatch import SyntheticDataBatcher
from src.cnn.focalloss import  FocalLossMultiClass as focal_loss
import src.utils as utils
import src.wandb_setup as wset
import torch.nn.init as init
import src.wandb_setup as wsetup
from src.cnn.training_cnn import *

gpu: bool = True # fail when true is selected
device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")

class CNN(nn.Module):
    def __init__(self, num_classes: int = 2, layers = 2, kernel_size = 6, stride = 1) -> None:
        super(CNN, self).__init__()

        self.layers = layers
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=kernel_size, 
                               stride=stride, padding=int(kernel_size/2), padding_mode='replicate', 
                              groups=2)
        init.xavier_uniform_(self.conv1.weight)  

        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(3)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, 
                               stride=stride, padding=int(kernel_size/2), padding_mode='replicate', 
                              groups=2)
        init.xavier_uniform_(self.conv2.weight)  

        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(3)

        if self.layers > 2: 
            self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, 
                                   padding=int(kernel_size/2), padding_mode='replicate', 
                                    groups=2)
            init.xavier_uniform_(self.conv3.weight)  
            self.bn3 = nn.BatchNorm1d(64)
            self.pool3 = nn.MaxPool1d(3)  
        
        if self.layers > 3: 
            self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride, 
                                   padding=int(kernel_size/2), padding_mode='replicate', 
                                  groups=2)
            init.xavier_uniform_(self.conv4.weight)  
            self.bn4 = nn.BatchNorm1d(128)
            self.pool4 = nn.MaxPool1d(3)  

        if self.layers == 2:
            self.fc1 = nn.Linear(1056, 200)
            init.xavier_uniform_(self.fc1.weight)  
        elif self.layers == 3:
            self.fc1 = nn.Linear(704, 200)
            init.xavier_uniform_(self.fc1.weight) 
        elif self.layers == 4:
            self.fc1 = nn.Linear(512, 200)
            init.xavier_uniform_(self.fc1.weight) 
        
        self.fc2 = nn.Linear(200, num_classes)
        init.xavier_uniform_(self.fc2.weight)  




    def forward(self, x):
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
        
        if (nn_config['training']['loss']=='NLLLoss') or (nn_config['training']['loss']=='focalLoss'):
            return F.log_softmax(x, dim=1)  
        else: 
            return x

def setup_model(num_classes: int, device: torch.device, show_architecture: bool = True) -> nn.Module:

    print('----- model setup --------')
    model = CNN(num_classes=num_classes, layers = nn_config['training']['layers'])
    if torch.cuda.is_available():
        model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    if show_architecture:
        print("Model Architecture:")
        print(model)

    return model

def get_criterion(nn_config, class_weights, focal_loss_scale =None):
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
        criterion = focal_loss(alpha=class_weights, gamma=nn_config['training']['focal_loss_scale'])
    else:
        raise ValueError(f'The required loss {loss_type} is not supported.')

    # Assuming the same criterion is used for synthetic samples in this context
    criterion_synthetic_samples = criterion

    return criterion, criterion_synthetic_samples

def run_cnn(create_samples: Any, mean_prior_dict: Dict = None, 
            vae_model=None, PP=[], wandb_active = True, prior=False) -> None:
    """
    Main function to run a Convolutional Neural Network (CNN) for classification tasks.

    Parameters:
    - create_samples: The function or flag to generate synthetic samples.
    - mode_running: Flag to indicate mode of operation ('load' for loading data, etc.)
    - mean_prior_dict: Dictionary containing prior distributions for synthetic data.
    - vae_model: Variational AutoEncoder model for synthetic data.
    - PP: A list of Posterior Predictive checks.
    - nn_config['training']['opt_method']: Optimization method ('twolosses' or 'oneloss').

    Returns:
    None
    """
    with open('src/nn_config.yaml', 'r') as file:
        nn_config = yaml.safe_load(file)

    with open('src/configuration/regressor.yaml', 'r') as file:
        config_file: Dict[str, Any] = yaml.safe_load(file)

    vae_model: str = config_file['model_parameters']['ID']

    hyperparam_opt = True

    print('#'*50)
    print('TRAINING CNN')
    print('#'*50)
    
    device = setup_torch_environment()
    
    print('------ Data loading -------------------')
    print('mode: ', nn_config['data']['mode_running'], nn_config['data']['sample_size'])
    x_train, x_test, y_train, y_test, x_val, y_val, \
    label_encoder, y_train_labeled, y_test_labeled = get_data(nn_config['data']['sample_size'], 
                                                              nn_config['data']['mode_running'])


    print(label_encoder)
    print('Training set')
    classes = np.unique(y_train_labeled.numpy())
    num_classes = len(classes)
    print('num_classes: ', num_classes)
    # Count the occurrences of each unique value
    unique_classes, counts = np.unique(y_train_labeled.numpy(), return_counts=True)
    # Display the counts for each class
    for cls, count in zip(unique_classes, counts):
        print(f"Class {cls}: {count} occurrences")      

    print('Testing set')
    unique_classes, counts = np.unique(y_test_labeled.numpy(), return_counts=True)
    # Display the counts for each class
    for cls, count in zip(unique_classes, counts):
        print(f"Class {cls}: {count} occurrences")    

    model = setup_model(num_classes, device)

    wset.setup_gradients(wandb_active, model)

    class_weights = compute_class_weight('balanced', np.unique(y_train_labeled.numpy()), y_train_labeled.numpy())
    class_weights = np.sqrt(class_weights)
    class_weights = torch.tensor(class_weights).to(device, dtype=x_train.dtype)
    
    training_data = utils.move_data_to_device((x_train, y_train), device)
    val_data = utils.move_data_to_device((x_val, y_val), device)
    testing_data = utils.move_data_to_device((x_test, y_test), device)

    # Main training loop
    best_val = np.iinfo(np.int64).max
    harder_samples = True
    no_improvement_count = 0
    train_loss_values = []
    val_loss_values = []
    train_accuracy_values = []
    val_accuracy_values = []
    counter = 0
    weight_f1_score_hyperparameter_search = 0
    
    nn_config, config_file = wsetup.cnn_hyperparameters(wandb_active, hyperparam_opt, nn_config, config_file)

    train_dataloader = create_dataloader(training_data, nn_config['training']['batch_size'])
    val_dataloader = create_dataloader(val_data, nn_config['training']['batch_size'])
    test_dataloader = create_dataloader(testing_data, nn_config['training']['batch_size'])

    criterion, criterion_synthetic_samples = get_criterion(nn_config, class_weights, 
                                                          focal_loss_scale=nn_config['training']['focal_loss_scale'])

    beta_actual = nn_config['training']['beta_initial']

    optimizer1, optimizer2, locked_masks, locked_masks2 = initialize_optimizers(model, opt_method = nn_config['training']['opt_method'],
                                                                                EPS=nn_config['training']['EPS'], base_learning_rate =nn_config['training']['base_learning_rate'], 
                                                                                scaling_factor =nn_config['training']['scaling_factor'])

    batcher = SyntheticDataBatcher(PP = PP, vae_model=vae_model, n_samples=nn_config['training']['synthetic_samples_by_class'], 
                                    seq_length = x_train.size(-1), prior=prior)

    for epoch in range(nn_config['training']['epochs']):
        print(nn_config['training']['opt_method'], create_samples, harder_samples, counter, nn_config['training']['ranking_method'])
        if nn_config['training']['opt_method']=='twolosses' and create_samples and harder_samples:
            if (nn_config['training']['ranking_method']=='no_priority') or (epoch < 2):
                synthetic_data_loader = batcher.create_synthetic_batch(b=beta_actual, 
                                                                    wandb_active=wandb_active, 
                                                                    samples_dict = None, oversampling = True, 
                                                                    n_oversampling=nn_config['training']['n_oversampling'])
            elif nn_config['training']['ranking_method']=='proportion':
                ranking, proportions = get_dict_class_priorization(model, train_dataloader, ranking_method = nn_config['training']['ranking_method'])
                
                dict_priorization = {}
                proportions = (proportions - np.min(proportions))/(np.max(proportions) - np.min(proportions))*16 + 8
                counter2 = 0
                for o in ranking:
                    dict_priorization[label_encoder[o]] =  int(proportions[counter2])
                    counter2 = counter2 + 1

                print(dict_priorization)
                synthetic_data_loader = batcher.create_synthetic_batch(b=beta_actual, 
                                                                    wandb_active=wandb_active, 
                                                                    samples_dict = dict_priorization, 
                                                                    oversampling = True, 
                                                                    n_oversampling=nn_config['training']['n_oversampling'])            
            else:
                ranking, _ = get_dict_class_priorization(model, train_dataloader, 
                                                        ranking_method =nn_config['training']['ranking_method'])
                dict_priorization = {}
                ranking_penalization = 1.25
                
                for o in ranking:
                    dict_priorization[label_encoder[o]] =  int(nn_config['training']['synthetic_samples_by_class']*ranking_penalization)
                    if ranking_penalization>0.5:
                        ranking_penalization = ranking_penalization/1.25

                synthetic_data_loader = batcher.create_synthetic_batch(b=beta_actual, 
                                                                    wandb_active=wandb_active, 
                                                                    samples_dict = dict_priorization, 
                                                                    oversampling = True, 
                                                                    n_oversampling = nn_config['training']['n_oversampling'])                
            beta_actual = 0.85 + 0.15 * np.exp(-0.1 * epoch)
            harder_samples = False
            print(nn_config['training']['opt_method'], create_samples, harder_samples, counter)

        elif  nn_config['training']['opt_method']=='twolosses' and create_samples: 
            print("Using available synthetic data")
        else:
            print("Skipping synthetic sample creation")
            synthetic_data_loader = None

        running_loss, model, val_loss = train_one_epoch_alternative(model, criterion, optimizer1, train_dataloader, val_dataloader, device, 
                                        mode = nn_config['training']['opt_method'], 
                                        criterion_2= criterion_synthetic_samples, 
                                        dataloader_2 = synthetic_data_loader,
                                        optimizer_2 = optimizer2, locked_masks2 = locked_masks2, 
                                        locked_masks = locked_masks, repetitions = nn_config['training']['repetitions'])
                                 
        _, accuracy_val, f1_val = evaluate_dataloader_weighted_metrics(model, val_dataloader, criterion, device)
        _, accuracy_train, f1_train = evaluate_dataloader_weighted_metrics(model, train_dataloader, criterion, device)


        synthetic_loss, accuracy_train_synthetic, f1_synthetic =  evaluate_dataloader_weighted_metrics(model, synthetic_data_loader, 
                                                                    criterion_synthetic_samples, device)
       
        condition1 = (accuracy_train_synthetic>nn_config['training']['threshold_acc_synthetic'])
        condition2 = (counter > 3) 
        
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
            weighted_f1 = f1_synthetic*weight_f1_score_hyperparameter_search + f1_val*(1-weight_f1_score_hyperparameter_search)
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
            wandb.log({'epoch': epoch, 'loss': running_loss, 'accuracy_train':accuracy_train, 
                        'synthetic_loss':synthetic_loss, 'acc_synthetic_samples': accuracy_train_synthetic, 
                        'val_loss': val_loss, 'val_accu': accuracy_val, 'f1_val': f1_val, 
                        'f1_train': f1_train, 'f1_synthetic': f1_synthetic, 'weighted_f1': weighted_f1,
                        'patience': no_improvement_count, 'best_val': best_val})

        print('epoch:', epoch, ' loss:', running_loss, ' acc_train:', accuracy_train,  ' f1_train:', f1_train,
                                ' synth_loss:', synthetic_loss, ' acc_synth:', accuracy_train_synthetic, ' f1_synthetic:', f1_synthetic, 
                                ' val_loss', val_loss, ' val_accu:', accuracy_val, ' val_f1:', f1_val, 
                                ' patience:', no_improvement_count, ' best_val:', best_val)
    
    # Post-training tasks
    model.load_state_dict(best_model)

    #plot_training(range(len(train_loss_values)), train_loss_values, val_loss_values, train_accuracy_values, val_accuracy_values)
    _, accuracy_test, f1_test = evaluate_dataloader_weighted_metrics(model, test_dataloader, criterion, device)
    _, accuracy_train, f1_train = evaluate_dataloader_weighted_metrics(model, train_dataloader, criterion, device)
    _, accuracy_val, f1_val = evaluate_dataloader_weighted_metrics(model, val_dataloader, criterion, device)
    
    if nn_config['training']['opt_method'] == 'twolosses':
        _, accuracy_train_synthetic, f1_synthetic = evaluate_dataloader_weighted_metrics(model, synthetic_data_loader, criterion, device)

    results_dict = {'epoch': epoch, 'loss': running_loss, 'accuracy_train':accuracy_train, 
                        'synthetic_loss':synthetic_loss, 'acc_synthetic_samples': accuracy_train_synthetic, 
                        'val_loss': val_loss, 'val_accu': accuracy_val, 'f1_val': f1_val, 'f1_train': f1_train,
                        'f1_synthetic': f1_synthetic, 
                        'patience': no_improvement_count, 'best_val':best_val}

    wsetup.save_metrics(wandb_active, nn_config['training']['opt_method'], 
                        results_dict, weight_f1_score_hyperparameter_search,
                         accuracy_test, f1_test)


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