
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module  # Use nn.Module instead of _Loss
import numpy as np
from src.utils import *
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import wandb
from typing import Union, Tuple, Optional, Any, Dict, List
import yaml 
from src.sampler.getbatch import SyntheticDataBatcher
from src.focalloss import  FocalLossMultiClass as focal_loss
import src.utils as utils
import torch.optim as optim
import torch.nn.init as init
from sklearn.metrics import accuracy_score, f1_score
with open('src/nn_config.yaml', 'r') as file:
    nn_config = yaml.safe_load(file)

with open('src/paths.yaml', 'r') as file:
    YAML_FILE: Dict[str, Any] = yaml.safe_load(file)

PATHS: Dict[str, str] = YAML_FILE['paths']
PATH_PRIOS: str = PATHS['PATH_PRIOS']
PATH_MODELS: str = PATHS['PATH_MODELS']
CLASSES: List[str] = ['ACEP','CEP', 'DSCT', 'ECL',  'ELL', 'LPV',  'RRLYR', 'T2CEP']
PATH_DATA_FOLDER: str =  PATHS['PATH_DATA_FOLDER']
mean_prior_dict: Dict[str, Any] = load_yaml_priors(PATH_PRIOS)
PATH_DATA: str = PATHS["PATH_DATA_FOLDER"]

with open('src/regressor.yaml', 'r') as file:
    config_file: Dict[str, Any] = yaml.safe_load(file)

vae_model: str = config_file['model_parameters']['ID']
sufix_path: str = config_file['model_parameters']['sufix_path']
gpu: bool = True # fail when true is selected

device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")

class CNN(nn.Module):
    def __init__(self, num_classes: int = 2, layers = 2) -> None:
        super(CNN, self).__init__()

        self.layers = layers
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=6, stride=1)
        init.xavier_uniform_(self.conv1.weight)  

        self.bn1 = nn.BatchNorm1d(8)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=6, stride=1)
        init.xavier_uniform_(self.conv2.weight)  

        self.bn2 = nn.BatchNorm1d(16)
        self.pool2 = nn.MaxPool1d(2)

        if self.layers > 2: 
            self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=6, stride=1)
            init.xavier_uniform_(self.conv3.weight)  
            self.bn3 = nn.BatchNorm1d(32)
            self.pool3 = nn.MaxPool1d(2)  
        
        if self.layers > 3: 
            self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=6, stride=1)
            init.xavier_uniform_(self.conv4.weight)  
            self.bn4 = nn.BatchNorm1d(64)
            self.pool4 = nn.MaxPool1d(2)  

        if self.layers == 2:
            self.fc1 = nn.Linear(1136, 200)
            init.xavier_uniform_(self.fc1.weight)  
        elif self.layers == 3:
            self.fc1 = nn.Linear(1056, 200)
            init.xavier_uniform_(self.fc1.weight) 
        elif self.layers == 4:
            self.fc1 = nn.Linear(896, 200)
            init.xavier_uniform_(self.fc1.weight) 

        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(200, num_classes)
        init.xavier_uniform_(self.fc2.weight)  

        self.dropout2 = nn.Dropout(0.2)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.tanh(x)
        x = self.pool1(x)        

        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = F.tanh(x)
        
        x = self.pool2(x)

        if self.layers == 3: 
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.tanh(x)
            x = self.pool3(x)
            
        if self.layers == 4: 
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.tanh(x)
            x = self.pool3(x)
            
            x = self.conv4(x)
            x = self.bn4(x)
            x = F.tanh(x)
            x = self.pool4(x)
        
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.dropout2(x)
        
        if (nn_config['training']['loss']=='NLLLoss') or (nn_config['training']['loss']=='focalLoss'):
            return F.log_softmax(x, dim=1)  
        else: 
            return x

def setup_environment() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(0)  # Set the GPU device index
    print('CUDA active:', torch.cuda.is_available())
    return device

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

def evaluate_and_plot_cm_from_dataloader(model, dataloader, label_encoder, title):
    all_y_data = []
    all_predicted = []
    
    model.eval()  # Switch to evaluation mode

    with torch.no_grad():  # No need to calculate gradients in evaluation
        for batch in dataloader:
            x_data, y_data = batch
            x_data, y_data = x_data.float(), y_data.float()
            outputs = model(x_data)
            _, predicted = torch.max(outputs.data, 1)
            
            # Check if y_data is one-hot encoded and convert to label encoding if it is
            if len(y_data.shape) > 1 and y_data.shape[1] > 1:
                y_data = torch.argmax(y_data, dim=1)
            
            # Move data to CPU and append to lists
            all_y_data.extend(y_data.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
    
    # Compute confusion matrix
    try:
        cm = confusion_matrix(all_y_data, all_predicted, normalize='true')
    except ValueError as e:
        print(f"An error occurred: {e}")
        print(f"y_data shape: {len(all_y_data)}, predicted shape: {len(all_predicted)}")
        return None
    
    plot_cm(cm, label_encoder, title=title)
    #export_recall_latex(all_y_data, all_predicted, label_encoder)
    
    return cm

def get_dict_class_priorization(model, dataloader, ranking_method='CCR'):
    all_y_data = []
    all_predicted = []
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():  # No need to calculate gradients in evaluation
        for batch in dataloader:
            x_data, y_data = batch
            x_data, y_data = x_data.float(), y_data.float()
            outputs = model(x_data)
            _, predicted = torch.max(outputs.data, 1)
            
            # Check if y_data is one-hot encoded and convert to label encoding if it is
            if len(y_data.shape) > 1 and y_data.shape[1] > 1:
                y_data = torch.argmax(y_data, dim=1)
            
            # Move data to CPU and append to lists
            all_y_data.extend(y_data.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy()) 
    try:
        cm = confusion_matrix(all_y_data, all_predicted)
    except ValueError as e:
        print(f"An error occurred: {e}")
        print(f"y_data shape: {len(all_y_data)}, predicted shape: {len(all_predicted)}")
        return None
    print(cm)
    ranking = rank_classes(cm, method=ranking_method)
    return ranking

def find_argmax_off_diagonal(matrix):
    # Create a copy to avoid modifying the original matrix
    matrix_copy = np.array(matrix)
    # Zero out the diagonal elements
    np.fill_diagonal(matrix_copy, 0)
    # Finding the indices of the maximum off-diagonal value
    argmax_indices = np.unravel_index(np.argmax(matrix_copy, axis=None), matrix_copy.shape)
    return argmax_indices

def rank_classes(confusion_matrix, method='CCR', verbose=False):
    if method == 'CCR':
        ccrs = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        class_ranking = np.argsort(ccrs)
        return class_ranking, ccrs[class_ranking]

    elif method == 'max_confusion':
        off_diagonal_sum = np.sum(confusion_matrix, axis=1) - np.diag(confusion_matrix)
        class_ranking = np.argsort(off_diagonal_sum)[::-1]  
        return class_ranking, off_diagonal_sum[class_ranking]

    elif method == 'max_pairwise_confusion':
        m = confusion_matrix
        class_ranking = []
        while len(class_ranking)<m.shape[0]:
            if verbose:
                print(find_argmax_off_diagonal(m))
            (a,b) = find_argmax_off_diagonal(m)
            m[a,b] = 0
            m[b,a] = 0
            if a not in class_ranking:
                class_ranking.append(a)
            if b not in class_ranking:
                class_ranking.append(b)
            if verbose:
                print(a,b)
                print(m)
                print(class_ranking)
        return np.asarray(class_ranking), m
    else:
        raise ValueError("Unknown method: {}".format(method))

def train_one_epoch_alternative(
        model: torch.nn.Module, 
        criterion: Module,  # Here
        optimizer: Optimizer, 
        dataloader: DataLoader, 
        val_dataloader: DataLoader, 
        device: torch.device, 
        mode: str = 'oneloss', 
        criterion_2: Optional[Module] = None,  # And here
        dataloader_2: Optional[DataLoader] = None, 
        optimizer_2: Optional[Optimizer] = None, 
        locked_masks2 = None, 
        locked_masks = None, 
        repetitions = 10
    ) -> float:    
    
    running_loss = 0.0
    num_batches = 0
    val_loss = 0
    if mode=='oneloss':
        for inputs, labels in dataloader:
            num_batches += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, labels_indices = torch.max(labels, 1)  # Convert one-hot to indices
            loss = criterion(outputs, labels_indices)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=6.0, norm_type=2)
            optimizer.step()
            running_loss += loss.item()
        val_loss, _, val_f1 = evaluate_dataloader_weighted_metrics(model, val_dataloader, criterion, device)

        return running_loss, model, val_loss
    
    elif mode=='twolosses':
        if criterion_2 is None or dataloader_2 is None or optimizer_2 is None:
            raise ValueError("For 'twolosses' mode, criterion_2, dataloader_2, and optimizer_2 must be provided.")     
        running_loss = 0.0
        running_loss_prior = 0.0 
        for inputs, labels in dataloader:
            num_batches += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, labels_indices = torch.max(labels, 1)  # Convert one-hot to indices
            loss = criterion(outputs, labels_indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=6.0, norm_type=2)
            for name, param in model.named_parameters():
                if name in locked_masks2:
                    mask = locked_masks2[name].float().to(param.grad.data.device)
                    param.grad.data *= (mask == 0).float()
                    param.grad.data += mask * param.grad.data.clone()                                                    
            optimizer.step()
            running_loss += loss.item()   
        
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, labels_indices = torch.max(labels, 1)  # Convert one-hot to indices
            loss = criterion(outputs, labels_indices)
            val_loss += loss.item()   

        #val_loss, _ = evaluate_dataloader(model, val_dataloader, criterion, device)

        num_batches = 0
        for _ in range(repetitions):
            for inputs_2, labels_2 in dataloader_2:
                num_batches += 1
                inputs, labels = inputs_2.to(device), labels_2.to(device)
                inputs = inputs.float()
                optimizer_2.zero_grad()
                outputs = model(inputs)
                _, labels_indices = torch.max(labels, 1)  # Convert one-hot to indices
                loss_prior = criterion_2(outputs, labels_indices)
                loss_prior.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5, norm_type=2)
                for name, param in model.named_parameters():
                    if name in locked_masks:
                        mask = locked_masks[name].float().to(param.grad.data.device)
                        param.grad.data *= (mask == 0).float()
                        param.grad.data += mask * param.grad.data.clone()
                optimizer_2.step()
                running_loss_prior += loss_prior.item() 
               
        return running_loss, model, val_loss
  
def evaluate_dataloader(model, dataloader, criterion, device):
    """
    Evaluate the model on a given dataset.

    Parameters:
    - model: The PyTorch model to evaluate.
    - dataloader: DataLoader containing the dataset to evaluate.
    - criterion: The loss function.
    - device: The computing device (CPU or GPU).

    Returns:
    - avg_loss: Average loss over the entire dataset.
    - avg_accuracy: Average accuracy over the entire dataset.
    """
    
    model.eval()  # Set the model to evaluation mode
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()

            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            _, labels_indices = torch.max(labels, 1)  # Convert one-hot to indices
            loss = criterion(outputs, labels_indices)
            
            # Calculate predictions
            predicted = torch.max(outputs, 1)[1]
            
            # Update statistics
            total_loss += loss.item()
            total_correct += (predicted == labels_indices).sum().item()
            total_samples += len(labels_indices)

    avg_accuracy = total_correct / total_samples
    
    model.train()  # Set the model back to training mode
    
    return total_loss, avg_accuracy

def evaluate_dataloader_weighted_metrics(model, dataloader, criterion, device):
    """
    Evaluate the model on a given dataset.

    Parameters:
    - model: The PyTorch model to evaluate.
    - dataloader: DataLoader containing the dataset to evaluate.
    - criterion: The loss function.
    - device: The computing device (CPU or GPU).

    Returns:
    - avg_loss: Average loss over the entire dataset.
    - avg_accuracy: Average accuracy over the entire dataset.
    - f1_score_val: F1 score over the entire dataset.
    """
    
    model.eval()  # Set the model to evaluation mode
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predicted = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()

            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            _, labels_indices = torch.max(labels, 1)  # Convert one-hot to indices
            loss = criterion(outputs, labels_indices)
            
            # Calculate predictions
            predicted = torch.max(outputs, 1)[1]
            
            # Update statistics
            total_loss += loss.item()
            total_correct += (predicted == labels_indices).sum().item()
            total_samples += len(labels_indices)
            
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels_indices.cpu().numpy())

    avg_accuracy = total_correct / total_samples
    f1_score_val = f1_score(all_labels, all_predicted, average='weighted')
    
    model.train()  # Set the model back to training mode
    
    return total_loss, avg_accuracy, f1_score_val

def create_dataloader(data, batch_size):
    data = TensorDataset(*data)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader 

def initialize_masks(model, device='cuda', EPS=0.25, layers=2):
    locked_masks = {}
    locked_masks2 = {}
    conv1_bias_mask = None
    conv2_bias_mask = None
    conv3_bias_mask = None
    conv4_bias_mask = None
    if layers > 4: 
        raise('The current implementation does not support more than 4 layers')
    for name, param in model.named_parameters():
        if ("conv1" in name) and ("weight" in name):
            #print(utils.quantile(torch.abs(param.mean(dim=[1, 2])), EPS))
            quantile = utils.quantile(torch.abs(param.mean(dim=[1, 2])), EPS)
            mask_value = (torch.abs(param.mean(dim=[1, 2])) > quantile).float()
            mask = mask_value.view(-1, 1, 1).repeat(1, param.shape[1], param.shape[2])
            locked_masks[name] = mask
            conv1_bias_mask = mask_value  # Save for bias
        elif ("conv2" in name) and ("weight" in name):
            #print(utils.quantile(torch.abs(param.mean(dim=[1, 2])), EPS))
            quantile = utils.quantile(torch.abs(param.mean(dim=[1, 2])), EPS)
            mask_value = (torch.abs(param.mean(dim=[1, 2])) > quantile).float()
            mask = mask_value.view(-1, 1, 1).repeat(1, param.shape[1], param.shape[2])
            locked_masks[name] = mask
            conv2_bias_mask = mask_value  # Save for bias
        elif "conv3" in name and "weight" in name:
            #print(utils.quantile(torch.abs(param.mean(dim=[1, 2])), EPS))
            quantile = utils.quantile(torch.abs(param.mean(dim=[1, 2])), EPS)
            mask_value = (torch.abs(param.mean(dim=[1, 2])) > quantile).float()
            mask = mask_value.view(-1, 1, 1).repeat(1, param.shape[1], param.shape[2])
            locked_masks[name] = mask
            conv3_bias_mask = mask_value  # Save for bias
        elif "conv4" in name and "weight" in name:
            #print(utils.quantile(torch.abs(param.mean(dim=[1, 2])), EPS))
            quantile = utils.quantile(torch.abs(param.mean(dim=[1, 2])), EPS)
            mask_value = (torch.abs(param.mean(dim=[1, 2])) > quantile).float()
            mask = mask_value.view(-1, 1, 1).repeat(1, param.shape[1], param.shape[2])
            locked_masks[name] = mask
            conv4_bias_mask = mask_value  # Save for bias
        elif "conv1.bias" in name:
            locked_masks[name] = conv1_bias_mask
        elif "conv2.bias" in name:
            locked_masks[name] = conv2_bias_mask
        elif "conv3.bias" in name:
            locked_masks[name] = conv3_bias_mask
        elif "conv4.bias" in name:
            locked_masks[name] = conv4_bias_mask
        elif "fc1" in name and "weight" in name:
            #print(utils.quantile(torch.abs(param.mean(dim=[1])), EPS))
            quantile = utils.quantile(torch.abs(param.mean(dim=[1])), EPS)
            mask_value = (torch.abs(param.mean(dim=1)) > quantile).float()
            mask = mask_value.view(-1, 1).repeat(1, param.shape[1])
            locked_masks[name] = mask
        elif "fc2" in name and "weight" in name:
            #print(utils.quantile(torch.abs(param.mean(dim=[1])), EPS))
            quantile = utils.quantile(torch.abs(param.mean(dim=[1])), EPS)
            mask_value = (torch.abs(param.mean(dim=1)) > quantile).float()
            mask = mask_value.view(-1, 1).repeat(1, param.shape[1])
            locked_masks[name] = mask
        else:
            mask = torch.ones_like(param)
            locked_masks[name] = mask

    # For inverse masks and moving to device
    for name, mask in locked_masks.items():
        mask = mask.to(device)
        mask_inv = 1 - mask
        locked_masks2[name] = mask_inv.to(device)
    print("Summary for locked_masks:")
    for name, mask in locked_masks.items():
        print(f"For parameter {name}, number of locked weights: {mask.sum().item()}")
    print("Summary for locked_masks2:")
    for name, mask in locked_masks2.items():
        print(f"For parameter {name}, number of trainable weights: {mask.sum().item()}")

    return locked_masks, locked_masks2

def initialize_optimizers(model, opt_method, EPS=0.35, base_learning_rate=0.001, scaling_factor=0.5):
    """
    Initializes the optimizers based on the specified optimization method.
    Args:
    - model: The model for which the optimizer needs to be created.
    - opt_method (str): The optimization method ('twolosses' or 'oneloss').
    - EPS (float, optional): EPS value for the 'twolosses' method. Defaults to 0.35.
    - base_learning_rate (float, optional): Base learning rate. Defaults to 0.001.
    - scaling_factor (float, optional): Scaling factor for the second optimizer in 'twolosses'. Defaults to 0.5.

    Returns:
    - optimizer1: The primary optimizer.
    - optimizer2: The secondary optimizer (only for 'twolosses'). Returns None for 'oneloss'.
    """
    if opt_method == 'twolosses':
        print('Using mode: two masks')
        locked_masks, locked_masks2 = initialize_masks(model, EPS=EPS, layers=nn_config['training']['layers'])  # Assuming you have this function
        optimizer1 = optim.Adam(model.parameters(), lr=base_learning_rate)
        optimizer2 = optim.Adam(model.parameters(), lr=scaling_factor * base_learning_rate)
    elif opt_method == 'oneloss':
        print('Using mode: classic backpropagation')
        optimizer1 = optim.Adam(model.parameters(), lr=base_learning_rate)
        optimizer2 = None
        locked_masks, locked_masks2 = None, None
    else:
        raise ValueError(f"Unsupported optimization method: {opt_method}. Supported methods are 'twolosses' and 'oneloss'.")
    
    return optimizer1, optimizer2, locked_masks, locked_masks2

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
    - opt_method: Optimization method ('twolosses' or 'oneloss').

    Returns:
    None
    """
    
    hyperparam_opt = True

    print('#'*50)
    print('TRAINING CNN')
    print('#'*50)
    
    device = setup_environment()
    

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
    class_weights = compute_class_weight('balanced', np.unique(y_train_labeled.numpy()), y_train_labeled.numpy())
    class_weights = np.sqrt(class_weights)
    class_weights = torch.tensor(class_weights).to(device, dtype=x_train.dtype)
    


    training_data = utils.move_data_to_device((x_train, y_train), device)
    val_data = utils.move_data_to_device((x_val, y_val), device)
    testing_data = utils.move_data_to_device((x_test, y_test), device)

    # Main training loop
    best_val_loss = float('inf')
    best_f1_val = 0
    harder_samples = True
    no_improvement_count = 0
    train_loss_values = []
    val_loss_values = []
    train_accuracy_values = []
    val_accuracy_values = []
    counter = 0
    
    if wandb_active:
        if hyperparam_opt:
            nn_config['training']['base_learning_rate'] = wandb.config.learning_rate
            nn_config['training']['batch_size'] = wandb.config.batch_size
            nn_config['training']['patience'] =  wandb.config.patience
            nn_config['training']['repetitions'] =  wandb.config.repetitions
            nn_config['training']['sinthetic_samples_by_class'] = wandb.config.sinthetic_samples_by_class
            nn_config['training']['threshold_acc_synthetic'] = wandb.config.threshold_acc_synthetic
            nn_config['training']['beta_decay_factor'] = wandb.config.beta_decay_factor
            nn_config['training']['EPS'] = wandb.config.EPS
            nn_config['training']['scaling_factor'] = wandb.config.scaling_factor
            vae_model = wandb.config.vae_model
            sufix_path = wandb.config.sufix_path            
            nn_config['training']['layers'] = wandb.config.layers
            nn_config['training']['loss'] = wandb.config.loss
            nn_config['training']['alpha'] = wandb.config.alpha
            config_file['model_parameters']['ID'] =  wandb.config.vae_model
            config_file['model_parameters']['sufix_path'] = wandb.config.sufix_path
            nn_config['training']['focal_loss_scale'] = wandb.config.focal_loss_scale
            nn_config['training']['n_oversampling'] = wandb.config.n_oversampling
            nn_config['training']['ranking_method'] = wandb.config.ranking_method

            with open('src/regressor_output.yaml', 'w') as file:
                yaml.dump(config_file, file)

        wandb.config.epochs = nn_config['training']['epochs']
        wandb.config.patience = nn_config['training']['patience']
        wandb.config.batch_size = nn_config['training']['batch_size']
        wandb.config.repetitions = nn_config['training']['repetitions']
        wandb.config.sinthetic_samples_by_class = nn_config['training']['sinthetic_samples_by_class']
        wandb.config.threshold_acc_synthetic = nn_config['training']['threshold_acc_synthetic']
        wandb.config.beta_decay_factor = nn_config['training']['beta_decay_factor']
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
        wandb.config.alpha =  nn_config['training']['alpha']
        wandb.config.focal_loss_scale = nn_config['training']['focal_loss_scale']
        wandb.config.n_oversampling = nn_config['training']['n_oversampling']
        wandb.config.ranking_method = nn_config['training']['ranking_method']


    epochs = nn_config['training']['epochs']
    patience =  nn_config['training']['patience']
    batch_size = nn_config['training']['batch_size']
    repetitions = nn_config['training']['repetitions']
    sinthetic_samples_by_class = nn_config['training']['sinthetic_samples_by_class']
    threshold_acc_synthetic = nn_config['training']['threshold_acc_synthetic'] 
    beta_decay_factor= nn_config['training']['beta_decay_factor'] 
    beta_initial= nn_config['training']['beta_initial'] 
    EPS= nn_config['training']['EPS'] 
    base_learning_rate= nn_config['training']['base_learning_rate'] 
    scaling_factor= nn_config['training']['scaling_factor'] 
    opt_method= nn_config['training']['opt_method']
    alpha = nn_config['training']['alpha']
    focal_loss_scale = nn_config['training']['focal_loss_scale']
    n_oversampling = nn_config['training']['n_oversampling'] 
    ranking_method = nn_config['training']['ranking_method']

    train_dataloader = create_dataloader(training_data, batch_size)
    val_dataloader = create_dataloader(val_data, batch_size)
    test_dataloader = create_dataloader(testing_data, batch_size)


    print(class_weights)
    if nn_config['training']['loss']=='CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(weight=class_weights) 
        criterion_synthetic_samples = nn.CrossEntropyLoss(weight=class_weights) 
    elif nn_config['training']['loss']=='NLLLoss': 
        criterion = nn.NLLLoss(weight=class_weights) 
        criterion_synthetic_samples = nn.NLLLoss(weight=class_weights) 
    elif  nn_config['training']['loss']=='focalLoss':
        criterion = focal_loss(alpha=class_weights, gamma=focal_loss_scale)
        criterion_synthetic_samples = focal_loss(alpha=class_weights, gamma=focal_loss_scale)      
    else: 
        raise('The required loss is not supported, '+ nn_config['training']['loss'])

    beta_actual = beta_initial

    optimizer1, optimizer2, locked_masks, locked_masks2 = initialize_optimizers(model, opt_method= opt_method,
                                                                                EPS=EPS, base_learning_rate=base_learning_rate, 
                                                                                scaling_factor=scaling_factor)

    batcher = SyntheticDataBatcher(PP = PP, vae_model=vae_model, n_samples=sinthetic_samples_by_class, 
                                    seq_length = x_train.size(-1), prior=prior)

    priorization = True
    for epoch in range(epochs):
        print(opt_method, create_samples, harder_samples)
        if opt_method=='twolosses' and create_samples and harder_samples: 
            if epoch>10 and priorization:

                ranking, _ = get_dict_class_priorization(model, train_dataloader, ranking_method=ranking_method)
                dict_priorization = {}
                ranking_penalization = 2
                
                for o in ranking:
                    dict_priorization[label_encoder[o]] =  int(sinthetic_samples_by_class*ranking_penalization)
                    if ranking_penalization>0.5:
                        ranking_penalization = ranking_penalization/2

                synthetic_data_loader = batcher.create_synthetic_batch(b=beta_actual, 
                                                                    wandb_active=wandb_active, 
                                                                    samples_dict = dict_priorization, 
                                                                    oversampling = True, n_oversampling=n_oversampling)
            else: 
                synthetic_data_loader = batcher.create_synthetic_batch(b=beta_actual, 
                                                    wandb_active=wandb_active, 
                                                    samples_dict = None, oversampling = True, n_oversampling=n_oversampling)

            beta_actual = beta_actual*beta_decay_factor
            harder_samples = False
        elif  opt_method=='twolosses' and create_samples: 
            print("Using available synthetic data")
        else:
            print("Skipping synthetic sample creation")
            synthetic_data_loader = None

        running_loss, model, val_loss = train_one_epoch_alternative(model, criterion, optimizer1, train_dataloader, val_dataloader, device, 
                                        mode = opt_method, 
                                        criterion_2= criterion_synthetic_samples, 
                                        dataloader_2 = synthetic_data_loader,
                                        optimizer_2 = optimizer2, locked_masks2 = locked_masks2, 
                                        locked_masks = locked_masks, repetitions = repetitions)
                                 
        _, accuracy_val, f1_val = evaluate_dataloader_weighted_metrics(model, val_dataloader, criterion, device)
        _, accuracy_train, f1_train = evaluate_dataloader_weighted_metrics(model, train_dataloader, criterion, device)

        try:
            synthetic_loss, accuracy_train_synthetic, f1_synthetic =  evaluate_dataloader_weighted_metrics(model, synthetic_data_loader, 
                                                                        criterion_synthetic_samples, device)
            condition1 = (accuracy_train_synthetic>threshold_acc_synthetic)
            condition3 = (counter > 15) 
            condition4 = (counter > 2)
            
            if condition4: 
                if condition1 or condition3:
                    harder_samples = True
                    counter = 0
                else:
                    counter = counter + 1
            else: 
                counter = counter + 1
        except Exception as error:
            print(error) 
            synthetic_loss=0
            accuracy_train_synthetic=0
            f1_synthetic = 0
            print('Synthetic data loader is: ', synthetic_data_loader)

        train_loss_values.append(running_loss)
        val_loss_values.append(val_loss)
        train_accuracy_values.append(accuracy_train)
        val_accuracy_values.append(accuracy_val)
        
        if opt_method=='twolosses':
            weighted_f1 = f1_synthetic*alpha + f1_val*(1-alpha)
        else: 
            weighted_f1 = f1_val
        # Early stopping criteria
        if  (weighted_f1 > best_f1_val):
            best_f1_val = weighted_f1
            best_model = model.state_dict()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count >= patience:
            print(f"Stopping early after {epoch + 1} epochs")
            break

        if wandb_active:
            wandb.log({'epoch': epoch, 'loss': running_loss, 'accuracy_train':accuracy_train, 
                        'synthetic_loss':synthetic_loss, 'acc_synthetic_samples': accuracy_train_synthetic, 
                        'val_loss': val_loss, 'val_accu': accuracy_val, 'f1_val': f1_val, 
                        'f1_train': f1_train, 'f1_synthetic': f1_synthetic, 'weighted_f1': weighted_f1})

        print('epoch:', epoch, ' loss:', running_loss, ' acc_train:', accuracy_train,  ' f1_train:', f1_train,
                                ' synth_loss:', synthetic_loss, ' acc_synth:', accuracy_train_synthetic, ' f1_synthetic:', f1_synthetic, 
                                ' val_loss', val_loss, ' val_accu:', accuracy_val, ' val_f1:', f1_val)
    
    # Post-training tasks
    model.load_state_dict(best_model)
    #plot_training(range(len(train_loss_values)), train_loss_values, val_loss_values, train_accuracy_values, val_accuracy_values)
    _, accuracy_test, f1_test = evaluate_dataloader_weighted_metrics(model, test_dataloader, criterion, device)
    _, accuracy_train, f1_train = evaluate_dataloader_weighted_metrics(model, train_dataloader, criterion, device)
    _, accuracy_val, f1_val = evaluate_dataloader_weighted_metrics(model, val_dataloader, criterion, device)
    if opt_method == 'twolosses':
        _, accuracy_train_synthetic, f1_synthetic = evaluate_dataloader_weighted_metrics(model, synthetic_data_loader, criterion, device)

    if wandb_active:
        weighted_f1 = f1_synthetic*alpha + f1_val*(1-alpha)
        wandb.log({'epoch': epoch, 'loss': running_loss, 'accuracy_train':accuracy_train, 
                    'synthetic_loss':synthetic_loss, 'acc_synthetic_samples': accuracy_train_synthetic, 
                    'val_loss': val_loss, 'val_accu': accuracy_val, 'f1_val': f1_val, 'f1_train': f1_train,
                     'f1_synthetic': f1_synthetic, 'weighted_f1': weighted_f1})

    if wandb_active: 
        wandb.config.acc_test =  accuracy_test
        wandb.config.f1_test =  f1_test
    # Using the function
    _ = evaluate_and_plot_cm_from_dataloader(model, train_dataloader, label_encoder, 'Confusion Matrix - Training set')
    _ = evaluate_and_plot_cm_from_dataloader(model, val_dataloader, label_encoder, 'Confusion Matrix - Validation set')
    _ = evaluate_and_plot_cm_from_dataloader(model, test_dataloader, label_encoder, 'Confusion Matrix - Testing set')
    if opt_method == 'twolosses':
        _ = evaluate_and_plot_cm_from_dataloader(model, synthetic_data_loader, label_encoder, 'Confusion Matrix - Synthetic') 