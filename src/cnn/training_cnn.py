
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module  # Use nn.Module instead of _Loss
from src.utils import *
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Any, Dict, List
import src.utils as utils
import torch.optim as optim
from sklearn.metrics import f1_score

def setup_torch_environment() -> torch.device:    
    print('#'*50)
    print('TRAINING CNN')
    print('#'*50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(0)  # Set the GPU device index
    print('CUDA active:', torch.cuda.is_available())
    return device

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
    
    cm = confusion_matrix(all_y_data, all_predicted, normalize='true')
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

    elif method == 'proportion':
        m = confusion_matrix
        m = m / m.sum()
        m_copy = m.copy()
        np.fill_diagonal(m_copy, 0)
        proportions = np.sum(m_copy, axis=1)  
        class_ranking = np.argsort(proportions)[::-1]
        return class_ranking, proportions[class_ranking]
        
    elif method == 'max_pairwise_confusion':
        m = confusion_matrix + confusion_matrix.T 
        print(m)
        class_ranking = []
        counter = 0
        while (len(class_ranking)<m.shape[0]) and (counter < m.shape[0]):
            counter = counter + 1
            if verbose:
                print(find_argmax_off_diagonal(m))
            (a,b) = find_argmax_off_diagonal(m)
            if verbose:
                print(a,b)
                print(m)
                print(class_ranking)
            m[a,b] = 0
            m[b,a] = 0
            if a not in class_ranking:
                class_ranking.append(a)
            if b not in class_ranking:
                class_ranking.append(b)

        class_ranking = utils.remove_duplicates(class_ranking)
        for i in range(m.shape[0]): 
            if i not in class_ranking: 
                class_ranking.append(i)
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
    val_loss = 0
    if mode=='oneloss':
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, labels_indices = torch.max(labels, 1)  # Convert one-hot to indices
            loss = criterion(outputs, labels_indices)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=6.0, norm_type=2)
            optimizer.step()
            running_loss += loss.item()
        val_loss, _, _ = evaluate_dataloader(model, val_dataloader, criterion, device)

        return running_loss, model, val_loss
    
    elif mode=='twolosses':
        if criterion_2 is None or dataloader_2 is None or optimizer_2 is None:
            raise ValueError("For 'twolosses' mode, criterion_2, dataloader_2, and optimizer_2 must be provided.")     
        
        running_loss_prior = 0.0 
        for _ in range(repetitions):
            for inputs_2, labels_2 in dataloader_2:
                inputs, labels = inputs_2.to(device), labels_2.to(device)
                inputs = inputs.float()
                optimizer_2.zero_grad()
                outputs = model(inputs)
                _, labels_indices = torch.max(labels, 1)  # Convert one-hot to indices
                loss_prior = criterion_2(outputs, labels_indices)
                loss_prior.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=6.0, norm_type=2)
                for name, param in model.named_parameters():
                    if name in locked_masks:
                        mask = locked_masks[name].float().to(param.grad.data.device)
                        param.grad.data *= (mask == 0).float()
                        param.grad.data += mask * param.grad.data.clone()
                optimizer_2.step()
                running_loss_prior += loss_prior.item() 
        
        
        running_loss = 0.0
        for inputs, labels in dataloader:
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
    f1_score_val = f1_score(all_labels, all_predicted, average='macro')
    
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
    
    print('EPS: ', EPS)
    for name, param in model.named_parameters():
        if ("conv1" in name) and ("weight" in name):
            quantile = utils.quantile(torch.abs(param.mean(dim=[1, 2])), EPS)
            mask_value = (torch.abs(param.mean(dim=[1, 2])) > quantile).float()
            mask = mask_value.view(-1, 1, 1).repeat(1, param.shape[1], param.shape[2])
            locked_masks[name] = mask
            conv1_bias_mask = mask_value  # Save for bias
        elif ("conv2" in name) and ("weight" in name):
            quantile = utils.quantile(torch.abs(param.mean(dim=[1, 2])), EPS)
            mask_value = (torch.abs(param.mean(dim=[1, 2])) > quantile).float()
            mask = mask_value.view(-1, 1, 1).repeat(1, param.shape[1], param.shape[2])
            locked_masks[name] = mask
            conv2_bias_mask = mask_value  # Save for bias
        elif "conv3" in name and "weight" in name:
            quantile = utils.quantile(torch.abs(param.mean(dim=[1, 2])), EPS)
            mask_value = (torch.abs(param.mean(dim=[1, 2])) > quantile).float()
            mask = mask_value.view(-1, 1, 1).repeat(1, param.shape[1], param.shape[2])
            locked_masks[name] = mask
            conv3_bias_mask = mask_value  # Save for bias
        elif "conv4" in name and "weight" in name:
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
            quantile = utils.flat_and_quantile(param, EPS).item()
            mask = (param > quantile).float()
            locked_masks[name] = mask
        elif "fc2" in name and "weight" in name:
            quantile = utils.flat_and_quantile(param, EPS).item()
            mask = (param > quantile).float()
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

def initialize_optimizers(model, nn_config_dict = None):
    """
    Initializes the optimizers based on the specified optimization method.
    Args:
    - model: The model for which the optimizer needs to be created.
    - nn_config (dict): Dictionary with parameters

    Returns:
    - optimizer1: The primary optimizer.
    - optimizer2: The secondary optimizer (only for 'twolosses'). Returns None for 'oneloss'.
    """
    opt_method = nn_config['training']['opt_method']
    eps = nn_config['training']['EPS'] 
    base_learning_rate = nn_config['training']['base_learning_rate']
    scaling_factor = nn_config['training']['scaling_factor']
    layers = nn_config['training']['layers']

    if opt_method == 'twolosses':
        print('Using mode: two masks')
        locked_masks, locked_masks2 = initialize_masks(model, EPS=eps, layers=layers) 
        optimizer1 = optim.Adam(model.parameters(), lr=base_learning_rate)
        secondary_lr = scaling_factor * base_learning_rate
        optimizer2 = optim.Adam(model.parameters(), lr= secondary_lr)
    elif opt_method == 'oneloss':
        print('Using mode: classic backpropagation')
        optimizer1 = optim.Adam(model.parameters(), lr=base_learning_rate)
        optimizer2 = None
        locked_masks, locked_masks2 = None, None
    else:
        raise ValueError(f"Unsupported optimization method: {opt_method}.")
    
    return optimizer1, optimizer2, locked_masks, locked_masks2

