
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
import src.gmm.modifiedgmm as mgmm
import src.sampler.fit_regressor as reg
from src.sampler.getbatch import SyntheticDataBatcher
import src.utils as utils
import pickle 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.optim.lr_scheduler import ExponentialLR

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
gpu: bool = True # fail when true is selected

device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")

class CNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=60, kernel_size=6, stride=1)
        self.bn1 = nn.BatchNorm1d(60)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(in_channels=60, out_channels=30, kernel_size=6, stride=1)
        self.bn2 = nn.BatchNorm1d(30)
        self.pool2 = nn.MaxPool1d(2)
        
        self.fc1 = nn.Linear(2130, 200)  # Adjust this number based on your actual output size
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(200, num_classes)
        self.dropout2 = nn.Dropout(0.2)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.dropout2(x)

        return F.log_softmax(x, dim=1)  # Optional softmax at the end


def setup_environment() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(0)  # Set the GPU device index
    print('CUDA active:', torch.cuda.is_available())
    return device

def setup_model(num_classes: int, device: torch.device, show_architecture: bool = True) -> nn.Module:

    print('----- model setup --------')
    model = CNN(num_classes=num_classes)
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
        cm = confusion_matrix(all_y_data, all_predicted, normalize=None)
    except ValueError as e:
        print(f"An error occurred: {e}")
        print(f"y_data shape: {len(all_y_data)}, predicted shape: {len(all_predicted)}")
        return None
    
    plot_cm(cm, label_encoder, title=title)
    #export_recall_latex(all_y_data, all_predicted, label_encoder)
    
    return cm

def train_one_epoch_alternative(
        model: torch.nn.Module, 
        criterion: Module,  # Here
        optimizer: Optimizer, 
        dataloader: DataLoader, 
        device: torch.device, 
        mode: str = 'oneloss', 
        criterion_2: Optional[Module] = None,  # And here
        dataloader_2: Optional[DataLoader] = None, 
        optimizer_2: Optional[Optimizer] = None, 
        locked_masks2 = None, 
        locked_masks = None
    ) -> float:    
    
    running_loss = 0.0
    num_batches = 0

    if mode=='oneloss':
        for inputs, labels in dataloader:
            num_batches += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, labels_indices = torch.max(labels, 1)  # Convert one-hot to indices
            loss = criterion(outputs, labels_indices)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()
            running_loss += loss.item()
        return running_loss, model
    
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
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            for name, param in model.named_parameters():
                if name in locked_masks2:
                    mask = locked_masks2[name].float().to(param.grad.data.device)
                    param.grad.data *= (mask == 0).float()
                    param.grad.data += mask * param.grad.data.clone()                                                    
            optimizer.step()
            running_loss += loss.item()   
        
        num_batches = 0
        for inputs_2, labels_2 in dataloader_2:
            num_batches += 1
            inputs, labels = inputs_2.to(device), labels_2.to(device)
            inputs = inputs.float()
            optimizer_2.zero_grad()
            outputs = model(inputs)
            _, labels_indices = torch.max(labels, 1)  # Convert one-hot to indices
            loss_prior = criterion_2(outputs, labels_indices)
            loss_prior.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            for name, param in model.named_parameters():
                if name in locked_masks:
                    mask = locked_masks[name].float().to(param.grad.data.device)
                    param.grad.data *= (mask == 0).float()
                    param.grad.data += mask * param.grad.data.clone()
            
            optimizer_2.step()
            running_loss_prior += loss_prior.item() 
        return running_loss, model
  
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
            total_loss += loss.item() * len(labels)
            total_correct += (predicted == labels_indices).sum().item()
            total_samples += len(labels_indices)

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    
    model.train()  # Set the model back to training mode
    
    return avg_loss, avg_accuracy

def initialize_masks(model, device='cuda', EPS=0.25):
    locked_masks = {}
    locked_masks2 = {}
    conv1_bias_mask = None
    conv2_bias_mask = None
    for name, param in model.named_parameters():
        if "conv1" in name and "weight" in name:
            print(utils.quantile(torch.abs(param.mean(dim=[1, 2])), EPS))
            quantile = utils.quantile(torch.abs(param.mean(dim=[1, 2])), EPS)
            mask_value = (torch.abs(param.mean(dim=[1, 2])) >= quantile).float()
            mask = mask_value.view(-1, 1, 1).repeat(1, param.shape[1], param.shape[2])
            locked_masks[name] = mask
            conv1_bias_mask = mask_value  # Save for bias
        elif "conv2" in name and "weight" in name:
            print(utils.quantile(torch.abs(param.mean(dim=[1, 2])), EPS))
            quantile = utils.quantile(torch.abs(param.mean(dim=[1, 2])), EPS)
            mask_value = (torch.abs(param.mean(dim=[1, 2])) >= quantile).float()
            mask = mask_value.view(-1, 1, 1).repeat(1, param.shape[1], param.shape[2])
            locked_masks[name] = mask
            conv2_bias_mask = mask_value  # Save for bias
        elif "fc1" in name and "weight" in name:
            print(utils.quantile(torch.abs(param.mean(dim=[1])), EPS))
            quantile = utils.quantile(torch.abs(param.mean(dim=[1])), EPS)
            mask_value = (torch.abs(param.mean(dim=1)) >= quantile).float()
            mask = mask_value.view(-1, 1).repeat(1, param.shape[1])
            locked_masks[name] = mask
        elif "fc2" in name and "weight" in name:
            print(utils.quantile(torch.abs(param.mean(dim=[1])), EPS))
            quantile = utils.quantile(torch.abs(param.mean(dim=[1])), EPS)
            mask_value = (torch.abs(param.mean(dim=1)) >= quantile).float()
            mask = mask_value.view(-1, 1).repeat(1, param.shape[1])
            locked_masks[name] = mask
        elif "conv1.bias" in name:
            locked_masks[name] = conv1_bias_mask
        elif "conv2.bias" in name:
            locked_masks[name] = conv2_bias_mask
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


def run_cnn(create_samples: Any, mean_prior_dict: Dict = None, 
            vae_model=None, PP=[], opt_method='twolosses') -> None:
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
    wandb.init(project='train-classsifier', entity='fjperez10')
    print('#'*50)
    print('TRAINING CNN')
    print('#'*50)
    
    device = setup_environment()
    

    print('------ Data loading -------------------')
    print('mode: ', nn_config['data']['mode_running'], nn_config['data']['sample_size'])
    x_train, x_test, y_train, y_test, x_val, y_val, label_encoder, y_train_labeled = get_data(nn_config['data']['sample_size'],
                                                                             nn_config['data']['mode_running'])
    classes = np.unique(y_train_labeled.numpy())
    num_classes = len(classes)
    print('num_classes: ', num_classes)
    print(x_train.size(-1))
    model = setup_model(num_classes, device)
    class_weights = compute_class_weight('balanced', np.unique(y_train_labeled.numpy()), y_train_labeled.numpy())
    class_weights = torch.tensor(class_weights).to(device, dtype=x_train.dtype)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    criterion_synthetic_samples = nn.CrossEntropyLoss()

    if opt_method == 'twolosses':
        print('Using mode: two masks')
        locked_masks, locked_masks2 = initialize_masks(model, EPS=0.5)
        learning_rate = 0.001
        scaling_lr = 0.5
        optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate)  
        optimizer2 = torch.optim.Adam(model.parameters(), lr=scaling_lr*learning_rate)
        #optimizer1 = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99)
        #optimizer2 = optim.RMSprop(model.parameters(), lr=scaling_lr*learning_rate, alpha=0.99)

        # ExponentialLR Schedulers
        #gamma = 0.95  # decay factor

        #scheduler1 = ExponentialLR(optimizer1, gamma=gamma)
        #scheduler2 = ExponentialLR(optimizer2, gamma=gamma)

    elif opt_method == 'oneloss':
        print('Using mode: classic backpropagation')
        optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer2 = None
    else:
        raise ValueError(f"Unsupported optimization method: {opt_method}. Supported methods are 'twolosses' and 'oneloss'.")
    training_data = utils.move_data_to_device((x_train, y_train), device)
    val_data = utils.move_data_to_device((x_val, y_val), device)
    testing_data = utils.move_data_to_device((x_test, y_test), device)


    batch_size = nn_config['training']['batch_size']
    train_dataset = TensorDataset(*training_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # For validation data
    val_dataset = TensorDataset(*val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Typically we don't shuffle validation data

        # For validation data
    test_dataset = TensorDataset(*testing_data)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Typically we don't shuffle validation data

    # Main training loop
    best_val_loss = float('inf')
    harder_samples = True
    threshold_acc_synthetic = 0.95
    no_improvement_count = 0
    train_loss_values = []
    val_loss_values = []
    train_accuracy_values = []
    val_accuracy_values = []
    epochs = nn_config['training']['epochs']
    patience =  nn_config['training']['patience']

    batcher = SyntheticDataBatcher(PP = PP, vae_model=vae_model, n_samples=256, seq_length = x_train.size(-1))
    for epoch in range(epochs):
        if opt_method=='twolosses' and create_samples and harder_samples: 
            synthetic_data_loader = batcher.create_synthetic_batch()
            harder_samples = False
        elif  opt_method=='twolosses' and create_samples: 
            print("Using available synthetic data")
            synthetic_data_loader = synthetic_data_loader
        else:
            print("Skipping synthetic sample creation")
            synthetic_data_loader = None

        running_loss, model = train_one_epoch_alternative(model, criterion, optimizer1, train_dataloader, device, 
                                        mode = opt_method, 
                                        criterion_2= criterion_synthetic_samples, 
                                        dataloader_2 = synthetic_data_loader,
                                        optimizer_2 = optimizer2, locked_masks2 = locked_masks2, 
                                        locked_masks = locked_masks)
                                 
        val_loss, accuracy_val = evaluate_dataloader(model, val_dataloader, criterion, device)
        _, accuracy_train = evaluate_dataloader(model, train_dataloader, criterion, device)

        try:
            synthetic_loss, accuracy_train_synthetic =  evaluate_dataloader(model, synthetic_data_loader, criterion_synthetic_samples, device)
            if accuracy_train_synthetic>threshold_acc_synthetic:
                harder_samples = True
        except Exception as error:
            print(error) 
            synthetic_loss=0
            accuracy_train_synthetic=0
            print('Sinthetic data loader is: ', synthetic_data_loader)

        train_loss_values.append(running_loss)
        val_loss_values.append(val_loss)
        train_accuracy_values.append(accuracy_train)
        val_accuracy_values.append(accuracy_val)

        # Early stopping criteria
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count >= patience:
            print(f"Stopping early after {epoch + 1} epochs")
            break

        wandb.log({'epoch': epoch, 'loss': running_loss, 'accuracy_train':accuracy_train, 'synthetic_loss':synthetic_loss, 
                'acc_synthetic_samples': accuracy_train_synthetic, 'val_loss': val_loss, 'val_accu': accuracy_val})
        print('epoch:', epoch, ' loss:', running_loss, ' acc_train:', accuracy_train, ' synth_loss:',synthetic_loss, 
                ' acc_synth:', accuracy_train_synthetic, ' val_loss', val_loss, ' val_accu:', accuracy_val)
    # Post-training tasks
    model.load_state_dict(best_model)
    plot_training(range(len(train_loss_values)), train_loss_values, val_loss_values, train_accuracy_values, val_accuracy_values)

    # Using the function
    _ = evaluate_and_plot_cm_from_dataloader(model, train_dataloader, label_encoder, 'Confusion Matrix - Training set')
    _ = evaluate_and_plot_cm_from_dataloader(model, val_dataloader, label_encoder, 'Confusion Matrix - Validation set')
    _ = evaluate_and_plot_cm_from_dataloader(model, test_dataloader, label_encoder, 'Confusion Matrix - Testing set')
    _ = evaluate_and_plot_cm_from_dataloader(model, synthetic_data_loader, label_encoder, 'Confusion Matrix - Synthetic') 