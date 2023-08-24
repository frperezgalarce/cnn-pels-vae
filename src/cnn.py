
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
import src.sampler.create_lc as creator

with open('src/nn_config.yaml', 'r') as file:
    nn_config = yaml.safe_load(file)


with open('src/paths.yaml', 'r') as file:
    YAML_FILE: Dict[str, Any] = yaml.safe_load(file)

PATHS: Dict[str, str] = YAML_FILE['paths']
PATH_PRIOS: str = PATHS['PATH_PRIOS']
PATH_MODELS: str = PATHS['PATH_MODELS']
CLASSES: List[str] = ['CEP']
mean_prior_dict: Dict[str, Any] = load_yaml_priors(PATH_PRIOS)


with open('src/regressor.yaml', 'r') as file:
    config_file: Dict[str, Any] = yaml.safe_load(file)

vae_model: str = config_file['model_parameters']['ID']

#vae_model: str = '1pjeearx'#'20twxmei' trained using TPM using GAIA3 ... using 5 PP 1pjeearx

# Define the number of classes
def print_grad_norm(grad: torch.Tensor) -> None:
    if (param.grad is not None) and torch.isnan(param.grad).any():
                print(f"NaN value in gradient of {grad}")

# Define the 1D CNN model
class CNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=128, kernel_size=6, stride=4, groups=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=6, stride=4, groups=1)
        self.fc1 = nn.Linear(64*5, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        if verbose:
            print('forward')
        x = self.conv1(x)
        if verbose: 
            print('After conv1: ', x.size())
        x = torch.tanh(x)
        if verbose: 
            print('After relu: ', x.size())
        x = self.conv2(x)
        if verbose:
            print('After conv2: ', x.size())
        x = torch.tanh(x)
        if verbose:
            print('After relu: ', x.size())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x

def setup_environment() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(0)  # Set the GPU device index
    print('CUDA active:', torch.cuda.is_available())
    return device


def setup_model(num_classes: int, device: torch.device) -> nn.Module:
    model = CNN(num_classes=num_classes)
    if torch.cuda.is_available():
        model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    return model

def create_synthetic_batch(mean_prior_dict): 
    #print(len(mean_prior_dict['StarTypes'][CLASSES[0]].keys())-1)
    for star_class in CLASSES:
        components: int = 3 # len(mean_prior_dict['StarTypes'][CLASSES[0]].keys())-1 TODO: check number of components
        sampler: mgmm.ModifiedGaussianSampler = mgmm.ModifiedGaussianSampler(b=0.5, components=components)
        model_name: str = PATH_MODELS+'bgm_model_'+str(star_class)+'.pkl'
        samples: np.ndarray = sampler.modify_and_sample(model_name)
        z_hat: Any = reg.main(samples, vae_model, train_rf=True)
        samples, z_hat = None, None
        creator.main(samples, z_hat) #TODO: check error


def move_data_to_device(data: Tuple, device: torch.device) -> Tuple:
    return tuple(d.to(device) for d in data)

def train_one_epoch(
        model: torch.nn.Module, 
        criterion: Module,  # Here
        optimizer: Optimizer, 
        dataloader: DataLoader, 
        device: torch.device, 
        mode: str = 'oneloss', 
        criterion_2: Optional[Module] = None,  # And here
        dataloader_2: Optional[DataLoader] = None, 
        optimizer_2: Optional[Optimizer] = None
    ) -> float:    
    
    running_loss = 0.0
    if mode=='oneloss':
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    elif mode=='twolosses':
        if criterion_2 is None or dataloader_2 is None or optimizer_2 is None:
            raise ValueError("For 'twolosses' mode, criterion_2, dataloader_2, and optimizer_2 must be provided.") 

        for (inputs_1, labels_1), (inputs_2, labels_2) in zip(dataloader, dataloader_2):
            
            # Forward pass for dataloader 1
            inputs_1, labels_1 = inputs_1.to(device), labels_1.to(device)
            outputs_1 = model(inputs_1)
            loss_1 = criterion(outputs_1, labels_1)
            
            # Forward pass for dataloader 2
            inputs_2, labels_2 = inputs_2.to(device), labels_2.to(device)
            outputs_2 = model(inputs_2)
            loss_2 = criterion_2(outputs_2, labels_2)
            
            # Backward pass and optimization for parameters designated to loss_1
            optimizer.zero_grad()
            loss_1.backward(retain_graph=True)
            optimizer.step()

            # Backward pass and optimization for parameters designated to loss_2
            optimizer_2.zero_grad()
            loss_2.backward()
            optimizer_2.step()

            running_loss += loss_1.item() + loss_2.item()

    return running_loss

    return running_loss

def evaluate(model, data, criterion, device):
    inputs, labels = data
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        predicted = torch.max(outputs, 1)[1]
        accuracy = (predicted == labels).sum().item() / len(labels)
    return loss, accuracy

def run_cnn(create_samples: Any, mode_running: str = 'load', mean_prior_dict) -> None:  # Adjust the type of create_samples if known
    # Initialization
    wandb.init(project='cnn-pelsvae', entity='fjperez10')
    device = setup_environment()
    x_train, x_test, y_train, y_test, x_val, y_val, label_encoder = get_data(sample_size=nn_config['data']['sample_size'], mode=mode_running)
    classes = np.unique(y_train.numpy())
    num_classes = len(classes)
    model = setup_model(num_classes, device)
    class_weights = compute_class_weight('balanced', classes, y_train.numpy())
    class_weights = torch.tensor(class_weights).to(device, dtype=x_train.dtype)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    #TODO: create method to define to optimizers and masks, this should consider a cnn.
    optimizer = optim.Adam(model.parameters(), lr=nn_config['training']['lr'], weight_decay=nn_config['training']['weight_decay'])
    
    training_data = move_data_to_device((x_train, y_train), device)
    val_data = move_data_to_device((x_val, y_val), device)
    test_data = move_data_to_device((x_test, y_test), device)
    train_dataset = TensorDataset(*training_data)
    train_dataloader = DataLoader(train_dataset, batch_size=nn_config['training']['batch_size'], shuffle=True)

    # Main training loop
    best_val_loss = float('inf')
    no_improvement_count = 0
    train_loss_values = []
    val_loss_values = []
    train_accuracy_values = []
    val_accuracy_values = []
    epochs = nn_config['training']['epochs']
    patience =  nn_config['training']['patience']

    for epoch in range(epochs):
        if create_samples:
            create_synthetic_batch(mean_prior_dict) 

        running_loss = train_one_epoch(model, criterion, optimizer, train_dataloader, device)

        val_loss, accuracy_val = evaluate(model, val_data, criterion, device)
        _, accuracy_train = evaluate(model, training_data, criterion, device)

        train_loss_values.append(running_loss)
        val_loss_values.append(val_loss.item())
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

        wandb.log({'epoch': epoch, 'loss': running_loss, 'val_loss': val_loss.item(), 'val_accu': accuracy_val})

    # Post-training tasks
    model.load_state_dict(best_model)
    plot_training(range(len(train_loss_values)), train_loss_values, val_loss_values, train_accuracy_values, val_accuracy_values)

    train_outputs = model(x_train)
    _, predicted_train = torch.max(train_outputs.data.cpu(), 1)
    cm_train = confusion_matrix(y_train.cpu(), predicted_train.cpu(), normalize='true')
    plot_cm(cm_train, label_encoder.classes_, title='Confusion Matrix - Training set')

    test_outputs = model(x_test)
    _, predicted_test = torch.max(test_outputs.data.cpu(), 1)
    cm_test = confusion_matrix(y_test.cpu(), predicted_test.cpu(), normalize='true')
    plot_cm(cm_test, label_encoder.classes_, title='Confusion Matrix - Testing set')
    export_recall_latex(y_train.cpu(), predicted_train, label_encoder)
    export_recall_latex(y_test.cpu(), predicted_test, label_encoder)