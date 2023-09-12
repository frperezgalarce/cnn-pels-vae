
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
import src.utils as utils

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

def count_subclasses(star_type_data):
    # Exclude the 'CompleteName' key and count the remaining keys as subclasses.
    # This is because the main class key is also used as a subclass key.
    return len([key for key in star_type_data.keys() if key != 'CompleteName'])
 
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


def construct_model_name(star_class, priors, PP, base_path=PATH_MODELS):
    """Construct a model name given parameters."""
    file_name =  base_path + 'bgm_model_' + str(star_class) + '_priors_' + str(priors) + '_PP_' + str(PP) + '.pkl'
    return base_path + 'bgm_model_' + str(star_class) + '_priors_' + str(priors) + '_PP_' + str(PP) + '.pkl'


def attempt_sample_load(model_name: str, 
                        sampler: 'YourSamplerType', 
                        n_samples: int = 16) -> Tuple[Union[np.ndarray, None], bool]:
    """
    Attempts to load samples given a model name.
    
    Parameters:
        model_name (str): The name or ID of the model to load samples from.
        sampler (YourSamplerType): The sampler object for generating or loading samples.
        n_samples (int, optional): The number of samples to load. Defaults to 16.
        
    Returns:
        tuple: A tuple containing the loaded samples and a success flag.
        
    Raises:
        Exception: Raises an exception if samples cannot be loaded.
    """
    try:
        samples = sampler.modify_and_sample(model_name, n_samples=n_samples)
        return samples, True
    except Exception as e:
        raise Exception(f"Failed to load samples from model {model_name}. Error: {str(e)}")


def create_synthetic_batch(mean_prior_dict: dict, 
                           priors: bool = True, 
                           PP: List[int] = [], 
                           vae_model: Optional[str] = None, 
                           n_samples: int = 16) -> DataLoader:
    """
    Creates a synthetic batch of data using VAEs and pre-trained GMMs.
    
    Parameters:
        mean_prior_dict (dict): Dictionary containing the mean priors for each star class.
        priors (bool, optional): Flag to indicate if priors should be used. Default is True.
        PP (list, optional): List of periods. Default is an empty list.
        vae_model (str, optional): Name or ID of the VAE model to be used. Default is None.
        n_samples (int, optional): Number of samples to generate for each class. Default is 16.
        
    Returns:
        DataLoader: DataLoader containing the synthetic data batch.
    """

    print('#'*50)
    print('CREATING BATCH WITH SYNTHETIC SAMPLES')

    lb: List[str] = []
    onehot: np.ndarray = np.empty((0, len(CLASSES)), dtype=np.float32)

    for star_class in CLASSES:
        print('------- sampling ' +star_class+'---------')
        lb += [star_class] * n_samples
        position = 2  # For demonstration, placing 'one' at index 2 for all K vectors
        one_hot_vector = np.zeros(len(CLASSES), dtype=np.float32)
        one_hot_vector[position] = 1
        
        # Replicate the one-hot encoder K times
        replicated_one_hot_vectors = np.tile(one_hot_vector, (n_samples, 1))
        
        # Vertically stack the new one-hot vectors with the existing array
        onehot = np.vstack((onehot, replicated_one_hot_vectors))

        components = count_subclasses(mean_prior_dict['StarTypes'][star_class])
        print(star_class +' includes '+ str(components) +' components ')
        sampler: mgmm.ModifiedGaussianSampler = mgmm.ModifiedGaussianSampler(b=1.0, components=components, features=PP)
        model_name = construct_model_name(star_class, priors, len(PP))
        samples, error = attempt_sample_load(model_name, sampler)

        # If we have priors and failed to load the model, try with priors=False
        if priors and samples is None:
            model_name = construct_model_name(star_class, False, len(PP))
            samples, error = attempt_sample_load(model_name, sampler, n_samples=n_samples)
        
        # If still not loaded, raise an error
        if samples is None:
            raise ValueError("The model can't be loaded." + str(error))

        if 'all_classes_samples' in locals() and all_classes_samples is not None: 
            all_classes_samples = np.vstack((samples, all_classes_samples))
        else: 
            all_classes_samples = samples

    print(all_classes_samples.shape)
    print('cuda: ', torch.cuda.is_available())
    print('model: ', vae_model)

    times = [i/600 for i in range(600)]
    times = np.tile(times, (128, 1))

    pp = all_classes_samples
    columns = ['Period', 'teff_val', '[Fe/H]_J95', 'abs_Gmag', 'radius_val', 'logg']
    index_period = columns.index('Period')

    mu_ = reg.process_regressors(config_file, phys2=columns, samples= pp, 
                                        from_vae=False, train_rf=False)
    onehot = np.array(onehot)  
    lb = np.array(lb)  
    times = np.array(times)  

    mu_ = torch.from_numpy(mu_).to(device)
    onehot = torch.from_numpy(onehot).to(device)
    pp = torch.from_numpy(pp).to(device)
    times = torch.from_numpy(times).to(device)
    times = times.to(dtype=torch.float32)

    vae, _ = load_model_list(ID=vae_model, device=device)
    
    xhat_mu = vae.decoder(mu_, times, label=onehot, phy=pp)
    xhat_mu = torch.cat([times.unsqueeze(-1), xhat_mu], dim=-1).cpu().detach().numpy()
    lc_reverted = utils.revert_light_curve(pp[:,index_period], xhat_mu, classes = lb)
    print('saving data ', lc_reverted.shape)

    labels = np.random.choice([0, 1, 2], size=128)

    utils.save_arrays_to_folder(lc_reverted, labels , PATH_DATA)
    numpy_array_x = np.load(PATH_DATA+'/x_batch_pelsvae.npy', allow_pickle=True)
    numpy_array_y = np.load(PATH_DATA+'/y_batch_pelsvae.npy', allow_pickle=True)

    print(numpy_array_x.shape)
    print(numpy_array_y.shape)
    synth_data = move_data_to_device((numpy_array_x, numpy_array_y), device)
    synthetic_dataset = TensorDataset(*synth_data)
    train_dataloader = DataLoader(synthetic_dataset, batch_size=64, shuffle=True)

    return train_dataloader
    
def move_data_to_device(data, device):
    return tuple(torch.tensor(d).to(device) if isinstance(d, np.ndarray) else d.to(device) for d in data)

def evaluate_and_plot_cm(model, x_data, y_data, label_encoder, title):
    outputs = model(x_data)
    _, predicted = torch.max(outputs.data.cpu(), 1)
    cm = confusion_matrix(y_data.cpu(), predicted.cpu(), normalize='true')
    plot_cm(cm, label_encoder.classes_, title=title)
    export_recall_latex(y_data.cpu(), predicted, label_encoder)
    return cm

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

def evaluate(model, data, criterion, device):
    inputs, labels = data
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        predicted = torch.max(outputs, 1)[1]
        accuracy = (predicted == labels).sum().item() / len(labels)
    return loss, accuracy


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
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Calculate predictions
            predicted = torch.max(outputs, 1)[1]
            
            # Update statistics
            total_loss += loss.item() * len(labels)
            total_correct += (predicted == labels).sum().item()
            total_samples += len(labels)

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    
    model.train()  # Set the model back to training mode
    
    return avg_loss, avg_accuracy

def create_optimizers(model: nn.Module,
                      kk_conv1: int, 
                      kk_conv2: int, 
                      kk_fc1: int, 
                      kk_fc2: int) -> Tuple[optim.Optimizer, optim.Optimizer]:
    
    params_group2 = [
        {"params": [model.module.conv1.weight[:kk_conv1].detach().requires_grad_(), model.module.conv1.bias[:kk_conv1].detach().requires_grad_()]},
        {"params": [model.module.conv2.weight[:kk_conv2].detach().requires_grad_(), model.module.conv2.bias[:kk_conv2].detach().requires_grad_()]},
        {"params": [model.module.fc1.weight[:kk_fc1, :].detach().requires_grad_(), model.module.fc1.bias[:kk_fc1].detach().requires_grad_()]},
        {"params": [model.module.fc2.weight[:kk_fc2, :].detach().requires_grad_(), model.module.fc2.bias[:kk_fc2].detach().requires_grad_()]},
    ]

    params_group1 = [
        {"params": [model.module.conv1.weight[kk_conv1:].detach().requires_grad_(), model.module.conv1.bias[kk_conv1:].detach().requires_grad_()]},
        {"params": [model.module.conv2.weight[kk_conv2:].detach().requires_grad_(), model.module.conv2.bias[kk_conv2:].detach().requires_grad_()]},
        {"params": [model.module.fc1.weight[kk_fc1:, :].detach().requires_grad_(), model.module.fc1.bias[kk_fc1:].detach().requires_grad_()]},
        {"params": [model.module.fc2.weight[kk_fc2:, :].detach().requires_grad_(), model.module.fc2.bias[kk_fc2:].detach().requires_grad_()]},
    ]
    
    optimizer1 = optim.Adam(params_group1, lr=0.001)
    optimizer2 = optim.Adam(params_group2, lr=0.001)
    
    return optimizer1, optimizer2



def run_cnn(create_samples: Any, mode_running: str = 'load', mean_prior_dict: Dict = None, 
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
    #wandb.init(project='train-classsifier', entity='fjperez10')
    print('#'*50)
    print('TRAINING CNN')
    print('#'*50)
    
    device = setup_environment()
    

    print('------ Data loading -------------------')
    print('mode: ', mode_running)
    x_train, x_test, y_train, y_test, x_val, y_val, label_encoder = get_data(sample_size=nn_config['data']['sample_size'], mode=mode_running)
    classes = np.unique(y_train.numpy())
    num_classes = len(classes)
    model = setup_model(num_classes, device)
    class_weights = compute_class_weight('balanced', classes, y_train.numpy())
    class_weights = torch.tensor(class_weights).to(device, dtype=x_train.dtype)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
   #twolosses, oneloss

    if opt_method == 'twolosses':
        print('Using mode: two masks')
        kk_conv1 = 10  
        kk_conv2 = 10   
        kk_fc1 = 30    
        kk_fc2 = 30  
        optimizer1, optimizer2 = create_optimizers(model, kk_conv1= kk_conv1, 
                                                    kk_conv2= kk_conv2, 
                                                    kk_fc1= kk_fc1, 
                                                    kk_fc2= kk_fc2)
    else: 
        print('Using mode: classic bapckpropagation')
        optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer2 = None

    print(x_train.shape)
    print(y_train.shape)

    training_data = move_data_to_device((x_train, y_train), device)
    val_data = move_data_to_device((x_val, y_val), device)
    #test_data = move_data_to_device((x_test, y_test), device)


    batch_size = nn_config['training']['batch_size']
    train_dataset = TensorDataset(*training_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # For validation data
    val_dataset = TensorDataset(*val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Typically we don't shuffle validation data

    # For test data
    #test_dataset = TensorDataset(*test_data)
    #test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Typically we don't shuffle test data

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
        print(f"Starting epoch {epoch+1}/{epochs}")
        if create_samples:
            print("Creating synthetic samples")
            synthetic_data_loader = create_synthetic_batch(mean_prior_dict, priors=False, PP = PP, vae_model=vae_model) #TODO:check
        else:
            print("Skipping synthetic sample creation")
            synthetic_data_loader = None

        running_loss = train_one_epoch(model, criterion, optimizer1, train_dataloader, device, 
                                        mode = opt_method, 
                                        criterion_2= criterion, 
                                        dataloader_2 = synthetic_data_loader,
                                        optimizer_2 = optimizer2)

        #val_loss, accuracy_val = evaluate(model, val_data, criterion, device)
        #_, accuracy_train = evaluate(model, training_data, criterion, device)

        val_loss, accuracy_val = evaluate_dataloader(model, val_dataloader, criterion, device)
        _, accuracy_train = evaluate_dataloader(model, train_dataloader, criterion, device)

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

        #wandb.log({'epoch': epoch, 'loss': running_loss, 'val_loss': val_loss, 'val_accu': accuracy_val})

    # Post-training tasks
    model.load_state_dict(best_model)
    plot_training(range(len(train_loss_values)), train_loss_values, val_loss_values, train_accuracy_values, val_accuracy_values)

    # Using the function
    _ = evaluate_and_plot_cm(model, x_train, y_train, label_encoder, 'Confusion Matrix - Training set')
    _ = evaluate_and_plot_cm(model, x_val, y_val, label_encoder, 'Confusion Matrix - Validation set')
    _ = evaluate_and_plot_cm(model, x_test, y_test, label_encoder, 'Confusion Matrix - Testing set')