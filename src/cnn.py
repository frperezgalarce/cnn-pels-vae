
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
import src.utils as utils
import pickle 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=256, kernel_size=6, stride=4)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=6, stride=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.fc1 = nn.Linear(128*1, 100)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(100, num_classes)
        self.dropout2 = nn.Dropout(0.5)

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
        x = self.dropout1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.dropout2(x)

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

    with open(PATH_MODELS+'label_encoder_vae.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    with open('src/nn_config.yaml', 'r') as file:
        nn_config = yaml.safe_load(file)

    for star_class in list(nn_config['data']['classes']):
        print('------- sampling ' +star_class+'---------')
        lb += [star_class] * n_samples

        integer_encoded = label_encoder.transform(lb)
        n_values = len(label_encoder.classes_)
        onehot = np.eye(n_values)[integer_encoded]

        encoded_labels, _ = utils.transform_to_consecutive(integer_encoded, label_encoder)
        n_values = len(np.unique(encoded_labels))
        onehot_to_train = np.eye(n_values)[encoded_labels]


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



    print('cuda: ', torch.cuda.is_available())
    print('model: ', vae_model)

    times = [i/600 for i in range(600)]
    times = np.tile(times, (n_samples*len(list(nn_config['data']['classes'])), 1))
    

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

    #utils.get_time_sequence()
    #times2 = utils.get_time_from_period(period, phased_time,  example_sequence, sequence_length=600)


    xhat_mu = torch.cat([times.unsqueeze(-1), xhat_mu], dim=-1).cpu().detach().numpy()
    indices = np.random.choice(xhat_mu.shape[0], 24, replace=False)
    sampled_arrays = xhat_mu[indices, :, :]

    utils.plot_wall_lcs_sampling(sampled_arrays, sampled_arrays,  cls=lb[indices],  column_to_sensivity=index_period,
                            to_title = pp[indices], sensivity = 'Period', all_columns=columns, save=True) 

    lc_reverted = utils.revert_light_curve(pp[:,index_period], xhat_mu, classes = lb)

    reverted_sample_array = lc_reverted[indices, :, :]
    reverted_sample_array = np.swapaxes(reverted_sample_array, 2, 1)

    #utils.plot_wall_lcs_sampling(reverted_sample_array, reverted_sample_array,  cls=lb[indices], save=True,  column_to_sensivity=index_period,
    #                        to_title = pp[indices], sensivity = 'Period', all_columns=columns) 

    # Sort by time, which is assumed to be the first channel (axis=1, index=0)
    lc_reverted = np.sort(lc_reverted, axis=-1)

    # Calculate differences in sorted time and magnitude, axis=-1 for the last dimension (100 points)
    lc_reverted = np.diff(lc_reverted, axis=-1)

    #TODO: check oversampling, it does not work
    oversampling = False
    if oversampling:
        k = 4
        lc_reverted_samples = np.zeros((lc_reverted.shape[0]*k, 2, 100))
        one_hot_to_train_samples = np.zeros((onehot_to_train.shape[0]*k, onehot_to_train.shape[1]))
        print(onehot_to_train.shape)
        
        for i in range(lc_reverted.shape[0]):
            for j in range(k):  # 4 samples per light curve
                # Generate 100 random unique indices
                random_indices = np.random.choice(lc_reverted.shape[2], 100, replace=False)
                random_indices.sort()  # Optional: sort indices
            
                # Select 100 random points for the i-th light curve, j-th sample
                lc_reverted_samples[4*i + j, :, :] = lc_reverted[i, :, random_indices].T
                one_hot_to_train_samples[4*i + j, :] = onehot_to_train[i].T
        utils.save_arrays_to_folder(lc_reverted_samples, one_hot_to_train_samples , PATH_DATA)
    else: 
        utils.save_arrays_to_folder(lc_reverted[:,:,:100], onehot_to_train , PATH_DATA)

    numpy_array_x = np.load(PATH_DATA+'/x_batch_pelsvae.npy', allow_pickle=True)
    numpy_array_y = np.load(PATH_DATA+'/y_batch_pelsvae.npy', allow_pickle=True)

    synth_data = move_data_to_device((numpy_array_x, numpy_array_y), device)
    synthetic_dataset = TensorDataset(*synth_data)
    train_dataloader = DataLoader(synthetic_dataset, batch_size=16, shuffle=True)

    return train_dataloader
    
def move_data_to_device(data, device):
    return tuple(torch.tensor(d).to(device) if isinstance(d, np.ndarray) else d.to(device) for d in data)

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
        cm = confusion_matrix(all_y_data, all_predicted, normalize='true')
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
        return running_loss
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            for name, param in model.named_parameters():
                if name in locked_masks:
                    mask = locked_masks[name].float().to(param.grad.data.device)
                    param.grad.data *= (mask == 0).float()
                    param.grad.data += mask * param.grad.data.clone()
            
            optimizer_2.step()
            running_loss_prior += loss_prior.item() 
        return running_loss
  
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

def initialize_masks(model, EPS1 = 0.1): 
    locked_masks2 = {}
    locked_masks = {}
    for name, param in model.named_parameters():
        if "conv" in name and "weight" in name:
            mask_value = (torch.abs(param.mean(dim=[1, 2])) > EPS1).float().view(-1, 1, 1)
            mask = mask_value.repeat(1, param.shape[1], param.shape[2])
        elif "fc" in name and "weight" in name:
            mask = (torch.abs(param.mean(dim=1)) > EPS1).float().view(-1, 1).repeat(1, param.shape[1])
        elif name.endswith('bias') or "bn" in name:
            mask = torch.ones_like(param)
        else:
            continue
        locked_masks2[name] = mask
        if name.endswith('bias') or "bn" in name:
            mask_inv = torch.zeros_like(param)  # This will never be used because of the bias/bn condition
        else:
            mask_inv = 1 - mask

        locked_masks[name] = mask_inv

    # Move the masks to the same device as your model
    for name in locked_masks:
        locked_masks[name] = locked_masks[name].to(device='cuda').float()
        locked_masks2[name] = locked_masks2[name].to(device='cuda').float()
    
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
    model = setup_model(num_classes, device)
    
    class_weights = compute_class_weight('balanced', np.unique(y_train_labeled.numpy()), y_train_labeled.numpy())
    class_weights = torch.tensor(class_weights).to(device, dtype=x_train.dtype)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion_synthetic_samples = nn.CrossEntropyLoss()

    if opt_method == 'twolosses':
        print('Using mode: two masks')
        locked_masks, locked_masks2 = initialize_masks(model, EPS1 = 0.1)
        learning_rate = 0.001
        optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate)  
        optimizer2 = torch.optim.Adam(model.parameters(), lr=0.5*learning_rate)

    elif opt_method == 'oneloss':
        print('Using mode: classic backpropagation')
        optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer2 = None
    else:
        raise ValueError(f"Unsupported optimization method: {opt_method}. Supported methods are 'twolosses' and 'oneloss'.")


    training_data = move_data_to_device((x_train, y_train), device)
    val_data = move_data_to_device((x_val, y_val), device)
    testing_data = move_data_to_device((x_test, y_test), device)


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


    for epoch in range(epochs):
        if opt_method=='twolosses' and create_samples and harder_samples: 
            print("Creating synthetic samples")
            synthetic_data_loader = create_synthetic_batch(mean_prior_dict, priors=False, PP = PP, vae_model=vae_model) #TODO:check
            harder_samples = False
        elif  opt_method=='twolosses' and create_samples: 
            print("Using available synthetic data")
            synthetic_data_loader = synthetic_data_loader
        else:
            print("Skipping synthetic sample creation")
            synthetic_data_loader = None

        running_loss = train_one_epoch_alternative(model, criterion, optimizer1, train_dataloader, device, 
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