
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.utils import *
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import wandb

# Define the number of classes
def print_grad_norm(grad):
    if (param.grad is not None) and torch.isnan(param.grad).any():
                print(f"NaN value in gradient of {grad}")

def get_data_fake():
    # Generate some random data for training and validation
    x_train = np.random.rand(1000, 100, 2)
    y_train = np.random.randint(num_classes, size=(1000,))
    x_val = np.random.rand(200, 100, 2)
    y_val = np.random.randint(num_classes, size=(200,))

    # Convert the data to PyTorch tensors
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).long()

    # Permute the dimensions of the input tensor
    x_train = x_train.permute(0, 2, 1)
    x_val = x_val.permute(0, 2, 1)

    # Print the shapes of the data
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_val shape:", x_val.shape)
    print("y_val shape:", y_val.shape)

    return x_train, y_train, x_val, y_val

# Define the 1D CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=128, kernel_size=6, stride=4, groups=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=6, stride=4, groups=1)
        self.fc1 = nn.Linear(64*5, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x, verbose=False):
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

def run_cnn(create_samples, mode_running='load'):
    'mode_running: load or create, load take a training set and create considers a new dataset'
    #TODO: ensure good performance of get data method
    wandb.init(project='cnn-pelsvae', entity='fjperez10')
    # Define the input shape of the data
    light_curve_lenght = 100
    input_shape = (light_curve_lenght, 2)
    epochs = 10000
    # Define the early stopping criteria
    patience = 200
    min_delta = 0.00
    best_val_loss = float('inf')
    best_model = None
    no_improvement_count = 0
    batch_size = 256
    kernel_size=8
    stride=1
    out_channels=64
    in_size= out_channels*light_curve_lenght
    out_size = ((in_size - kernel_size)/stride) + 1
    print(out_size)

    x_train, x_test,  y_train, y_test, x_val, y_val, label_encoder = get_data(sample_size=400000, mode='create')

    label_encode = label_encoder.classes_


    classes = np.unique(y_train.numpy())
    num_classes = len(classes) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('cuda active: ', torch.cuda.is_available())

    # Create an instance of the CNN model
    model = CNN(num_classes=num_classes)

    # Define the class weights
    class_weights = compute_class_weight('balanced', classes, y_train.numpy())

    class_weights =  torch.tensor(class_weights)
    class_weights = class_weights.to(device, dtype=x_train.dtype)

    print(class_weights)
    # Define your loss function with class weights

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA device
        torch.cuda.set_device(0)  # Set the GPU device index

        # Move your model to the device
        model = model.to(device)

        # Move your data to the device
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        # Enable CUDA for computations
        torch.backends.cudnn.benchmark = True


    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Create a TensorDataset for your training data
    train_dataset = TensorDataset(x_train, y_train)

    # Create a DataLoader for your training data
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Enable CUDA for computations
    torch.backends.cudnn.benchmark = True
    train_loss_values = []
    val_loss_values = []
    train_accuracy_values = []
    val_accuracy_values = []

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader):
            # Move the batch to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

        # Evaluate the model on the validation set
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            predicted = torch.max(val_outputs, 1)[1]        
            accuracy_val = (predicted == y_val).sum().item() / len(y_val)

            train_outputs = model(x_train)
            train_loss = criterion(train_outputs, y_train)
            predicted = torch.max(train_outputs, 1)[1]        
            accuracy = (predicted == y_train).sum().item() / len(y_train)

            # Check if the validation loss has improved
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_model = model.state_dict()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Check if we should stop early
            if no_improvement_count >= patience:
                print(f"Stopping early after {epoch + 1} epochs")
                break
        
        print(val_loss.cpu().item())

        # Append the values to the lists
        train_loss_values.append(running_loss)
        val_loss_values.append(val_loss.cpu().item())
        train_accuracy_values.append(accuracy)
        val_accuracy_values.append(accuracy_val)
        # Plot the training and validation behavior
        epochs_range = range(0, epoch+1)

        print(f"Epoch {epoch + 1}, loss: {running_loss:.4f}, val_loss: {val_loss:.4f}, accuracy: {accuracy:.4f}, accuracy_val: {accuracy_val:.4f}")
        wandb.log({'epoch': epoch, 'loss': running_loss, 'val_loss':val_loss, 'val_accu': accuracy_val})

    # Load the best model state
    model.load_state_dict(best_model)
    model = model.to(device)

    plot_training(epochs_range, train_loss_values, val_loss_values,train_accuracy_values,  val_accuracy_values)

    # Move testing data to GPU if not already on GPU
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    train_outputs = model(x_train)
    _, predicted_train = torch.max(train_outputs.data.cpu(), 1)
    cm = confusion_matrix(y_train.cpu(), predicted_train.cpu(), normalize='true')
    plot_cm(cm, label_encode, title='Confusion Matrix - Training set')

    torch.cuda.empty_cache()

    test_outputs = model(x_test)
    _, predicted_test = torch.max(test_outputs.data.cpu(), 1)
    cm = confusion_matrix(y_test.cpu(), predicted_test.cpu(), normalize='true')
    plot_cm(cm, label_encode, title='Confusion Matrix - Testing set')

    torch.cuda.empty_cache()

    export_recall_latex(y_train.cpu(),  predicted_train.cpu(), label_encoder)

    export_recall_latex(y_test.cpu(), predicted_test.cpu(), label_encoder)