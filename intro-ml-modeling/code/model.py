import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import json

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Only take the output from the last timestep
        output = self.fc(lstm_out)
        output = self.softmax(output)
        return output

def build_model(input_size):
    """
    Build a simple feedforward neural network model.
    
    Parameters:
    - input_size (int): The number of input features.
    
    Returns:
    - model (Sequential): A PyTorch Sequential model.
    """
    model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    return model

def load_model(model, date_time):
    """
    Load model parameters.
    """
    model_path = os.path.join('models', f'{date_time}/model.pth')
    
    # Check if the model path exists
    if not os.path.exists(model_path):
        raise ValueError(f"No model found at {model_path}")

    # Move model to appropriate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the entire model state
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if 'model_state_dict' key is in the checkpoint. If not, it might be an old format.
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded through checkpoint")
    else:
        # Attempt to load the state dict directly
        try:
            model.load_state_dict(checkpoint)
            print("Model loaded directly")
        except RuntimeError as e:
            print(f"Failed to load model with error: {str(e)}")
    
    model = model.to(device)

    return model

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_model(model, X_train, y_train, X_val=None, y_val=None, learning_rate=0.001, num_epochs=100, patience=5, weight_decay=1e-4, batch_size=32):
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.apply(init_weights)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    
    # Convert numpy arrays to PyTorch tensors and move them to the device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    
    # Validation data
    if X_val is not None and y_val is not None:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    
    # Create DataLoader
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop initialization
    best_loss = np.inf
    patience_counter = 0
    
    # To store training and validation loss at each epoch
    loss_history = {'train': [], 'val': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0 
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history['train'].append(avg_epoch_loss)
        
        # Validation phase
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            model.train()
        else:
            val_loss = avg_epoch_loss  # If no validation data, use training data for early stopping
        
        loss_history['val'].append(val_loss)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Early stopping and best model checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0  
            best_model_wts = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Early stopping: stopping training as the loss has not improved in the last {patience} epochs')
            break
        
        print(f'Epoch [{epoch}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_history

def train_model_classification(model, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=64, lr=0.001):
    criterion = nn.CrossEntropyLoss()  # Considering that you're using Softmax activation in the model
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Converting data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Using long as expected by CrossEntropyLoss

    if X_val is not None and y_val is not None:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # Move model to appropriate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(torch.cuda.is_available())
    

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            # Move tensors to the right device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Optional: evaluate on validation data
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor.to(device))
                val_loss = criterion(val_outputs, y_val_tensor.to(device))
                _, preds = torch.max(val_outputs, 1)
                val_acc = torch.sum(preds == y_val_tensor.to(device)) / y_val_tensor.size(0)
            print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

    #################################################################
    # METADATA:                                                     #
    #################################################################
    # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # model_save_path = os.path.join("models", current_time)
    # os.makedirs(model_save_path, exist_ok=True)
    #
    # metadata = {
    #     "training_date": current_time,
    #     "dataset_metadata": {
    #         "input_features": list(X_train.columns),
    #         "output_features": list(y_train),
    #         "is_normalized": is_normalized,
    #     },
    #     "training_metadata": {
    #         "model_type": "",
    #         "learning_rate": learning_rate,
    #         "num_epochs": num_epochs,
    #         "patience": patience,
    #         "model_config": get_model_config(model),
    #     }
    # }
    # save_metadata(metadata, model_save_path)

def denormalize_data(normalized_data, mean, std_dev):
    """
    Denormalize the data.
    """
    denormalized_data = normalized_data * std_dev + mean
    return denormalized_data

def evaluate_model(model, X_test, y_test, n_examples=5):
    # assert isinstance(model, LSTMModel), f"Expected LSTMModel, but got {type(model)}"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Ensure X_test and y_test are numpy arrays, and convert them to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    
    model.eval()
    
    with torch.no_grad():
        predictions = model(X_test_tensor)
    
    loss = nn.MSELoss()(predictions, y_test_tensor)
    print(f'Model Loss on Test Data: {loss.item()}')
    
    predictions = predictions.cpu().numpy()

    squeezed_predictions = np.squeeze(predictions)
    correlation_coefficient = np.corrcoef(y_test, squeezed_predictions)[0, 1]
    print(f'Correlation between y_test and predictions: {correlation_coefficient}')

    # Plotting overall results
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, label='True', alpha=0.6)
    plt.scatter(range(len(predictions)), predictions, label='Predicted', alpha=0.6)
    plt.legend()
    plt.title('Predicted vs Actual Variations')
    plt.xlabel('Time (if time series)')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

    # Plotting examples as points
    if n_examples > len(y_test):
        n_examples = len(y_test)

    sample_indices = np.random.choice(len(y_test), n_examples, replace=False)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(range(n_examples), y_test[sample_indices], label='True')
    plt.scatter(range(n_examples), predictions[sample_indices], label='Predicted')
    plt.xticks(range(n_examples), sample_indices)
    plt.xlabel('Example')
    plt.ylabel('Variation')
    plt.title('Predicted vs Actual Variations for Selected Points')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss_history(loss_history, title='Model Loss', xlabel='Epoch', ylabel='Loss'):
    """
    Plots the loss history of the model training.

    Parameters:
    - loss_history (list of float): The history of the loss during training.
    - title (str): Title of the plot.
    - xlabel (str): Label of the x-axis.
    - ylabel (str): Label of the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def save_metadata(metadata, model_save_path):
    """
    Save metadata about the model.
    """
    metadata_path = os.path.join(model_save_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def evaluate_model_classification(model, X_test, y_test, class_labels=None):
    """
    Evaluate the model on the test data.
    
    Args:
    - model: Trained model
    - X_test: Test features
    - y_test: Test labels
    - class_labels (optional): List of labels names
    
    Returns:
    - acc: Accuracy of the model on the test data
    """
    if class_labels is None:
        class_labels = ["--", "0+/-", "++"]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Convert data to PyTorch tensors and move to device
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    model.to(device).eval()  # Move model to device and set it to evaluation mode
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, preds = torch.max(outputs, 1)
        
    # Move predictions to CPU to use sklearn metrics
    preds_cpu = preds.cpu().numpy()
    y_test_cpu = y_test_tensor.cpu().numpy()
    
    # Calculate and display accuracy
    acc = accuracy_score(y_test_cpu, preds_cpu)
    print(f"Accuracy: {acc*100:.2f}%")
    
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test_cpu, preds_cpu))
    
    # Display confusion matrix
    conf_matrix = confusion_matrix(y_test_cpu, preds_cpu)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    return acc
