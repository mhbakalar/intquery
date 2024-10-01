import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

# Create a function to get a TensorBoard writer
def get_tensorboard_writer(log_dir="runs"):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(log_dir, current_time)
    return SummaryWriter(log_dir=log_dir)

# Create a function to log metrics to TensorBoard
def log_metrics(logger, epoch, train_loss):
    logger.add_scalar('Loss/Train', train_loss, epoch)

# Define the model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.05):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        return y

# Function to one-hot encode DNA sequences
def one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0, 0], 'C': [0, 1, 0, 0, 0], 'T': [0, 0, 1, 0, 0], 'G': [0, 0, 0, 1, 0], 'N': [0, 0, 0, 0, 1]}
    return np.array([mapping[base] for base in sequence]).flatten()

# Generate some dummy data
def generate_dummy_data(num_samples, seq_length):
    bases = ['A', 'C', 'T', 'G', 'N']
    X = [''.join(np.random.choice(bases, seq_length)) for _ in range(num_samples)]
    y = np.random.rand(num_samples)  # Random scores between 0 and 1
    return X, y

# Read data from csv
def read_data(file_path, decoy_path, decoy_mul=0):
    df = pd.read_csv(file_path)

    # Group by genomic coordinates and take the mean of count, ignoring strand
    df = df.groupby(['reference_name', 'seq_start', 'seq_end']).agg({
        'count': 'mean',
        'seq': 'first'  # Keep the first sequence for each group
    }).reset_index()
    
    X = df['seq'].values
    y = df['count'].values    

    if decoy_mul > 0:
        # Add decoy sequences
        decoy_df = pd.read_csv(decoy_path)

        # Select random sequences from decoy_df
        num_samples = int(len(X)*decoy_mul)
        decoy_X = decoy_df['seq'].sample(n=num_samples, replace=True)
        decoy_y = np.zeros(len(decoy_X))

        X = np.concatenate([X, decoy_X])
        y = np.concatenate([y, decoy_y])

    # Log transform scores
    y = np.log(1 + y)

    return X, y

def select_data_subset(X, y, n_samples):
        # Sample indices by weight
        weights = y
        weights = weights / weights.sum()
        indices = np.random.choice(np.arange(len(X)), size=n_samples, p=weights)
        return X[indices], y[indices]

# Main training function
def train_model(model, X, y, epochs=100, n_samples=10000, batch_size=32, lr=1e-3, log=False):
    
    # Initialize optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if log:
        logger = get_tensorboard_writer()
    else:
        logger = None
    
    # Move data to GPU if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Training on device: {device}")
    model.to(device)
    
    X_select, y_select = select_data_subset(X, y, n_samples=n_samples)

    for epoch in range(epochs):
        for i in range(0, len(X_select), batch_size):
            batch_X = X_select[i:i+batch_size]
            batch_y = y_select[i:i+batch_size]
            
            inputs = torch.FloatTensor(np.array([one_hot_encode(seq) for seq in batch_X])).to(device)
            targets = torch.FloatTensor(batch_y).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            if logger is not None:
                log_metrics(logger, epoch, loss.item())

        # Select new data subset
        X_select, y_select = select_data_subset(X, y, n_samples=10000)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def eval_model(model, X, batch_size=32):
    model.eval()
    # Move data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    predictions = []
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]        
        inputs = torch.FloatTensor(np.array([one_hot_encode(seq) for seq in batch_X])).to(device)
        outputs = model(inputs)
        predictions.extend(outputs.squeeze().tolist())
    
    return np.array(predictions)


# Main execution
if __name__ == "__main__":
    # Read data from file
    seq_length = 46
    file_path = 'data/cs_data.csv'
    decoy_path = 'data/cs_data.csv'
    X, y = read_data(file_path, decoy_path, decoy_mul=0)

    # Mask the central dinucleotide from X
    X = np.array([seq[:22] + 'NN' + seq[24:] for seq in X])

    # Initialize the model
    input_size = seq_length * 5  # 4 bases + N, one-hot encoded
    hidden_size = 128
    output_size = 1
    model = DNAScorePredictor(input_size, hidden_size, output_size)
    
    # Train the model
    train_model(model, X, y, log=True)
    print("Training completed!")

    # Evaluate the model
    predictions = eval_model(model, X)
    
    # Compute the Pearson correlation coefficient
    pearson_corr = np.corrcoef(y, predictions)[0, 1]
    print(f"Pearson correlation coefficient: {pearson_corr:.4f}")
