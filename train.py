from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, spearmanr
from model import *

# Train a model on the complete dataset
if __name__ == "__main__":
    # Load cryptic seq data
    seq_length = 46
    file_path = 'data/train.csv'
    decoy_path = 'data/decoys.csv'
    hidden_sizes = [200, 200]

    X, y = read_data(file_path, decoy_path, decoy_mul=0)

    # Mask the central dinucleotide from X
    X = np.array([seq[:22] + 'NN' + seq[24:] for seq in X])

    # Lists to store results
    fold_results = []
    all_y_val = []
    all_y_val_pred = []

    # Train on complete dataset
    input_size = seq_length * 5  # 4 bases + N, one-hot encoded
    output_size = 1
    model = MLP(input_size, hidden_sizes, output_size)

    # Train the model
    train_model(model, X, y, batch_size=256, n_samples=10000, epochs=100, lr=1e-3, log=True)
    torch.save(model.state_dict(), 'weights/200_200_mlp.pt')