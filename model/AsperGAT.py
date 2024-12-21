import numpy as np
import pandas as pd
import torch
import pickle
import torch_geometric
from torch_geometric.data import Data, Dataset
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn import Linear
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

# Fix for deprecation warning
from torch_geometric.loader import DataLoader as NewDataLoader

class GraphDataset(Dataset):
    """Custom dataset for loading graph data into a PyTorch-Geometric format."""
    def __init__(self, data_frame, root=None, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data_frame = data_frame

    def len(self):
        return len(self.data_frame)

    def get(self, idx):
        row = self.data_frame.iloc[idx]
        
        # Cloning and detaching tensors to avoid memory issues
        node_labels = np.nan_to_num(row['node_labels'], nan=-1).astype(int)
        edge_index = row['edge_index'].clone().detach().cpu()
        edge_weight = row['edge_weights'].clone().detach().cpu()
        node_features = torch.tensor(np.vstack(row['node_features']), dtype=torch.float).cpu()
        train_mask = row['train_mask'].clone().detach().cpu().to(torch.bool)
        test_mask = row['test_mask'].clone().detach().cpu().to(torch.bool)

        return Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight, y=torch.tensor(node_labels, dtype=torch.long),
                    train_mask=train_mask, test_mask=test_mask)

class AsperGAT(torch.nn.Module):
    """AsperGAT model for essentiality prediction."""
    def __init__(self, num_features, embedding_dim=100, hidden_channels=64, heads=1, num_layers=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_features, embedding_dim)
        self.conv1 = GATConv(embedding_dim, hidden_channels, heads=heads, dropout=0.2, concat=True)
        self.ln1 = LayerNorm(hidden_channels * heads)
        self.hidden_layers = torch.nn.ModuleList(
            [GATConv(heads * hidden_channels, hidden_channels, heads=heads, dropout=0.5, concat=True) for _ in range(num_layers - 1)]
        )
        self.fc = Linear(heads * hidden_channels, 1)

    def forward(self, data):
        device = next(self.parameters()).device  # Get the device of the model
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        
        # Ensure x is on the same device as the model (GPU or CPU)
        x = x.to(device).float()  # Move x to the correct device and ensure it's float
        
        # Checkpointing with explicit `use_reentrant=False` for future compatibility
        x = checkpoint(self.embedding, torch.argmax(x, dim=1), use_reentrant=False)
        x = F.leaky_relu(self.ln1(self.conv1(x, edge_index, edge_weight)))
        x = F.dropout(x, p=0.2, training=self.training)
        
        for layer in self.hidden_layers:
            # Checkpoint each layer with `use_reentrant=False`
            x = checkpoint(layer, x, edge_index, edge_weight, use_reentrant=False)
            x = F.dropout(F.leaky_relu(x), p=0.5, training=self.training)
        
        return self.fc(x).squeeze(-1)


def train(model, data, optimizer, criterion, epoch, accumulation_steps=1, scaler=None):
    model.train()
    optimizer.zero_grad()

    # Gradient accumulation
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask].float())

    # Use mixed precision if scaler is provided
    if scaler:
        with torch.amp.autocast('cuda'):  # For GPU
            scaled_loss = scaler.scale(loss)  # Scale the loss for mixed precision
            scaled_loss.backward()  # Backpropagate the scaled loss
        scaler.step(optimizer)  # Update optimizer using the scaled gradients
        scaler.update()  # Update the scaler for next iteration
    else:
        loss.backward()  # Standard backpropagation if no scaler

    if accumulation_steps > 1:
        if (epoch + 1) % accumulation_steps == 0:  # Update weights after accumulation
            optimizer.step()
            optimizer.zero_grad()
    else:
        optimizer.step()

    pred = torch.sigmoid(out[data.train_mask]) > 0.5  # Change threshold to 0.5 for now
    acc = pred.eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()

    return loss.item(), acc


def test(model, data):
    model.eval()
    with torch.no_grad():  # Disable gradient tracking for evaluation
        out = model(data)
        pred = torch.sigmoid(out[data.test_mask]) > 0.5  # Change threshold to 0.5
        true = data.y[data.test_mask]

        acc = pred.eq(true).sum().item() / data.test_mask.sum().item()
        f1 = f1_score(true.cpu(), pred.cpu())
        precision = precision_score(true.cpu(), pred.cpu(), zero_division=1)
        recall = recall_score(true.cpu(), pred.cpu(), zero_division=1)
        probabilities = torch.sigmoid(out[data.test_mask]).cpu().numpy()

    return acc, f1, precision, recall, probabilities, true.cpu(), pred.cpu()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    df = pd.read_pickle('../data/k_fold_masks.pkl')
    dataset = GraphDataset(df)

    num_repeats = 5
    num_epochs = 20

    # Gradient Scaler for Mixed Precision
    scaler = GradScaler()

    best_f1 = 0.0  # Initialize the best F1 score
    best_model_state_dict = None  # Placeholder for the best model's state_dict

    for repeat in range(num_repeats):
        print(f'Repeat {repeat + 1}/{num_repeats}')

        fold_metrics = []

        # Create a DataLoader for batching graphs
        loader = torch_geometric.loader.DataLoader(dataset, batch_size=1, shuffle=True)

        for fold, data in enumerate(loader):
            print(f'  Fold {fold + 1}:')
            data = data.to(device)
            
            # Model setup with reduced parameters
            model = AsperGAT(num_features=data.x.size(1), embedding_dim=100, hidden_channels=64, heads=1, num_layers=2).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

            # Adjusting class imbalance with BCEWithLogitsLoss
            number_of_zeros = (data.y == 0).sum()
            number_of_ones = (data.y == 1).sum()
            weight = torch.tensor([number_of_zeros / number_of_ones]).to(device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

            # Training loop with gradient accumulation
            for epoch in range(num_epochs):
                train_loss, train_acc = train(model, data, optimizer, criterion, epoch, accumulation_steps=2, scaler=scaler)
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

                # Clear GPU memory at the end of each epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Test the model and compare F1 score
            test_acc, test_f1, precision, recall, probabilities, true, pred = test(model, data)
            print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test Precision: {precision:.4f}, Test Recall: {recall:.4f}")
            print('--------------------------------------------------')

            # Track the best model based on the F1 score
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_model_state_dict = model.state_dict()

            fold_metrics.append((test_acc, test_f1))

        avg_metrics = np.mean(fold_metrics, axis=0)
        print(f'Average Test Accuracy: {avg_metrics[0]:.4f}, Average Test F1: {avg_metrics[1]:.4f}')

    # Save the best model checkpoint as a pickle file
    if best_model_state_dict is not None:
        print("Saving the best model checkpoint as a pickle file.")
        with open('best_model_checkpoint.pkl', 'wb') as f:
            pickle.dump(best_model_state_dict, f)

    # Load the best model
    model = AsperGAT(num_features=data.x.size(1), embedding_dim=100, hidden_channels=64, heads=1, num_layers=2).to(device)
    
    # Load the saved model checkpoint
    with open('best_model_checkpoint.pkl', 'rb') as f:
        best_model_state_dict = pickle.load(f)
    
    # Load the state_dict into the model
    model.load_state_dict(best_model_state_dict)
    model.eval()

    # Make predictions on the test set with the best model
    out = model(data)
    pred = torch.sigmoid(out[data.test_mask]) > 0.5  # Use the same threshold as before
    true = data.y[data.test_mask]

    print("Final Test Results:")
    final_acc = pred.eq(true).sum().item() / data.test_mask.sum().item()
    final_f1 = f1_score(true.cpu(), pred.cpu())
    final_precision = precision_score(true.cpu(), pred.cpu(), zero_division=1)
    final_recall = recall_score(true.cpu(), pred.cpu(), zero_division=1)
    print(f"Final Accuracy: {final_acc:.4f}, Final F1: {final_f1:.4f}, Final Precision: {final_precision:.4f}, Final Recall: {final_recall:.4f}")

if __name__ == '__main__':
    main()
