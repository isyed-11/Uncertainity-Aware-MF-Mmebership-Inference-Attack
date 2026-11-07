"""Standard PyTorch MLP baseline. All work requires an explicit function call."""
import copy
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_mlp(input_dim, hidden_dims=(128, 64), dropout=0.0):
    """Two output logits; no batch normalization, to ease later DP experiments."""
    layers = []
    width = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(width, hidden), nn.ReLU()])
        if dropout:
            layers.append(nn.Dropout(dropout))
        width = hidden
    layers.append(nn.Linear(width, 2))
    return nn.Sequential(*layers)


def predict_probabilities(model, X, *, batch_size=1024, device=None):
    """Return P(income<=50k), P(income>50k) for each row, in original order."""
    device = torch.device(device or next(model.parameters()).device)
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            predictions.append(model(batch).softmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def evaluate_model(model, X, y, *, batch_size=1024, device=None):
    probabilities = predict_probabilities(model, X, batch_size=batch_size, device=device)
    return {
        'accuracy': float(accuracy_score(y, probabilities.argmax(axis=1))),
        'roc_auc': float(roc_auc_score(y, probabilities[:, 1])),
        'log_loss': float(log_loss(y, probabilities, labels=[0, 1])),
    }


def train_target_model(X_train, y_train, X_validation, y_validation, *,
                       hidden_dims=(128, 64), dropout=0.0, epochs=50,
                       batch_size=256, learning_rate=1e-3, weight_decay=1e-4,
                       patience=5, seed=42, device='cpu', early_stopping=True,
                       restore_best=True, monitor_train=False):
    """Fit an MLP; by default select and restore the best validation epoch.

    No test or MIA data is accepted. Returns (model, history, metadata).
    CPU is the reproducible default; GPU reproducibility can vary by platform.
    For a fixed-budget overfitting experiment use early_stopping=False and
    restore_best=False. monitor_train adds end-of-epoch training metrics.
    """
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError('epochs, batch_size, and patience must be positive.')
    if X_train.ndim != 2 or X_validation.ndim != 2 or X_train.shape[1] != X_validation.shape[1]:
        raise ValueError('Training and validation feature dimensions must agree.')
    if len(X_train) != len(y_train) or len(X_validation) != len(y_validation):
        raise ValueError('Feature and label row counts must agree.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device)
    model = build_mlp(X_train.shape[1], hidden_dims, dropout).to(device)
    # Copy inputs: pandas-backed NumPy arrays may be read-only.
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_loss, stale_epochs = float('inf'), 0
    best_state, best_epoch = None, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        validation = evaluate_model(model, X_validation, y_validation, device=device)
        history.append({'epoch': epoch, 'train_loss': total_loss / len(dataset),
                        **{f'validation_{key}': value for key, value in validation.items()}})
        if monitor_train:
            training = evaluate_model(model, X_train, y_train, device=device)
            history[-1].update({f'train_eval_{key}': value for key, value in training.items()})
        print(f"Epoch {epoch:03d}: train loss={history[-1]['train_loss']:.4f}, "
              f"validation loss={validation['log_loss']:.4f}, "
              f"validation accuracy={validation['accuracy']:.4f}")
        if validation['log_loss'] < best_loss:
            best_loss, stale_epochs = validation['log_loss'], 0
            best_epoch = epoch
            if restore_best:
                best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if early_stopping and stale_epochs >= patience:
                break
    if restore_best:
        model.load_state_dict(best_state)
    model.eval()
    metadata = {'input_dim': X_train.shape[1], 'hidden_dims': list(hidden_dims),
                'dropout': dropout, 'best_epoch': best_epoch, 'seed': seed,
                'epochs_requested': epochs, 'epochs_run': len(history),
                'batch_size': batch_size, 'learning_rate': learning_rate,
                'weight_decay': weight_decay, 'patience': patience, 'device': str(device),
                'early_stopping': early_stopping, 'restore_best': restore_best,
                'monitor_train': monitor_train,
                'returned_epoch': best_epoch if restore_best else len(history)}
    return model, history, metadata


def save_target_model(model, metadata, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({'state_dict': state, 'metadata': metadata}, path)
