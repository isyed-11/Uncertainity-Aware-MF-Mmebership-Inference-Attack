"""Standard PyTorch MLP baseline. All work requires an explicit function call."""
import copy
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import sparse


def build_mlp(input_dim, hidden_dims=(128, 64), dropout=0.0, num_classes=2):
    """Configurable output logits; binary remains the default for ACS/Adult."""
    layers = []
    width = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(width, hidden), nn.ReLU()])
        if dropout:
            layers.append(nn.Dropout(dropout))
        width = hidden
    layers.append(nn.Linear(width, num_classes))
    return nn.Sequential(*layers)


def predict_probabilities(model, X, *, batch_size=1024, device=None):
    """Return class probabilities in output-index order for each original row."""
    device = torch.device(device or next(model.parameters()).device)
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            values = X[start:start + batch_size]
            if sparse.issparse(values):
                values = values.toarray()
            batch = torch.tensor(values, dtype=torch.float32, device=device)
            predictions.append(model(batch).softmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def evaluate_model(model, X, y, *, batch_size=1024, device=None):
    probabilities = predict_probabilities(model, X, batch_size=batch_size, device=device)
    auc = (roc_auc_score(y, probabilities[:, 1]) if probabilities.shape[1] == 2
           else roc_auc_score(y, probabilities, multi_class='ovr', average='macro',
                              labels=np.arange(probabilities.shape[1])))
    return {
        'accuracy': float(accuracy_score(y, probabilities.argmax(axis=1))),
        'roc_auc': float(auc),
        'log_loss': float(log_loss(y, probabilities, labels=np.arange(probabilities.shape[1]))),
    }


def train_target_model(X_train, y_train, X_validation, y_validation, *,
                       hidden_dims=(128, 64), dropout=0.0, epochs=50,
                       batch_size=256, learning_rate=1e-3, weight_decay=1e-4,
                       patience=5, seed=42, device='cpu', early_stopping=True,
                       restore_best=True, monitor_train=False, num_classes=2,
                       target_gap_pp=None):
    """Fit an MLP; by default select and restore the best validation epoch.

    No test or MIA data is accepted. Returns (model, history, metadata).
    CPU is the reproducible default; GPU reproducibility can vary by platform.
    For a fixed-budget overfitting experiment use early_stopping=False and
    restore_best=False. monitor_train adds end-of-epoch training metrics.
    Both validation arguments may be None only in that fixed-budget mode;
    no held-out test data is then passed to training or monitored by this helper.
    Optional target_gap_pp selects an intermediate model using the end-of-epoch
    train-minus-validation accuracy gap, in percentage points. Stop at the first
    crossing of that gap or the epoch limit and return the closest visited epoch
    (earlier epoch wins ties). Weights stay in memory. This mode requires
    validation, monitor_train=True and both loss-based selection flags off.
    """
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError('epochs, batch_size, and patience must be positive.')
    if (X_validation is None) != (y_validation is None):
        raise ValueError('Provide both validation inputs/labels, or neither.')
    has_validation = X_validation is not None
    if target_gap_pp is not None:
        if not np.isfinite(target_gap_pp) or not 0 < target_gap_pp <= 100:
            raise ValueError('target_gap_pp must be finite and in (0, 100].')
        if not has_validation or not monitor_train or early_stopping or restore_best:
            raise ValueError('Gap selection requires validation, monitor_train=True, '
                             'early_stopping=False and restore_best=False.')
    if not has_validation and (early_stopping or restore_best):
        raise ValueError('Without validation, early_stopping and restore_best must both be False.')
    if X_train.ndim != 2 or (has_validation and
            (X_validation.ndim != 2 or X_train.shape[1] != X_validation.shape[1])):
        raise ValueError('Training and validation feature dimensions must agree.')
    if X_train.shape[0] != len(y_train) or (has_validation and X_validation.shape[0] != len(y_validation)):
        raise ValueError('Feature and label row counts must agree.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device)
    model = build_mlp(X_train.shape[1], hidden_dims, dropout, num_classes=num_classes).to(device)
    # Copy inputs: pandas-backed NumPy arrays may be read-only.
    sparse_input = sparse.issparse(X_train)
    train_inputs = (torch.arange(X_train.shape[0]) if sparse_input
                    else torch.tensor(X_train, dtype=torch.float32))
    dataset = TensorDataset(train_inputs,
                            torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_loss, stale_epochs = float('inf'), 0
    best_state, best_epoch = None, None
    closest_gap_state, closest_gap_epoch = None, None
    closest_gap_error, selected_gap_pp = float('inf'), None
    gap_target_crossed = False
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            if sparse_input:
                X_batch = torch.tensor(X_train[X_batch.numpy()].toarray(), dtype=torch.float32)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        validation = evaluate_model(model, X_validation, y_validation, device=device) if has_validation else {}
        history.append({'epoch': epoch, 'train_loss': total_loss / len(dataset),
                        **{f'validation_{key}': value for key, value in validation.items()}})
        if monitor_train:
            training = evaluate_model(model, X_train, y_train, device=device)
            history[-1].update({f'train_eval_{key}': value for key, value in training.items()})
        if has_validation:
            print(f"Epoch {epoch:03d}: train loss={history[-1]['train_loss']:.4f}, "
                  f"validation loss={validation['log_loss']:.4f}, "
                  f"validation accuracy={validation['accuracy']:.4f}")
        else:
            print(f"Epoch {epoch:03d}: train loss={history[-1]['train_loss']:.4f}; fixed schedule, no validation")
        if has_validation and validation['log_loss'] < best_loss:
            best_loss, stale_epochs = validation['log_loss'], 0
            best_epoch = epoch
            if restore_best:
                best_state = copy.deepcopy(model.state_dict())
        elif has_validation:
            stale_epochs += 1
            if early_stopping and stale_epochs >= patience:
                break
        if target_gap_pp is not None:
            gap_pp = 100 * (training['accuracy'] - validation['accuracy'])
            history[-1]['train_validation_accuracy_gap_pp'] = gap_pp
            gap_error = abs(gap_pp - target_gap_pp)
            # Small numerical ties retain the earlier epoch.
            if gap_error < closest_gap_error - 1e-10:
                closest_gap_error, selected_gap_pp = gap_error, gap_pp
                closest_gap_epoch = epoch
                closest_gap_state = copy.deepcopy(model.state_dict())
            if gap_pp >= target_gap_pp - 1e-10:
                gap_target_crossed = True
                break
    if target_gap_pp is not None:
        model.load_state_dict(closest_gap_state)
    elif restore_best:
        model.load_state_dict(best_state)
    model.eval()
    metadata = {'input_dim': X_train.shape[1], 'num_classes': num_classes, 'hidden_dims': list(hidden_dims),
                'dropout': dropout, 'best_epoch': best_epoch, 'seed': seed,
                'epochs_requested': epochs, 'epochs_run': len(history),
                'batch_size': batch_size, 'learning_rate': learning_rate,
                'weight_decay': weight_decay, 'patience': patience, 'device': str(device),
                'early_stopping': early_stopping, 'restore_best': restore_best,
                'monitor_train': monitor_train,
                'returned_epoch': (closest_gap_epoch if target_gap_pp is not None
                                   else best_epoch if restore_best else len(history))}
    if target_gap_pp is not None:
        metadata.update(target_gap_pp=float(target_gap_pp),
                        selected_train_validation_gap_pp=float(selected_gap_pp),
                        gap_distance_pp=float(closest_gap_error),
                        gap_target_crossed=gap_target_crossed,
                        stopping_reason='gap_target_crossed' if gap_target_crossed else 'epoch_limit',
                        weight_selection='closest gap among epochs visited through first crossing or epoch limit')
    return model, history, metadata


def save_target_model(model, metadata, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({'state_dict': state, 'metadata': metadata}, path)
