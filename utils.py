"""
Utility functions for training and evaluation
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def get_device():
    """Get appropriate device for training (MPS for Apple Silicon)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def get_data_loaders(batch_size=128, use_augmentation=False):
    """Get MNIST data loaders with optional augmentation"""
    
    # Basic transforms
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Training transforms with optional augmentation
    if use_augmentation:
        train_transforms = transforms.Compose([
            transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transforms = test_transforms
    
    # Datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, device, train_loader, optimizer, epoch, *, loss_config=None, scheduler=None, ema=None, sam_config=None):
    """Train for one epoch with optional label smoothing, per-batch scheduler step, and EMA.

    Args:
        loss_config: dict or None. Example: {"type": "label_smoothing", "smoothing": 0.05}
        scheduler: Optional LR scheduler stepped per-batch (e.g., OneCycleLR)
        ema: Optional ExponentialMovingAverage instance to update after optimizer.step()
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    def _compute_loss(output, target):
        if not loss_config or loss_config.get("type", "nll") == "nll":
            return F.nll_loss(output, target)
        if loss_config.get("type") == "label_smoothing":
            smoothing = float(loss_config.get("smoothing", 0.05))
            num_classes = output.size(1)
            with torch.no_grad():
                true_dist = torch.full_like(output, smoothing / (num_classes - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
            # output expected to be log-probabilities
            return F.kl_div(output, true_dist, reduction='batchmean')
        return F.nll_loss(output, target)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        use_sam = sam_config is not None
        if not use_sam:
            optimizer.zero_grad()
            output = model(data)
            loss = _compute_loss(output, target)
            loss.backward()
            optimizer.step()
        else:
            # SAM two-step update (fore/back) with proper no_grad perturb/restore
            rho = float(sam_config.get("rho", 0.05))
            eps = 1e-12
            # First step: ascent to find sharpness direction
            optimizer.zero_grad()
            output = model(data)
            loss = _compute_loss(output, target)
            loss.backward()
            # Compute grad norm
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if len(grads) == 0:
                grad_norm = torch.tensor(0.0, device=data.device)
            else:
                grad_norm = torch.norm(torch.stack([g.norm(p=2) for g in grads]), p=2)
            scale = rho / (grad_norm + eps)
            # Perturb weights and store perturbations
            e_ws = []
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is None:
                        e_ws.append(None)
                        continue
                    e_w = p.grad * scale
                    p.add_(e_w)
                    e_ws.append(e_w)
            # Second step: descent at perturbed weights
            optimizer.zero_grad()
            output_adv = model(data)
            loss_adv = _compute_loss(output_adv, target)
            loss_adv.backward()
            optimizer.step()
            # Restore original weights
            with torch.no_grad():
                for p, e_w in zip(model.parameters(), e_ws):
                    if e_w is not None:
                        p.sub_(e_w)

        if ema is not None:
            # Support both custom EMA (with update) and PyTorch SWA AveragedModel (with update_parameters)
            if hasattr(ema, "update"):
                ema.update(model)
            elif hasattr(ema, "update_parameters"):
                ema.update_parameters(model)

        if scheduler is not None and not use_sam:
            # For OneCycle, step per batch during non-SAM phases
            scheduler.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(
                f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                f'Loss: { (loss_adv if sam_config is not None else loss).item():.6f}, '
                f'Acc: {100.*correct/max(1,total):.2f}%'
            )

    return train_loss/max(1, len(train_loader)), 100.*correct/max(1, total)

def test_epoch(model, device, test_loader, *, ema=None):
    """Evaluate on test set, optionally using EMA shadow weights."""
    class _NullCtx:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    context = ema.apply_and_restore(model) if ema is not None else _NullCtx()
    with context:
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        test_loss /= max(1, total)
        accuracy = 100. * correct / max(1, total)

        print(f'Test set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{total} ({accuracy:.2f}%)')

    return test_loss, accuracy


class ExponentialMovingAverage:
    """Exponential Moving Average of model parameters (simple implementation)."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: torch.nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.data.clone()
            param.data = self.shadow[name].clone()

    def restore(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data = self.backup[name].clone()
        self.backup = {}

    def apply_and_restore(self, model: torch.nn.Module):
        ema = self
        class _Ctx:
            def __enter__(self_nonlocal):
                ema.apply_shadow(model)
                return ema
            def __exit__(self_nonlocal, exc_type, exc_val, exc_tb):
                ema.restore(model)
                return False
        return _Ctx()

def plot_metrics(train_losses, train_accs, test_losses, test_accs, title="Training Metrics"):
    """Plot training metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.plot(test_losses)
    ax2.set_title('Test Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    
    ax3.plot(train_accs)
    ax3.set_title('Training Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    
    ax4.plot(test_accs)
    ax4.set_title('Test Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.axhline(y=99.4, color='r', linestyle='--', label='Target 99.4%')
    ax4.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def check_target_achievement(accuracies, target=99.4, last_n_epochs=3):
    """Check if target accuracy is consistently achieved in last n epochs"""
    if len(accuracies) < last_n_epochs:
        return False
    
    last_epochs = accuracies[-last_n_epochs:]
    return all(acc >= target for acc in last_epochs)

def get_model_summary(model, input_size=(1, 28, 28)):
    """Get model summary"""
    total_params = count_parameters(model)
    
    # Test forward pass
    model.eval()
    test_input = torch.randn(1, *input_size)
    try:
        output = model(test_input)
        output_shape = output.shape
    except Exception as e:
        output_shape = f"Error: {e}"
    
    print(f"Model Parameters: {total_params:,}")
    print(f"Input Shape: {input_size}")
    print(f"Output Shape: {output_shape}")
    print(f"Parameter Budget: {'✓' if total_params < 8000 else '✗'} (<8000)")
    
    return total_params
