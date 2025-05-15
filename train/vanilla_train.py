import sys 
sys.path.append("../")

import os
import time
import json
import argparse
import logging
import yaml
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import torchmetrics

import datasets
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# MRL components from the original code
from MRL import Matryoshka_CE_Loss, MRL_Linear_Layer, FixedFeatureLayer


class ImageNetDataset(Dataset):
    """Custom dataset class to handle HuggingFace dataset with transformations."""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        if self.transform:
            image = self.transform(image)
        
        return {'pixel_values': image, 'label': torch.tensor(label)}


class Config:
    """Configuration class to store all training parameters."""
    def __init__(self, **kwargs):
        # Model parameters
        self.arch = kwargs.get('arch', 'resnet50')
        self.pretrained = kwargs.get('pretrained', True)
        self.efficient = kwargs.get('efficient', False)  # MRL-E mode
        self.mrl = kwargs.get('mrl', True)  # Whether to use MRL
        self.nesting_start = kwargs.get('nesting_start', 3)
        self.fixed_feature = kwargs.get('fixed_feature', 2048)
        self.use_blurpool = kwargs.get('use_blurpool', False)
        
        # Data parameters
        self.train_dataset = kwargs.get('train_dataset', 'data/imagenet_1k_resized_256_train')
        self.val_dataset = kwargs.get('val_dataset', 'data/imagenet_1k_resized_256_val')
        self.num_workers = kwargs.get('num_workers', 4)
        
        # Training parameters
        self.batch_size = kwargs.get('batch_size', 256)
        self.val_batch_size = kwargs.get('val_batch_size', 512)
        self.epochs = kwargs.get('epochs', 30)
        self.eval_only = kwargs.get('eval_only', False)
        self.path = kwargs.get('path', None)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optimizer parameters - ensure numeric types
        self.optimizer = kwargs.get('optimizer', 'sgd')
        self.momentum = float(kwargs.get('momentum', 0.9))
        # Convert weight_decay to float (it might come as string from YAML)
        self.weight_decay = float(kwargs.get('weight_decay', 4e-5))
        self.label_smoothing = float(kwargs.get('label_smoothing', 0.1))
        
        # Learning rate parameters - ensure numeric types
        self.lr = float(kwargs.get('lr', 0.1))
        self.lr_schedule_type = kwargs.get('lr_schedule_type', 'cosine')
        self.step_ratio = float(kwargs.get('step_ratio', 0.1))
        self.step_length = int(kwargs.get('step_length', 30))
        self.lr_peak_epoch = int(kwargs.get('lr_peak_epoch', 2))
        
        # Resolution parameters - ensure numeric types
        self.min_res = int(kwargs.get('min_res', 160))
        self.max_res = int(kwargs.get('max_res', 224))
        self.start_ramp = int(kwargs.get('start_ramp', 0))
        self.end_ramp = int(kwargs.get('end_ramp', 0))
        
        # Validation parameters
        self.resolution = int(kwargs.get('resolution', 224))
        self.lr_tta = bool(kwargs.get('lr_tta', True))  # Test-time augmentation
        
        # Logging parameters
        self.log_folder = kwargs.get('log_folder', 'logs')
        self.log_level = int(kwargs.get('log_level', 1))
        
        # Create nesting list based on parameters
        if self.mrl or self.efficient:
            self.nesting_list = [2**i for i in range(self.nesting_start, 12)]
        else:
            self.nesting_list = None


class BlurPoolConv2d(nn.Module):
    """
    Implementation of BlurPool from "Making Convolutional Networks Shift-Invariant Again"
    """
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                          groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)


def apply_blurpool(model):
    """Apply BlurPool to all convolutional layers in the model."""
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d) and (max(child.stride if hasattr(child, 'stride') else [1]) > 1 and child.in_channels >= 16):
            setattr(model, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)


class MeanScalarMetric(torchmetrics.Metric):
    """Metric to compute the mean of scalar values."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.add_state('sum', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
    
    def update(self, sample: torch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()
    
    def compute(self):
        return self.sum.float() / self.count


def get_step_lr(epoch, config):
    """Step learning rate scheduler."""
    if epoch >= config.epochs:
        return 0
    num_steps = epoch // config.step_length
    return config.step_ratio**num_steps * config.lr


def get_constant_lr(epoch, config):
    """Constant learning rate scheduler."""
    return config.lr


def get_cyclic_lr(epoch, config):
    """Cyclic learning rate scheduler."""
    xs = [0, config.lr_peak_epoch, config.epochs]
    ys = [1e-4 * config.lr, config.lr, 0]
    return np.interp([epoch], xs, ys)[0]


def get_cosine_lr(epoch, config):
    """Cosine learning rate scheduler."""
    return config.lr * 0.5 * (1 + np.cos(np.pi * epoch / config.epochs))


def get_lr(epoch, config):
    """Get learning rate based on scheduler type and epoch."""
    lr_schedules = {
        'cyclic': get_cyclic_lr,
        'step': get_step_lr,
        'constant': get_constant_lr,
        'cosine': get_cosine_lr
    }
    return lr_schedules[config.lr_schedule_type](epoch, config)


def get_resolution(epoch, config):
    """Get resolution based on current epoch."""
    if epoch <= config.start_ramp:
        return config.min_res
    if epoch >= config.end_ramp:
        return config.max_res
    
    # Linearly interpolate to the nearest multiple of 32
    interp = np.interp([epoch], [config.start_ramp, config.end_ramp], 
                       [config.min_res, config.max_res])
    final_res = int(np.round(interp[0] / 32)) * 32
    return final_res


def create_train_loader(config):
    """Create data loader for training."""
    # Normalization values for ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Get current resolution for training
    res = get_resolution(0, config)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(int(res * 256 / 224)),  # Scale maintaining aspect ratio
        transforms.RandomResizedCrop(res),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load dataset
    raw_dataset = load_from_disk(config.train_dataset)
    
    # Create custom dataset with transforms
    dataset = ImageNetDataset(raw_dataset, transform=transform)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def create_val_loader(config):
    """Create data loader for validation."""
    # Normalization values for ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Define transforms for validation
    transform = transforms.Compose([
        transforms.Resize(int(config.resolution * 256 / 224)),
        transforms.CenterCrop(config.resolution),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load dataset
    raw_dataset = load_from_disk(config.val_dataset)
    
    # Create custom dataset with transforms
    dataset = ImageNetDataset(raw_dataset, transform=transform)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return dataloader


def create_model(config):
    """Create model based on config."""
    # Create model based on architecture
    model = getattr(models, config.arch)(pretrained=config.pretrained)
    
    # Modify the classifier layer based on the config
    if config.mrl or config.efficient:
        print(f"Creating classification layer of type: {'MRL-E' if config.efficient else 'MRL'}")
        model.fc = MRL_Linear_Layer(config.nesting_list, num_classes=1000, efficient=config.efficient)
    elif config.fixed_feature != 2048:
        print(f"Using Fixed Features with size: {config.fixed_feature}")
        model.fc = FixedFeatureLayer(config.fixed_feature, 1000)
    
    # Apply BlurPool if needed
    if config.use_blurpool:
        apply_blurpool(model)
    
    # Move model to device and channels last memory format for performance
    model = model.to(memory_format=torch.channels_last)
    model = model.to(config.device)
    
    return model


def create_optimizer_and_loss(model, config):
    """Create optimizer and loss function."""
    # Only do weight decay on non-batchnorm parameters
    all_params = list(model.named_parameters())
    bn_params = [v for k, v in all_params if ('bn' in k)]
    other_params = [v for k, v in all_params if not ('bn' in k)]
    param_groups = [{
        'params': bn_params,
        'weight_decay': 0.
    }, {
        'params': other_params,
        'weight_decay': config.weight_decay
    }]
    
    # Create optimizer
    if config.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(param_groups, lr=config.lr, momentum=config.momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    
    # Create loss function
    if config.mrl or config.efficient:
        loss_fn = Matryoshka_CE_Loss(label_smoothing=config.label_smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    return optimizer, loss_fn


def format_time(seconds):
    """Format time in seconds to a readable string."""
    return str(timedelta(seconds=int(seconds)))


def train_one_epoch(model, train_loader, optimizer, loss_fn, epoch, config):
    """Train the model for one epoch."""
    model.train()
    losses = []
    
    # Get learning rate for start and end of epoch
    lr_start, lr_end = get_lr(epoch, config), get_lr(epoch + 1, config)
    iters = len(train_loader)
    lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])
    
    # Create progress bar with detailed formatting including ETA
    progress_bar = tqdm(
        enumerate(train_loader), 
        total=len(train_loader),
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )
    
    for batch_idx, batch in progress_bar:
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lrs[batch_idx]
        
        # Move data to device
        images = batch['pixel_values'].to(config.device, non_blocking=True)
        targets = batch['label'].to(config.device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log progress
        losses.append(loss.detach())
        
        # Update progress bar with ETA and other info
        group_lrs = [f'{group["lr"]:.3f}' for group in optimizer.param_groups]
        eta = format_time(progress_bar._time() * (len(train_loader) - batch_idx - 1))
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.3f}',
            'LR': group_lrs,
            'Shape': tuple(images.shape),
            'ETA': eta
        })
    
    # Return average loss
    avg_loss = torch.stack(losses).mean().cpu().item()
    return avg_loss


def validate(model, val_loader, loss_fn, config):
    """Validate the model."""
    model.eval()
    
    # Create metrics
    if config.mrl or config.efficient:
        metrics = {}
        for dim in config.nesting_list:
            metrics[f'top1_{dim}'] = torchmetrics.Accuracy().to(config.device)
            metrics[f'top5_{dim}'] = torchmetrics.Accuracy(top_k=5).to(config.device)
    else:
        metrics = {
            'top1': torchmetrics.Accuracy().to(config.device),
            'top5': torchmetrics.Accuracy(top_k=5).to(config.device)
        }
    
    loss_meter = MeanScalarMetric().to(config.device)
    
    # Create progress bar with ETA
    progress_bar = tqdm(
        val_loader, 
        desc="Validation",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move data to device
            images = batch['pixel_values'].to(config.device, non_blocking=True)
            targets = batch['label'].to(config.device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            
            # If TTA is enabled, add predictions from flipped images
            if config.lr_tta:
                flipped_images = torch.flip(images, dims=[3])
                flipped_outputs = model(flipped_images)
                
                if config.mrl or config.efficient:
                    # For MRL, outputs is a tuple of tensors
                    outputs = tuple(out + f_out for out, f_out in zip(outputs, flipped_outputs))
                else:
                    outputs += flipped_outputs
            
            # Compute loss
            loss = loss_fn(outputs, targets)
            loss_meter.update(loss)
            
            # Update metrics
            if config.mrl or config.efficient:
                # For MRL, outputs is a tuple of tensors
                for i, dim in enumerate(config.nesting_list):
                    metrics[f'top1_{dim}'].update(outputs[i], targets)
                    metrics[f'top5_{dim}'].update(outputs[i], targets)
            else:
                metrics['top1'].update(outputs, targets)
                metrics['top5'].update(outputs, targets)
            
            # Update progress bar
            eta = format_time(progress_bar._time() * (len(val_loader) - progress_bar.n - 1))
            progress_bar.set_postfix({'Loss': f'{loss.item():.3f}', 'ETA': eta})
    
    # Compute and return metrics
    results = {name: metric.compute().item() for name, metric in metrics.items()}
    results['loss'] = loss_meter.compute().item()
    
    return results


def train(config):
    """Main training function."""
    # Create log folder
    uid = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    log_folder = Path(config.log_folder) / uid
    log_folder.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_folder / 'training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    
    # Log config
    logger.info(f"Starting training with config:\n{json.dumps(vars(config), indent=4)}")
    with open(log_folder / 'config.json', 'w') as f:
        json.dump(vars(config), f, indent=4)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = create_train_loader(config)
    val_loader = create_val_loader(config)
    
    # Create model
    logger.info(f"Creating {config.arch} model...")
    model = create_model(config)
    
    # Create optimizer and loss function
    logger.info(f"Setting up {config.optimizer} optimizer and loss function...")
    optimizer, loss_fn = create_optimizer_and_loss(model, config)
    
    # Optionally load checkpoint
    start_epoch = 0
    if config.path:
        logger.info(f"Loading checkpoint from {config.path}")
        checkpoint = torch.load(config.path, map_location=config.device)
        model.load_state_dict(checkpoint)
    
    # Eval only mode
    if config.eval_only:
        logger.info("Evaluation only mode")
        results = validate(model, val_loader, loss_fn, config)
        for k, v in results.items():
            logger.info(f"{k}: {v:.4f}")
        return
    
    # Training loop
    best_acc = 0.0
    start_time = time.time()
    
    # Calculate and log estimated training time
    total_iterations = config.epochs * len(train_loader)
    logger.info(f"Starting training for {config.epochs} epochs ({total_iterations} iterations)")
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time.time()
        
        # Update resolution if needed
        if config.start_ramp < config.end_ramp:
            res = get_resolution(epoch, config)
            logger.info(f"Epoch {epoch} - Using resolution {res}")
        
        # Train for one epoch
        logger.info(f"Epoch {epoch}/{config.epochs-1} - Training...")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, epoch, config)
        
        # Validate
        logger.info(f"Epoch {epoch}/{config.epochs-1} - Validating...")
        val_results = validate(model, val_loader, loss_fn, config)
        
        # Determine best accuracy
        if config.mrl or config.efficient:
            # Use the largest dimension for best accuracy
            largest_dim = max(config.nesting_list)
            current_acc = val_results[f'top1_{largest_dim}']
        else:
            current_acc = val_results['top1']
        
        # Save best model
        is_best = current_acc > best_acc
        best_acc = max(best_acc, current_acc)
        
        # Log results
        elapsed = time.time() - start_time
        epoch_time = time.time() - epoch_start_time
        estimated_remaining = epoch_time * (config.epochs - epoch - 1)
        
        log_dict = {
            'epoch': epoch,
            'train_loss': train_loss,
            'elapsed_time': elapsed,
            'epoch_time': epoch_time,
            'estimated_remaining': estimated_remaining,
            'current_lr': optimizer.param_groups[0]['lr'],
            **val_results
        }
        
        # Format remaining time for display
        remaining_time_str = format_time(estimated_remaining)
        logger.info(f"Epoch {epoch}/{config.epochs-1} completed in {epoch_time:.1f}s. Estimated remaining: {remaining_time_str}")
        logger.info(f"Results: {log_dict}")
        
        # Save to log file
        with open(log_folder / 'log.json', 'a') as f:
            f.write(json.dumps(log_dict) + '\n')
        
        # Save checkpoint
        torch.save(model.state_dict(), log_folder / 'latest_model.pt')
        if is_best:
            logger.info(f"New best accuracy: {best_acc:.4f}")
            torch.save(model.state_dict(), log_folder / 'best_model.pt')
    
    # Save final model
    torch.save(model.state_dict(), log_folder / 'final_model.pt')
    total_time = time.time() - start_time
    logger.info(f"Training completed in {format_time(total_time)}. Best accuracy: {best_acc:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Matryoshka Representation Learning with PyTorch')
    
    # Add config file argument
    parser.add_argument('--config', type=str, help='Path to config file (YAML or JSON)')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation only mode')
    parser.add_argument('--path', type=str, help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Create config
    if args.config:
        with open(args.config, 'r') as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        config = Config(**config_dict)
    else:
        config = Config()
    
    # Override config with command line arguments
    if args.eval_only:
        config.eval_only = True
    if args.path:
        config.path = args.path
    
    # Run training
    train(config)


if __name__ == '__main__':
    main()