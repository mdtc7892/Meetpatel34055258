import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml
import random
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup

from model import create_model
from data_loader import get_data_loaders
from utils import set_seed, get_device, save_checkpoint, load_checkpoint
import evaluate

def train_step(model, batch, device, criterion):
    """Single training step."""
    model.train()
    
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    decoder_input_ids = batch['decoder_input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        labels=labels
    )
    
    loss = outputs['loss']
    
    return loss

def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, 
                accumulation_steps=1, scheduler=None):
    """Full training loop for one epoch."""
    model.train()
    total_loss = 0
    total_steps = 0
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(progress_bar):
        loss = train_step(model, batch, device, criterion)
        
        # Scale the loss for mixed precision
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            
            if scheduler:
                scheduler.step()
        
        total_loss += loss.item()
        total_steps += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Handle remaining gradients if accumulation_steps doesn't divide the total number of batches
    if (i + 1) % accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / total_steps
    return avg_loss

def validate(model, dataloader, criterion, device):
    """Full validation loop."""
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
            total_steps += 1
            
    avg_loss = total_loss / total_steps
    return avg_loss

def train_fast(model, train_loader, val_loader, config_path):
    """Fast training function that only trains RAA layers."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Train ONLY RAA parameters
    trainable_params = []
    trainable_param_names = []
    for name, param in model.named_parameters():
        if 'raa' in name or 'reasoning' in name or 'projection' in name:
            trainable_params.append(param)
            trainable_param_names.append(name)
            param.requires_grad = True
        else:
            param.requires_grad = False  # Ensure base model stays frozen
    
    print(f"Trainable parameters: {len(trainable_param_names)}")
    print(f"Trainable parameter names: {trainable_param_names}")
    
    # Optimizer for only trainable parameters
    optimizer = torch.optim.Adam(trainable_params, lr=config['training']['learning_rate'])
    
    # Scheduler
    total_steps = len(train_loader) * config['training']['epochs']
    warmup_steps = config['training']['warmup_steps']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Criterion
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    epochs = config['training']['epochs']
    accumulation_steps = config['training']['gradient_accumulation_steps']
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Training
        train_loss = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            device, 
            accumulation_steps=accumulation_steps,
            scheduler=scheduler
        )
        
        # Validation
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save checkpoint
        checkpoint_path = os.path.join(config['paths']['checkpoints'], "raa_model.pth")
        os.makedirs(config['paths']['checkpoints'], exist_ok=True)
        save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_path)
    
    print("\nTraining complete.")
    return train_losses, val_losses

def train_model(config_path='config.yaml'):
    """Complete training function."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup
    set_seed(config['training']['seed'] if 'seed' in config['training'] else 42)
    
    # Create model
    print("Loading pre-trained model with RAA enhancements...")
    model = create_model(config_path)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, tokenizer = get_data_loaders(config_path)
    
    # Train the model
    print("Starting training...")
    train_losses, val_losses = train_fast(model, train_loader, val_loader, config_path)
    
    # Save final results
    results_path = os.path.join(config['paths']['results'], "loss_curves.pt")
    os.makedirs(config['paths']['results'], exist_ok=True)
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses
    }, results_path)
    
    return train_losses, val_losses

if __name__ == '__main__':
    # Run training
    train_losses, val_losses = train_model('config.yaml')
    
    print("Final training losses:", train_losses)
    print("Final validation losses:", val_losses)