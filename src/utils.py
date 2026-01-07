import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from datetime import datetime

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """Handle GPU/CPU compatibility."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model and optimizer state."""
    print(f"Saving checkpoint to {path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Prepare state dict with only RAA parameters
    state_dict = {}
    for name, param in model.named_parameters():
        if 'raa' in name or 'reasoning' in name or 'projection' in name:
            state_dict[name] = param.cpu()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path, device):
    """Load model and optimizer state from checkpoint."""
    if not os.path.exists(path):
        print(f"Checkpoint not found at {path}. Starting from scratch.")
        return 0, float('inf')
        
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device)
    
    # Load only the RAA parameters
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    return epoch, loss

def plot_loss_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Loss curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def generate_attention_heatmap(attention_weights, save_path=None):
    """Generate attention heatmap visualization."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights[0, 0].detach().cpu().numpy(), cmap='viridis', aspect='auto')
    plt.title('Attention Weights Heatmap (First Head, First Sample)')
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.colorbar()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Attention heatmap saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def save_sample_stories(stories, save_path):
    """Save sample generated stories to file."""
    with open(save_path, 'w', encoding='utf-8') as f:
        for i, story in enumerate(stories):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Input: {story.get('input', 'N/A')}\n")
            f.write(f"Generated: {story.get('generated', 'N/A')}\n")
            f.write(f"Reference: {story.get('reference', 'N/A')}\n")
            f.write("-" * 50 + "\n")

def format_metrics(metrics, save_path=None):
    """Format evaluation metrics for display or saving."""
    formatted_str = "Model Evaluation Metrics\n"
    formatted_str += "=" * 50 + "\n"
    
    if 'bleu_score' in metrics:
        formatted_str += f"BLEU Score: {metrics['bleu_score']:.4f}\n"
    
    if 'rouge_scores' in metrics:
        formatted_str += "\nROUGE Scores:\n"
        for rouge_type, score in metrics['rouge_scores'].items():
            formatted_str += f"  {rouge_type}: {score:.4f}\n"
    
    if 'coherence_metrics' in metrics:
        formatted_str += "\nCoherence Metrics:\n"
        for metric, score in metrics['coherence_metrics'].items():
            formatted_str += f"  {metric}: {score:.4f}\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(formatted_str)
        print(f"Metrics saved to {save_path}")
    
    return formatted_str

def create_directories():
    """Create necessary directories for the project."""
    dirs = [
        './checkpoints',
        './results',
        './pretrained_models',
        './src'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory {dir_path} created/verified")

def download_pretrained_model(model_name="t5-base", save_path="./pretrained_models/t5-base"):
    """Download and save a pretrained model."""
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    
    print(f"Downloading {model_name}...")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Create directory
    os.makedirs(save_path, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"Model and tokenizer saved to {save_path}")

def print_model_info(model):
    """Print information about the model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
    
    # Print parameter names that are trainable
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,} parameters")