"""
Data loading module for the RAA storytelling project
"""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer
import yaml
import os


class StoryDataset(Dataset):
    """Simple dataset for story generation"""
    
    def __init__(self, texts, targets, tokenizer, max_length=512):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = str(self.targets[idx])
        
        # Tokenize input and target
        source_encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels (for T5, we shift the target for teacher forcing)
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'decoder_input_ids': target_encoding['input_ids'].squeeze(),
            'labels': labels
        }


def get_data_loaders(config_path):
    """Create data loaders for training and validation"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config['model']['base_model'])
    
    # Get pad token ID
    pad_idx = tokenizer.pad_token_id
    
    # Create dummy data for demonstration (since actual data might not be available)
    dummy_texts = [
        "The sun was setting over the horizon.",
        "She walked through the dark forest.",
        "The old castle stood on the hill.",
        "In the distance, a wolf howled.",
        "The knight drew his sword.",
        "The dragon breathed fire.",
        "A princess waited in the tower.",
        "The ship sailed across the ocean."
    ]
    
    dummy_targets = [
        "The sky turned orange and red.",
        "Strange sounds echoed around her.",
        "Its walls were covered in moss.",
        "The night felt eerie and cold.",
        "He was ready for battle.",
        "The village was in danger.",
        "She dreamed of adventure.",
        "Pirates were approaching."
    ]
    
    # Create datasets
    train_dataset = StoryDataset(dummy_texts, dummy_targets, tokenizer, config['data']['max_source_length'])
    val_dataset = StoryDataset(dummy_texts[:2], dummy_targets[:2], tokenizer, config['data']['max_source_length'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0  # Using 0 to avoid potential issues in some environments
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0  # Using 0 to avoid potential issues in some environments
    )
    
    return train_loader, val_loader, tokenizer, pad_idx


if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...")
    train_loader, val_loader, tokenizer, pad_idx = get_data_loaders('config.yaml')
    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length: {len(val_loader)}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print("Data loader test completed successfully!")