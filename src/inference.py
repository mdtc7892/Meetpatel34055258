"""
Inference module for the RAA storytelling project
Contains only generation/inference functionality
"""
import torch
from transformers import T5Tokenizer
import yaml


def generate_story(model, tokenizer, input_text, max_length=128, device='cpu'):
    """
    Generate a story continuation given an input prompt
    
    Args:
        model: Trained storytelling model
        tokenizer: T5 tokenizer
        input_text: Input text to continue
        max_length: Maximum length of generated text
        device: Device to run inference on
    
    Returns:
        Generated story text
    """
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        input_text,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            temperature=0.8,
            do_sample=True
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def load_model_for_inference(model_class, config_path, checkpoint_path=None, device='cpu'):
    """
    Load a trained model for inference
    
    Args:
        model_class: Model class to instantiate
        config_path: Path to config file
        checkpoint_path: Path to trained checkpoint (optional)
        device: Device to load model on
    
    Returns:
        Loaded model ready for inference
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model instance
    model = model_class(config)
    model = model.to(device)
    
    # Load checkpoint if provided
    if checkpoint_path and torch.load(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.eval()
    return model


def batch_generate(model, tokenizer, input_texts, max_length=128, device='cpu'):
    """
    Generate stories for a batch of input texts
    
    Args:
        model: Trained storytelling model
        tokenizer: T5 tokenizer
        input_texts: List of input texts to continue
        max_length: Maximum length of generated text
        device: Device to run inference on
    
    Returns:
        List of generated story texts
    """
    model.eval()
    
    # Tokenize inputs
    inputs = tokenizer(
        input_texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            temperature=0.8,
            do_sample=True
        )
    
    # Decode outputs
    generated_texts = []
    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts


if __name__ == "__main__":
    print("Inference module ready for use.")