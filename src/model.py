import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers import T5ForConditionalGeneration, T5Config

@dataclass
class ModelConfig:
    """Configuration dataclass for reproducibility"""
    d_model: int = 512
    n_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 512
    n_layers: int = 6
    
class ReasoningAwareAttention(nn.Module):
    """Production-quality RAA implementation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_k = config.d_model // config.n_heads
        
        # Professional initialization
        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)
        self.W_r = nn.Linear(config.d_model, config.d_model)  # Reasoning projection
        self.W_o = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Xavier initialization
        self._init_parameters()
        
    def _init_parameters(self):
        """Proper weight initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, 
                reasoning_state: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            reasoning_state: Reasoning state [batch_size, d_model]
            mask: Optional attention mask
            
        Returns:
            output: Contextualized representations
            attention_weights: For visualization
        """
        batch_size = x.size(0)
        
        # Projections
        Q = self.W_q(x).view(batch_size, -1, self.config.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.config.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.config.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores with reasoning bias
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Incorporate reasoning state (KEY INNOVATION)
        R = self.W_r(reasoning_state).view(batch_size, 1, self.config.n_heads, self.d_k)
        R = R.transpose(1, 2)  # [batch_size, n_heads, 1, d_k]
        scores = scores + torch.matmul(Q, R.transpose(-2, -1))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Context
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.config.d_model
        )
        
        # Output projection
        output = self.W_o(context)
        output = self.dropout(output)
        output = self.layer_norm(output + x)  # Residual connection
        
        return output, attn_weights.detach()

class StoryReasoningState(nn.Module):
    """Explicit reasoning state module with GRU-like mechanism"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # State initialization network
        self.init_linear = nn.Linear(d_model, d_model)
        
        # State update network (GRU-like gates)
        self.update_gate = nn.Linear(d_model * 2, d_model)  # [context, previous_state]
        self.reset_gate = nn.Linear(d_model * 2, d_model)
        self.new_state_candidate = nn.Linear(d_model * 2, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, encoder_output: torch.Tensor, 
                previous_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            encoder_output: [batch_size, seq_len, d_model]
            previous_state: [batch_size, d_model] or None
            
        Returns:
            reasoning_state: [batch_size, d_model]
        """
        # Get context vector
        context = encoder_output.mean(dim=1)  # [batch_size, d_model]
        
        if previous_state is None:
            # Initialize state from context
            initial_state = torch.tanh(self.init_linear(context))
            return self.layer_norm(initial_state)
        
        # GRU-like update
        combined = torch.cat([context, previous_state], dim=-1)  # [batch_size, 2 * d_model]
        
        z = torch.sigmoid(self.update_gate(combined))  # Update gate
        r = torch.sigmoid(self.reset_gate(combined))   # Reset gate
        
        # New state calculation
        r_state = r * previous_state
        combined_candidate = torch.cat([context, r_state], dim=-1)
        h_candidate = torch.tanh(self.new_state_candidate(combined_candidate))
        
        # Final state
        new_state = (1 - z) * previous_state + z * h_candidate
        
        return self.layer_norm(new_state)

class FastStoryModel(nn.Module):
    """Complete storytelling model with pre-trained base"""
    
    def __init__(self, base_model_name: str = "t5-base", config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        
        # Load pre-trained model (FAST SETUP)
        self.base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
        
        # Freeze base model (CRITICAL FOR SPEED)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Add our innovations
        self.raa_layers = nn.ModuleList([
            ReasoningAwareAttention(self.config) for _ in range(2)
        ])
        
        self.reasoning_state_proj = nn.Linear(
            self.base_model.config.d_model, 
            self.config.d_model
        )
        
        # Story reasoning state module
        self.story_reasoning_state = StoryReasoningState(self.config.d_model)
        
        # Training statistics
        self.register_buffer('training_steps', torch.tensor(0))
        
    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None):
        """Forward pass with RAA enhancement"""
        # Get base encoder outputs
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Initialize reasoning state
        reasoning_state = self.story_reasoning_state(encoder_outputs.last_hidden_state, previous_state=None)
        
        # Enhance with RAA
        hidden_states = encoder_outputs.last_hidden_state
        
        for raa_layer in self.raa_layers:
            hidden_states, _ = raa_layer(hidden_states, reasoning_state)
            
        # Generate with enhanced representations
        outputs = self.base_model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True
        )
        
        self.training_steps += 1
        return outputs

def create_model(config_path: str = 'config.yaml'):
    """Helper function to create the model"""
    import yaml
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    model_config = ModelConfig(
        d_model=config_dict['model']['d_model'],
        n_heads=config_dict['model']['n_heads'],
        dropout=config_dict['model']['dropout'],
        max_seq_len=config_dict['data']['max_source_length']
    )
    
    return FastStoryModel(
        base_model_name=config_dict['model']['base_model'],
        config=model_config
    )