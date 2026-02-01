"""
Security classifier model for prompt safety detection.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification
from typing import Dict, Optional
import warnings


class PromptSecurityClassifier(nn.Module):
    """
    Locally finetuTransformer-based classifier for detecting malicious prompts.
    Uses a pre-trained language model with a classification head.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_base: bool = False
    ):
        """
        Initialize the security classifier.
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of output classes (2 for binary)
            dropout: Dropout probability
            freeze_base: Whether to freeze base model weights
        """
        super().__init__()
        
        print(f"Loading base model: {model_name}")
        
        try:
            # Try to load as a causal LM or sequence classification model
            self.config = AutoConfig.from_pretrained(model_name)
            
            # Check if it's a causal LM (like Gemma, Phi, TinyLlama)
            is_causal_lm = hasattr(self.config, 'is_decoder') and self.config.is_decoder
            
            if is_causal_lm or 'gemma' in model_name.lower() or 'phi' in model_name.lower() or 'llama' in model_name.lower():
                print("  Detected causal LM - using custom classification head")
                # Load base model without LM head
                from transformers import AutoModelForCausalLM
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32  # Use float32 for numerical stability
                )
                # Get the base transformer
                if hasattr(base_model, 'model'):
                    self.transformer = base_model.model
                elif hasattr(base_model, 'transformer'):
                    self.transformer = base_model.transformer
                else:
                    self.transformer = base_model
                
                # Get hidden size
                if hasattr(self.config, 'hidden_size'):
                    hidden_size = self.config.hidden_size
                elif hasattr(self.config, 'd_model'):
                    hidden_size = self.config.d_model
                else:
                    hidden_size = 768
            else:
                # Standard encoder model (BERT-like)
                print("  Using encoder model")
                self.transformer = AutoModel.from_pretrained(model_name)
                hidden_size = self.config.hidden_size
            
        except Exception as e:
            print(f"  Warning: {e}")
            print("  Falling back to standard AutoModel")
            self.config = AutoConfig.from_pretrained(model_name)
            self.transformer = AutoModel.from_pretrained(model_name)
            hidden_size = self.config.hidden_size

        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
            print("  Base model frozen")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # Match classifier dtype to transformer
        if hasattr(self.transformer, 'dtype'):
            self.classifier.to(self.transformer.dtype)
        
        # Enable gradient checkpointing if supported (trades compute for memory)
        if hasattr(self.transformer, 'gradient_checkpointing_enable'):
            self.transformer.gradient_checkpointing_enable()
            print("  Enabled gradient checkpointing")
        
        self.num_labels = num_labels
        print(f"  Model loaded with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size] (optional)
        
        Returns:
            Dictionary with logits, loss (if labels provided), and probabilities
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get pooled representation
        # Use mean pooling for all models (works best for classification)
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
            # Mean pooling: average over all non-padded tokens
            pooled_output = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Fallback to mean pooling
            hidden_states = outputs[0]
            pooled_output = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        
        # Get logits from classifier
        logits = self.classifier(pooled_output)
        
        # Calculate probabilities
        probs = torch.softmax(logits, dim=-1)
        
        result = {
            'logits': logits,
            'probabilities': probs
        }
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result['loss'] = loss
        
        return result
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with confidence scores.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            threshold: Classification threshold
        
        Returns:
            Dictionary with predictions and confidence scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probs = outputs['probabilities']
            
            # Get predicted class and confidence
            confidence, predicted = torch.max(probs, dim=-1)
            
            # Check if malicious class probability exceeds threshold
            malicious_prob = probs[:, 1]  # Assuming label 1 is malicious
            is_malicious = malicious_prob >= threshold
        
        return {
            'predictions': predicted,
            'confidence': confidence,
            'is_malicious': is_malicious,
            'malicious_probability': malicious_prob,
            'safe_probability': probs[:, 0]
        }


class LightweightSecurityClassifier(nn.Module):
    """
    Extremely lightweight classifier for edge deployment.
    Uses a smaller architecture for faster inference.
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_labels: int = 2,
        max_length: int = 128
    ):
        """
        Initialize lightweight classifier.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_labels: Number of output classes
            max_length: Maximum sequence length
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_labels)
        )
        
        self.num_labels = num_labels
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Embed tokens
        embeddings = self.embedding(input_ids)
        
        # LSTM encoding
        lstm_out, (hidden, _) = self.lstm(embeddings)
        
        # Use final hidden state
        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        
        # Classify
        logits = self.classifier(hidden)
        probs = torch.softmax(logits, dim=-1)
        
        result = {
            'logits': logits,
            'probabilities': probs
        }
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result['loss'] = loss
        
        return result


if __name__ == "__main__":
    # Test model instantiation
    print("Testing PromptSecurityClassifier...")
    model = PromptSecurityClassifier()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTesting LightweightSecurityClassifier...")
    light_model = LightweightSecurityClassifier()
    print(f"Model parameters: {sum(p.numel() for p in light_model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 2, (batch_size,))
    
    print("\nTesting forward pass...")
    outputs = model(input_ids, attention_mask, labels)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
