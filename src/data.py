"""
Data loading and preprocessing for prompt security classifier.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class PromptDataset(Dataset):
    """Dataset for prompt security classification."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            data: List of dicts with 'text' and 'label' keys
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item."""
        item = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


class DataLoaderFactory:
    """Factory for creating data loaders."""
    
    def __init__(self, config: Dict):
        """
        Initialize factory.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Load tokenizer
        model_name = config['model']['name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Data config
        self.max_length = int(config['model'].get('max_length', 512))
        self.batch_size = int(config['training']['batch_size'])
        self.num_workers = int(config['training'].get('num_workers', 0))
    
    def load_json_data(self, file_path: str) -> List[Dict]:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of data samples
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate data format
        if not isinstance(data, list):
            raise ValueError("Data must be a list of samples")
        
        for i, sample in enumerate(data):
            if 'text' not in sample or 'label' not in sample:
                raise ValueError(f"Sample {i} missing 'text' or 'label' field")
        
        return data
    
    def create_splits(
        self,
        data: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split data into train/val/test sets.
        
        Args:
            data: List of data samples
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Use config ratios if available
        train_ratio = float(self.config.get('data', {}).get('train_ratio', train_ratio))
        val_ratio = float(self.config.get('data', {}).get('val_ratio', val_ratio))
        
        # Calculate split indices
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data
    
    def create_dataloaders(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict]
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders for train/val/test sets.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = PromptDataset(train_data, self.tokenizer, self.max_length)
        val_dataset = PromptDataset(val_data, self.tokenizer, self.max_length)
        test_dataset = PromptDataset(test_data, self.tokenizer, self.max_length)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader, test_loader
