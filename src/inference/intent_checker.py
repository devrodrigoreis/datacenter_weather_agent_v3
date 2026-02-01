"""
Real-time inference module for intent classification.
Determines if queries are about weather/location or off-topic.
"""

import torch
from transformers import AutoTokenizer
from pathlib import Path
from typing import Dict, List, Any
import yaml
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model import PromptSecurityClassifier


def get_device():
    """Get the best available device for inference."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'  # Apple Silicon
    else:
        return 'cpu'


class IntentInferenceEngine:
    """
    Fast inference engine for intent classification.
    Optimized for low latency in production environments.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config.yaml",
        device: str | None = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            device: Device to run inference on (auto-detect if None)
        """
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Get intent model config
        intent_config = self.config.get('intent_model', self.config['model'])
        
        # Set device
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            intent_config['name']
        )
        
        # Load model
        self.model = PromptSecurityClassifier(
            model_name=intent_config['name'],
            num_labels=intent_config['num_labels'],
            dropout=intent_config['dropout']
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get max length
        self.max_length = intent_config.get('max_length', 128)
        
        print(f"Intent inference engine initialized on {self.device}")
    
    def check_intent(self, query: str) -> Dict[str, Any]:
        """
        Check if a query is about weather/location.
        
        Args:
            query: Input query text
        
        Returns:
            Dictionary with classification results
        """
        # Tokenize
        encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            
            predicted_label = int(torch.argmax(probs, dim=-1).item())
            confidence = probs[0][predicted_label].item()
        
        # Label 1 = weather/location, 0 = off-topic
        is_weather_location = (predicted_label == 1)
        weather_prob = probs[0][1].item()
        offtopic_prob = probs[0][0].item()
        
        result = {
            'query': query,
            'is_weather_location': is_weather_location,
            'is_off_topic': not is_weather_location,
            'confidence': confidence,
            'weather_probability': weather_prob,
            'offtopic_probability': offtopic_prob,
            'category': 'WEATHER/LOCATION' if is_weather_location else 'OFF-TOPIC'
        }
        
        return result
    
    def check_intents_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Check multiple queries in a batch for efficiency.
        
        Args:
            queries: List of query texts
        
        Returns:
            List of classification results
        """
        if not queries:
            return []
        
        # Tokenize batch
        encodings = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Batch inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            
            predicted_labels = torch.argmax(probs, dim=-1)
        
        # Process results
        results = []
        for i, query in enumerate(queries):
            predicted_label = int(predicted_labels[i].item())
            confidence = probs[i][predicted_label].item()
            weather_prob = probs[i][1].item()
            offtopic_prob = probs[i][0].item()
            
            is_weather_location = (predicted_label == 1)
            
            results.append({
                'query': query,
                'is_weather_location': is_weather_location,
                'is_off_topic': not is_weather_location,
                'confidence': confidence,
                'weather_probability': weather_prob,
                'offtopic_probability': offtopic_prob,
                'category': 'WEATHER/LOCATION' if is_weather_location else 'OFF-TOPIC'
            })
        
        return results
    
    @torch.no_grad()
    def benchmark_latency(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Benchmark inference latency.
        
        Args:
            num_samples: Number of samples to test
        
        Returns:
            Latency statistics in milliseconds
        """
        import time
        
        # Generate dummy input
        dummy_query = "What is the weather forecast for today?"
        
        latencies = []
        for _ in range(num_samples):
            start = time.perf_counter()
            _ = self.check_intent(dummy_query)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return {
            'mean_ms': sum(latencies) / len(latencies),
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'p50_ms': sorted(latencies)[len(latencies) // 2],
            'p95_ms': sorted(latencies)[int(len(latencies) * 0.95)],
            'p99_ms': sorted(latencies)[int(len(latencies) * 0.99)],
        }


def main():
    """Example usage of intent inference engine."""
    # Check if model exists
    model_path = Path("models/intent_checkpoints/best_intent_model.pt")
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first: python src/training/train_intent.py")
        return
    
    # Initialize engine
    print("Loading intent classifier...")
    engine = IntentInferenceEngine(str(model_path))
    
    # Test queries
    test_queries = [
        # Weather/Location queries (should be classified as on-topic)
        "What's the weather forecast?",
        "Where is the datacenter located?",
        "What's the temperature today?",
        "Show me the coordinates",
        "Is it going to rain?",
        "What are the weather conditions?",
        
        # Off-topic queries (should be classified as off-topic)
        "How do I bake a cake?",
        "What is machine learning?",
        "Tell me a joke",
        "Explain quantum physics",
        "Write a Python script",
    ]
    
    print("\nTesting intent classification:\n")
    for query in test_queries:
        result = engine.check_intent(query)
        status = "✓ ON-TOPIC" if result['is_weather_location'] else "✗ OFF-TOPIC"
        print(f"{status} [{result['category']}] (confidence: {result['confidence']:.2%})")
        print(f"  Weather Prob: {result['weather_probability']:.4f}")
        print(f"  Query: {query}")
        print()
    
    # Benchmark
    print("\nBenchmarking latency...")
    latency = engine.benchmark_latency(num_samples=50)
    print(f"Mean latency: {latency['mean_ms']:.2f}ms")
    print(f"P95 latency: {latency['p95_ms']:.2f}ms")
    print(f"P99 latency: {latency['p99_ms']:.2f}ms")


if __name__ == "__main__":
    main()
