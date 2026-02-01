"""
Main package initializer for Security Micro LLM.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "Micro LLM for MCP Agent Security Screening"

from src.model import PromptSecurityClassifier, LightweightSecurityClassifier
from src.inference import SecurityInferenceEngine

__all__ = [
    'PromptSecurityClassifier',
    'LightweightSecurityClassifier',
    'SecurityInferenceEngine',
]
