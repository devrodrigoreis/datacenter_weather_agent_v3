"""
Tier 1: Context Compression for Multi-Agent System

This module provides context window management to prevent token overflow.
It compresses messages, caches tool results, and tracks token usage.

Usage:
    from src.memory.context_manager import ContextManager
    
    manager = ContextManager(max_tokens=28000)
    compressed_state = manager.compress_if_needed(state)
"""

import tiktoken
from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from datetime import datetime
import json
from pathlib import Path


# Extended state definition with context management fields
class ManagedState(TypedDict):
    """State with context management fields."""
    # Original fields (from SharedState)
    question: str
    is_safe_query: bool | None
    security_threat_type: str | None
    is_weather_question: bool | None
    public_ip: str | None
    latitude: float | None
    longitude: float | None
    weather_data: str | None
    answer: str | None
    output_safe: bool | None
    next_agent: str | None
    current_agent: str | None
    error: str | None
    messages: list[BaseMessage]
    
    # NEW: Context management fields
    conversation_summary: str | None
    tool_results_cache: dict[str, str] | None
    context_token_count: int | None
    compression_history: list[dict] | None


class ContextManager:
    """
    Manages context window to prevent token overflow.
    
    Features:
    - Token counting
    - Message compression
    - Tool result deduplication
    - Automatic summarization
    """
    
    def __init__(
        self, 
        max_tokens: int = 28000,
        max_messages: int = 10,
        summarize_threshold: int = 20000,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize context manager.
        
        Args:
            max_tokens: Maximum tokens allowed in context
            max_messages: Maximum number of messages to keep
            summarize_threshold: Token count that triggers summarization
            model: Model name for token counting (tiktoken)
        """
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.summarize_threshold = summarize_threshold
        
        # Initialize token counter
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base (GPT-4/GPT-3.5)
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Stats
        self.compression_count = 0
        self.total_tokens_saved = 0
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(self.encoding.encode(text))
    
    def count_message_tokens(self, messages: List[BaseMessage]) -> int:
        """Count total tokens in message list."""
        total = 0
        for msg in messages:
            # Add role prefix tokens
            total += 4  # Approximate overhead per message
            total += self.estimate_tokens(msg.content)
        return total
    
    def count_state_tokens(self, state: Dict[str, Any]) -> int:
        """Count total tokens in state."""
        total = 0
        
        # Count messages
        if 'messages' in state and state['messages']:
            total += self.count_message_tokens(state['messages'])
        
        # Count tool results
        if 'tool_results_cache' in state and state['tool_results_cache']:
            for key, value in state['tool_results_cache'].items():
                total += self.estimate_tokens(f"{key}: {value}")
        
        # Count conversation summary
        if 'conversation_summary' in state and state['conversation_summary']:
            total += self.estimate_tokens(state['conversation_summary'])
        
        # Count other text fields
        text_fields = ['question', 'answer', 'weather_data', 'error', 
                      'security_threat_type']
        for field in text_fields:
            if field in state and state[field]:
                total += self.estimate_tokens(str(state[field]))
        
        return total
    
    def compress_if_needed(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress state if it exceeds token limits.
        
        This is the main entry point for context management.
        """
        # Initialize context management fields if not present
        if 'context_token_count' not in state:
            state['context_token_count'] = 0
        if 'tool_results_cache' not in state:
            state['tool_results_cache'] = {}
        if 'compression_history' not in state:
            state['compression_history'] = []
        if 'conversation_summary' not in state:
            state['conversation_summary'] = None
        
        # Count current tokens
        current_tokens = self.count_state_tokens(state)
        state['context_token_count'] = current_tokens
        
        # Check if compression needed
        if current_tokens <= self.summarize_threshold:
            return state
        
        print(f"\n[CONTEXT MANAGER] Token count: {current_tokens}/{self.max_tokens}")
        print(f"  Compression needed - optimizing context...")
        
        # Apply compression strategies in order of priority
        state = self._deduplicate_tool_results(state)
        state = self._prune_old_messages(state)
        state = self._compress_tool_results(state)
        
        # Recount after compression
        new_token_count = self.count_state_tokens(state)
        tokens_saved = current_tokens - new_token_count
        
        self.compression_count += 1
        self.total_tokens_saved += tokens_saved
        
        # Log compression
        state['compression_history'].append({
            'timestamp': datetime.now().isoformat(),
            'tokens_before': current_tokens,
            'tokens_after': new_token_count,
            'tokens_saved': tokens_saved,
            'strategy': 'deduplicate_prune_compress'
        })
        
        print(f"  ✓ Compressed: {current_tokens} → {new_token_count} tokens")
        print(f"  ✓ Saved: {tokens_saved} tokens ({tokens_saved/current_tokens*100:.1f}%)")
        
        state['context_token_count'] = new_token_count
        
        return state
    
    def _deduplicate_tool_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicate tool results and cache them."""
        if 'messages' not in state or not state['messages']:
            return state
        
        tool_cache = state.get('tool_results_cache', {})
        seen_results = set()
        filtered_messages = []
        
        for msg in state['messages']:
            # Check if this is a tool result message
            if isinstance(msg, AIMessage) and "Tool:" in msg.content:
                # Create a hash of the result
                result_hash = hash(msg.content[:100])  # Hash first 100 chars
                
                if result_hash not in seen_results:
                    seen_results.add(result_hash)
                    filtered_messages.append(msg)
                    # Cache the result
                    tool_cache[f"result_{result_hash}"] = msg.content
                # else: skip duplicate
            else:
                filtered_messages.append(msg)
        
        state['messages'] = filtered_messages
        state['tool_results_cache'] = tool_cache
        
        return state
    
    def _prune_old_messages(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep only recent messages, summarize the rest.
        
        Keeps the most recent N messages and creates a summary of older ones.
        """
        if 'messages' not in state or not state['messages']:
            return state
        
        messages = state['messages']
        
        if len(messages) <= self.max_messages:
            return state
        
        # Split into old and recent
        old_messages = messages[:-self.max_messages]
        recent_messages = messages[-self.max_messages:]
        
        # Create summary of old messages
        summary_text = self._create_message_summary(old_messages)
        
        # Update state
        if state.get('conversation_summary'):
            # Append to existing summary
            state['conversation_summary'] += f"\n\n{summary_text}"
        else:
            state['conversation_summary'] = summary_text
        
        # Keep only recent messages
        state['messages'] = recent_messages
        
        return state
    
    def _create_message_summary(self, messages: List[BaseMessage]) -> str:
        """
        Create a concise summary of messages.
        
        This is a simple rule-based summarization. For production,
        you could use an LLM to create better summaries.
        """
        summary_parts = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                # Extract user questions
                content = msg.content[:100]  # First 100 chars
                summary_parts.append(f"User asked: {content}")
            elif isinstance(msg, AIMessage):
                # Extract key agent responses
                if "safe" in msg.content.lower() or "threat" in msg.content.lower():
                    summary_parts.append(f"Security: {msg.content[:80]}")
                elif "weather" in msg.content.lower():
                    summary_parts.append(f"Weather: {msg.content[:80]}")
        
        # Limit summary size
        if len(summary_parts) > 5:
            summary_parts = summary_parts[:2] + ["..."] + summary_parts[-2:]
        
        return " | ".join(summary_parts)
    
    def _compress_tool_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress verbose tool results.
        
        Some tools return large JSON/text outputs. This function
        extracts only the essential information.
        """
        if 'tool_results_cache' not in state or not state['tool_results_cache']:
            return state
        
        compressed_cache = {}
        
        for key, value in state['tool_results_cache'].items():
            # Try to parse as JSON and extract key fields
            try:
                if value.startswith('{') or value.startswith('['):
                    data = json.loads(value)
                    # Extract only essential fields
                    compressed = self._extract_essential_fields(data)
                    compressed_cache[key] = json.dumps(compressed)
                else:
                    # For text results, keep first N characters
                    compressed_cache[key] = value[:200] + "..." if len(value) > 200 else value
            except json.JSONDecodeError:
                # Keep as-is if not JSON
                compressed_cache[key] = value[:200] + "..." if len(value) > 200 else value
        
        state['tool_results_cache'] = compressed_cache
        
        return state
    
    def _extract_essential_fields(self, data: Any) -> Any:
        """Extract only essential fields from tool results."""
        if isinstance(data, dict):
            # Keep only important fields
            essential_keys = ['ip', 'latitude', 'longitude', 'temperature', 
                            'weather', 'description', 'error', 'result']
            return {k: v for k, v in data.items() if k in essential_keys}
        elif isinstance(data, list):
            # For lists, keep only first few items
            return data[:3] if len(data) > 3 else data
        else:
            return data
    
    def get_stats(self) -> Dict[str, int]:
        """Get compression statistics."""
        return {
            'compression_count': self.compression_count,
            'total_tokens_saved': self.total_tokens_saved,
            'avg_tokens_saved': self.total_tokens_saved // max(self.compression_count, 1)
        }
    
    def save_stats(self, filepath: str = "memory/context_stats.json"):
        """Save compression statistics to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        stats = {
            **self.get_stats(),
            'max_tokens': self.max_tokens,
            'max_messages': self.max_messages,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)


# Utility function for easy integration
def compress_state(state: Dict[str, Any], 
                   max_tokens: int = 28000,
                   max_messages: int = 10) -> Dict[str, Any]:
    """
    Convenience function to compress state in one call.
    
    Usage:
        state = compress_state(state)
    """
    manager = ContextManager(max_tokens=max_tokens, max_messages=max_messages)
    return manager.compress_if_needed(state)


if __name__ == "__main__":
    # Test the context manager
    from langchain_core.messages import HumanMessage, AIMessage
    
    print("Testing Context Manager...\n")
    
    # Create a test state with many messages
    test_state = {
        'question': "What's the weather?",
        'messages': [
            HumanMessage(content="What's the weather at the data center?"),
            AIMessage(content="Analyzing query for security threats..."),
            AIMessage(content="Query is SAFE"),
            AIMessage(content="Tool: ipify - Result: 203.0.113.45"),
            AIMessage(content="Tool: ip_to_geo - Result: 37.7749,-122.4194"),
            AIMessage(content="Tool: weather - Result: {'temp': 72, 'conditions': 'sunny'}"),
            HumanMessage(content="What's the temperature?"),
            AIMessage(content="The temperature is 72°F"),
        ] * 5,  # Duplicate messages to trigger compression
        'tool_results_cache': {},
        'context_token_count': 0
    }
    
    # Test compression
    manager = ContextManager(
        max_tokens=28000, 
        max_messages=10,
        summarize_threshold=500  # Lower threshold for testing
    )
    
    print(f"Before compression:")
    print(f"  Messages: {len(test_state['messages'])}")
    print(f"  Tokens: {manager.count_state_tokens(test_state)}")
    
    compressed_state = manager.compress_if_needed(test_state)
    
    print(f"\nAfter compression:")
    print(f"  Messages: {len(compressed_state['messages'])}")
    print(f"  Tokens: {compressed_state['context_token_count']}")
    summary = compressed_state.get('conversation_summary') or 'None'
    print(f"  Summary: {summary[:100] if isinstance(summary, str) else summary}")
    
    print(f"\nStats:")
    stats = manager.get_stats()
    print(f"  Compressions: {stats['compression_count']}")
    print(f"  Tokens saved: {stats['total_tokens_saved']}")
    
    print("\n✓ Context Manager test complete!")
