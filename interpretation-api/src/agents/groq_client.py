"""
Legacy GroqClient module for backward compatibility.
Now redirects to UnifiedLLMClient.

Original implementation moved to llm_client.py
This file kept for backward compatibility with existing imports.
"""
from .llm_client import GroqClient

# For backward compatibility, re-export the main class
__all__ = ['GroqClient']
