"""
Pipeline module for biological data analysis.

This module provides a multi-step analysis workflow for processing
gene expression and pathway enrichment data to generate therapeutic insights.
"""

from .orchestrator import PipelineOrchestrator

__all__ = ['PipelineOrchestrator']
