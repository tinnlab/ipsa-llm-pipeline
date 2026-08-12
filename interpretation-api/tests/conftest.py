"""Shared test fixtures. All tests mock the LLM — no model, GPU, or network needed."""
import os
import sys

import pytest

# Make `src` (and the steps' sys.path tricks) importable when running pytest from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def allow_pathways():
    return {"Glycolysis", "TCA cycle", "Apoptosis"}


@pytest.fixture
def allow_genes():
    return {"TP53", "BRCA1", "EGFR"}


@pytest.fixture
def input_data():
    """Minimal but realistic pipeline input — the ground-truth allow-list."""
    return {
        "genes": [
            {"geneSymbol": "TP53", "log2_fold_change": 2.0},
            {"geneSymbol": "BRCA1", "log2_fold_change": -1.5},
            {"geneSymbol": "EGFR", "log2_fold_change": 1.2},
        ],
        "pathways": [
            {"name": "Glycolysis"},
            {"name": "TCA cycle"},
            {"name": "Apoptosis"},
        ],
        "metadata": {"organism": "Homo sapiens", "disease": "test"},
    }
