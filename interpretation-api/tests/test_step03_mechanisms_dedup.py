"""
Tests for Step 3 mechanism de-duplication and empty-section suppression.

Covers report items 1/8/9: duplicated pathway sections, empty backfilled sections,
and duplicated DE-gene entries within a pathway.
"""

from src.pipeline.steps.step03_pathway_mechanisms import (
    _dedupe_mechanisms,
    _dedupe_de_genes,
    _mechanism_richness,
)


def test_dedupe_mechanisms_keeps_richest():
    rich = {'pathway': 'Cell cycle', 'biologicalFunction': 'Coordinates division',
            'deGeneInvolvement': [{'gene': 'CDK1'}], 'functionalConsequences': 'x'}
    stub = {'pathway': 'Cell cycle', 'biologicalFunction': '',
            'deGeneInvolvement': [], 'functionalConsequences': ''}
    out = _dedupe_mechanisms([rich, stub])
    assert len(out) == 1
    assert out[0]['biologicalFunction'] == 'Coordinates division'


def test_dedupe_mechanisms_prefers_rich_even_if_stub_first():
    stub = {'pathway': 'DNA replication', 'biologicalFunction': ''}
    rich = {'pathway': 'dna replication', 'biologicalFunction': 'Duplicates genome'}
    out = _dedupe_mechanisms([stub, rich])
    assert len(out) == 1
    assert out[0]['biologicalFunction'] == 'Duplicates genome'


def test_dedupe_mechanisms_preserves_distinct_order():
    out = _dedupe_mechanisms([
        {'pathway': 'A', 'biologicalFunction': 'a'},
        {'pathway': 'B', 'biologicalFunction': 'b'},
    ])
    assert [m['pathway'] for m in out] == ['A', 'B']


def test_dedupe_de_genes_by_symbol_keeps_strongest_fc():
    genes = [
        {'gene': 'CYP1A1', 'foldChange': 1.48},
        {'gene': 'CYP1A1', 'foldChange': 1.48},
        {'gene': 'cyp1a1', 'foldChange': -2.0},  # case-insensitive, stronger |FC|
        {'gene': 'NAT2', 'foldChange': -2.47},
    ]
    out = _dedupe_de_genes(genes)
    symbols = [g['gene'] for g in out]
    assert symbols.count('CYP1A1') + symbols.count('cyp1a1') == 1
    assert 'NAT2' in symbols
    cyp = next(g for g in out if g['gene'].upper() == 'CYP1A1')
    assert cyp['foldChange'] == -2.0


def test_richness_scores_rich_over_stub():
    rich = {'biologicalFunction': 'x', 'functionalConsequences': 'y',
            'deGeneInvolvement': [1], 'curatedRelations': [1]}
    stub = {'biologicalFunction': '', 'deGeneInvolvement': []}
    assert _mechanism_richness(rich) > _mechanism_richness(stub)


def test_empty_mechanism_not_rendered():
    """A fully-empty mechanism must not appear as a blank section in the report."""
    from unittest.mock import MagicMock
    from src.pipeline.steps.step03_pathway_mechanisms import Step03PathwayMechanisms

    step = Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)
    step.step_number = 3
    step.step_name = 'Pathway Mechanisms and Interactions'
    step.llm = MagicMock()

    result = {'pathwayMechanisms': [
        {'pathway': 'Real pathway', 'biologicalFunction': 'Does things',
         'deGeneInvolvement': [{'gene': 'CDK1', 'foldChange': 2.4, 'roleInPathway': 'kinase'}]},
        {'pathway': 'Empty pathway', 'biologicalFunction': '', 'deGeneInvolvement': [],
         'curatedRelations': [], 'crosstalk': [], 'functionalConsequences': ''},
    ]}
    section = step._generate_report_section(result, pathway_structures=[])
    assert 'Real pathway' in section
    assert 'Empty pathway' not in section
