"""
Tests for report data hygiene:
- item 7:  zero/absent fold-change genes excluded (no direction arrow)
- item 11: confidence badge reflects evidence (no default "High Confidence")
- item 12: cluster significance driven by statistical strength
- item 17: enrichment score labelled by real metric (NES preferred) + calibrated magnitude
"""

from unittest.mock import MagicMock

from src.pipeline.fc_utils import fc_direction, fc_arrow, has_fc
from src.pipeline.steps.step03_pathway_mechanisms import (
    Step03PathwayMechanisms,
    _enrichment_metric,
    _magnitude_label,
)
from src.pipeline.services.pathway_clustering_service import PathwayClusteringService


def _bare_step03():
    step = Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)
    step.step_number = 3
    step.step_name = 'Pathway Mechanisms and Interactions'
    step.llm = MagicMock()
    return step


# ---- fold-change direction (item 7 / 10) ----------------------------------

def test_fc_direction_treats_zero_as_neutral():
    assert fc_direction(0) == 'neutral'
    assert fc_direction(0.0) == 'neutral'
    assert fc_direction(None) == 'neutral'
    assert fc_direction(1.2) == 'up'
    assert fc_direction(-1.2) == 'down'


def test_fc_arrow_neutral_has_no_arrow():
    assert fc_arrow(0) == ''
    assert fc_arrow(2) == '↑'
    assert fc_arrow(-2) == '↓'


def test_zero_fc_gene_excluded_from_report():
    step = _bare_step03()
    result = {'pathwayMechanisms': [{
        'pathway': 'MicroRNAs in cancer',
        'biologicalFunction': 'miRNA regulation',
        'deGeneInvolvement': [
            {'gene': 'MIR21', 'foldChange': 0.0, 'roleInPathway': 'implicated'},
            {'gene': 'MIR25', 'foldChange': 3.37, 'roleInPathway': 'oncomir'},
        ],
    }]}
    section = step._generate_report_section(result, pathway_structures=[])
    assert 'MIR25' in section
    assert 'MIR21' not in section


# ---- enrichment metric (item 17) ------------------------------------------

def test_enrichment_metric_prefers_nes():
    assert _enrichment_metric({'ES': 0.5, 'NES': -3.5}) == (-3.5, 'NES')
    assert _enrichment_metric({'ES': 0.5}) == (0.5, 'ES')
    assert _enrichment_metric({'NES': 2.1}) == (2.1, 'NES')


def test_magnitude_label_calibrated_by_metric():
    # NES scale: +1.30 is weak, -4.5 is strong (previously both "strong")
    assert _magnitude_label(1.30, 'NES') == 'weak'
    assert _magnitude_label(-4.5, 'NES') == 'strong'
    assert _magnitude_label(1.7, 'NES') == 'moderate'
    # ES scale unchanged
    assert _magnitude_label(0.7, 'ES') == 'strong'
    assert _magnitude_label(0.3, 'ES') == 'weak'


def test_report_labels_nes_and_calibrates():
    step = _bare_step03()
    result = {'pathwayMechanisms': [{'pathway': 'Thermogenesis', 'biologicalFunction': 'heat'}]}
    structures = [{
        'pathway': 'Thermogenesis', 'source': 'kegg', 'confidence': 'high',
        'enrichment_score': 1.297, 'enrichment_metric': 'NES',
        'enrichment_direction': 'upregulated',
    }]
    section = step._generate_report_section(result, pathway_structures=structures)
    assert 'NES = +1.297' in section
    assert 'weak signal' in section
    # metric is labelled NES, not the misleading bare "ES ="
    assert ' ES = ' not in section


# ---- confidence badge (item 11) -------------------------------------------

def test_confidence_not_defaulted_to_high_when_unmatched():
    step = _bare_step03()
    result = {'pathwayMechanisms': [{'pathway': 'Some pathway', 'biologicalFunction': 'x'}]}
    # No matching structure -> confidence unknown, must NOT say High Confidence
    section = step._generate_report_section(result, pathway_structures=[])
    assert 'High Confidence' not in section
    assert 'Unverified' in section


def test_confidence_reflects_gene_set():
    step = _bare_step03()
    result = {'pathwayMechanisms': [{'pathway': 'P', 'biologicalFunction': 'x'}]}
    structures = [{'pathway': 'P', 'source': 'kegg', 'confidence': 'gene_set'}]
    section = step._generate_report_section(result, pathway_structures=structures)
    assert 'Gene Set Only' in section
    assert 'High Confidence' not in section


# ---- cluster significance (item 12) ---------------------------------------

def test_headline_cluster_is_high_significance():
    svc = PathwayClusteringService()
    # avg FDR ~3.76e-4 with only 2 pathways used to be MEDIUM; now HIGH
    assert svc._classify_cluster_significance(3.76e-4, pathway_count=2) == 'high'
    assert svc._classify_cluster_significance(0.02, pathway_count=2) == 'medium'
    assert svc._classify_cluster_significance(0.2, pathway_count=10) == 'low'


# ---- audit regressions ----------------------------------------------------

def test_nes_in_es_field_detected_by_magnitude():
    """F5: a |value|>1 in the ES field can only be NES → treated/labelled as NES."""
    assert _enrichment_metric({'ES': -3.5}) == (-3.5, 'NES')   # out of [-1,1] → NES
    assert _enrichment_metric({'ES': 0.4}) == (0.4, 'ES')      # within [-1,1] → ES
    assert _enrichment_metric({'NES': 0.4}) == (0.4, 'NES')    # explicit NES wins


def test_report_relabels_es_field_holding_nes():
    """F5 end-to-end: legacy payload (NES packed in ES) renders as NES, calibrated."""
    step = _bare_step03()
    result = {'pathwayMechanisms': [{'pathway': 'Fatty acid degradation', 'biologicalFunction': 'x'}]}
    structures = [{'pathway': 'Fatty acid degradation', 'source': 'kegg', 'confidence': 'high',
                   'enrichment_score': -3.40, 'enrichment_direction': 'downregulated'}]
    # note: enrichment_metric intentionally absent (legacy) → derived from magnitude
    section = step._generate_report_section(result, pathway_structures=structures)
    assert 'NES = -3.400' in section
    assert 'strong signal' in section
    assert ' ES = ' not in section


def test_string_foldchange_does_not_crash():
    """F3: a numeric-string foldChange from the LLM must not crash dedup/render."""
    step = _bare_step03()
    result = {'pathwayMechanisms': [{
        'pathway': 'P', 'biologicalFunction': 'x',
        'deGeneInvolvement': [
            {'gene': 'AAA', 'foldChange': '2.5', 'roleInPathway': 'r'},
            {'gene': 'AAA', 'foldChange': '2.5', 'roleInPathway': 'r'},  # dup → dedup path
            {'gene': 'BBB', 'foldChange': 'not-a-number'},
        ],
    }]}
    section = step._generate_report_section(result, pathway_structures=[])
    assert 'AAA' in section
    assert 'BBB' in section  # unparseable FC kept, rendered without a value


def test_zero_enrichment_renders_neutral_direction():
    """F5: an enrichment score of exactly 0 is neither up nor down (→, not ↓)."""
    step = _bare_step03()
    result = {'pathwayMechanisms': [{'pathway': 'P', 'biologicalFunction': 'x'}]}
    structures = [{'pathway': 'P', 'source': 'kegg', 'confidence': 'high',
                   'enrichment_score': 0.0, 'enrichment_metric': 'NES',
                   'enrichment_direction': 'unchanged'}]
    section = step._generate_report_section(result, pathway_structures=structures)
    assert '→' in section
    assert '↓' not in section.split('Pathway Enrichment')[1].split('\n')[0]


def test_none_foldchange_kept_without_arrow():
    """F12: a gene with missing FC is kept (no arrow), only explicit-zero is dropped."""
    step = _bare_step03()
    result = {'pathwayMechanisms': [{
        'pathway': 'P', 'biologicalFunction': 'x',
        'deGeneInvolvement': [
            {'gene': 'KEEPME', 'foldChange': None, 'roleInPathway': 'role'},
            {'gene': 'DROPME', 'foldChange': 0.0, 'roleInPathway': 'artifact'},
            {'gene': 'REAL', 'foldChange': 2.1},
        ],
    }]}
    section = step._generate_report_section(result, pathway_structures=[])
    assert 'KEEPME' in section and 'DROPME' not in section and 'REAL' in section
    # no dangling "( )" and no bare arrow with no number
    assert '( )' not in section
    assert '↑ )' not in section and '(FC:  ' not in section
