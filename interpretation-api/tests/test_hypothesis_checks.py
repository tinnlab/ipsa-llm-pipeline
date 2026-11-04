"""
Items 12, 13 and 25: deterministic hypothesis checks (no LLM).
"""

from src.pipeline.hypothesis_checks import (
    strict_direction_errors, loose_direction_matches, correct_arrow_fc,
    hypothesis_direction_errors, relation_reference, relation_claim_errors,
    differs_at_printed_precision, correct_pathway_stats,
)

FC = {'SLC27A5': -1.31, 'CDK1': 2.41, 'REST': 0.8, 'HRAS': 1.22, 'Ndufs1': -0.9}


# ------------------------------------------------------------------ item 12 detection

def test_report3_slc27a5_case_with_u2011_hyphen():
    text = 'Loss of FXR signaling permits up‑regulation of SLC27A5 (↓1.31).'
    errs = strict_direction_errors(text, FC)
    assert [(e['gene'], e['claimed']) for e in errs] == [('SLC27A5', 'up')]


def test_fixed_shapes():
    for text in ('up-regulated SLC27A5 drives', 'SLC27A5 is up-regulated', 'SLC27A5 (↓1.31) is markedly elevated',
                 'SLC27A5 up-regulation', 'increased SLC27A5 expression'):
        assert strict_direction_errors(text, FC), text


def test_correct_direction_and_far_word_are_not_errors():
    assert strict_direction_errors('CDK1 is up-regulated', FC) == []
    # "increased" is far from SLC27A5 and refers to something else
    assert strict_direction_errors('increased flux through the pathway, while SLC27A5 falls', FC) == []


def test_gene_match_is_case_sensitive():
    # "rest" (English) must not be read as the gene REST.
    assert strict_direction_errors('reduced rest of the cohort', FC) == []
    assert strict_direction_errors('reduced REST levels', FC) != []


def test_mouse_cased_symbol():
    assert strict_direction_errors('elevated Ndufs1 drives', FC)[0]['gene'] == 'Ndufs1'


def test_loose_pattern_is_wider():
    text = 'Increased lipid accumulation and the fatty acid transporter SLC27A5'
    assert strict_direction_errors(text, FC) == []
    assert loose_direction_matches(text, FC)  # recorded as a warning only


def test_arrow_magnitude_corrected_and_wrong_arrow_is_conflict():
    text, conflicts = correct_arrow_fc('HRAS (↑1.30) and CDK1 ↓2.41', FC)
    assert text == 'HRAS (↑1.22) and CDK1 ↓2.41'
    assert conflicts == {'CDK1'}


def test_key_players_are_read():
    hyp = {'hypothesis': '', 'mechanisticModel': '', 'keyPlayers': ['CDK1 (↓2.41) – kinase']}
    assert [(e['gene'], e['claimed']) for e in hypothesis_direction_errors(hyp, FC)] == [('CDK1', 'down')]


def test_directional_prediction_is_not_read():
    hyp = {'hypothesis': '', 'mechanisticModel': '', 'directionalPrediction': 'reduced CDK1 expression'}
    assert hypothesis_direction_errors(hyp, FC) == []


# ------------------------------------------------------------------ item 13

def test_relation_claims_against_returned_relations():
    ref = relation_reference([{'source': 'E2F1', 'target': 'MCM2', 'type': 'expression'},
                              {'source': 'CDK1', 'target': 'CCNB1', 'type': 'binding/association'}])
    hyp = {'mechanisticClaims': [
        {'source': 'E2F1', 'target': 'MCM2', 'relation': 'expression'},       # returned
        {'source': 'MCM2', 'target': 'E2F1', 'relation': 'expression'},       # reversed directed
        {'source': 'CCNB1', 'target': 'CDK1', 'relation': 'binding'},         # undirected either way
        {'source': 'AMDHD1', 'target': 'histidine catabolism flux', 'relation': 'activation'},
    ]}
    errs = relation_claim_errors(hyp, ref)
    assert [(e['source'], e['reason']) for e in errs] == [
        ('MCM2', 'not returned by the tool calls'), ('AMDHD1', 'not gene-to-gene')]


def test_empty_reference_fails_every_claim():
    hyp = {'mechanisticClaims': [{'source': 'E2F1', 'target': 'MCM2'}]}
    assert len(relation_claim_errors(hyp, relation_reference([]))) == 1


# ------------------------------------------------------------------ item 25

def test_printed_precision():
    assert not differs_at_printed_precision('2.9e-12', 2.85e-12)
    assert differs_at_printed_precision('3.1e-12', 2.85e-12)
    assert differs_at_printed_precision('2.86e-12', 2.85e-12)   # was within 10x: now replaced
    assert not differs_at_printed_precision('0.0012', 0.00121)


def test_pathway_nes_and_fdr_next_to_name():
    stats = {'Histidine metabolism': {'nes': -2.50, 'fdr': 1.07e-3},
             'Ribosome': {'nes': 1.81, 'fdr': 2.24e-3},
             'Ribosome biogenesis in eukaryotes': {'nes': 1.60, 'fdr': 2.24e-3}}
    text = ('Histidine metabolism (NES = -2.31, FDR = 1.1e-3) is lost; '
            'Ribosome biogenesis in eukaryotes (NES 1.81) rises.')
    new, n = correct_pathway_stats(text, stats)
    assert new == ('Histidine metabolism (NES = -2.50, FDR = 1.1e-3) is lost; '
                   'Ribosome biogenesis in eukaryotes (NES 1.60) rises.')
    assert n == 2
