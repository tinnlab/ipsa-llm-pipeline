"""
Report bug 7 — upstream-regulator (TF) candidate list.

Driver TFs of a down-regulated, enzyme-centric program (e.g. the hepatocyte metabolic
masters HNF4A / PPARA / CEBPA) are usually NOT differentially expressed AND are NOT
reachable through KEGG GErel edges in the enriched metabolic maps, so the old GErel scan
surfaced only an off-program TF (TP53). The fix tests which TF's target set (CollecTRI
regulon) is over-represented among the down-regulated DE genes, with an LLM-proposed
fallback when the regulon DB is unavailable. Exercises the step-3 helper (in-memory
regulon, no network/LLM) plus the step-6 rendering.
"""

import json
from types import SimpleNamespace

import pytest

from src.pipeline.services.regulon_service import RegulonService
from src.pipeline.steps.step03_pathway_mechanisms import Step03PathwayMechanisms
from src.pipeline.steps.step06_report_generation import Step06ReportGeneration


def _bare_step03():
    return Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)


# HNF4A / PPARA are NOT in the DE gene list — they are the "invisible" drivers.
GENES = [
    {'geneSymbol': 'ACADS', 'foldChange': -1.4},
    {'geneSymbol': 'ALDH2', 'foldChange': -1.5},
    {'geneSymbol': 'CPT1A', 'foldChange': -2.1},
    {'geneSymbol': 'APOB', 'foldChange': -1.2},
    {'geneSymbol': 'CDK1', 'foldChange': +2.4},   # up-regulated proliferation gene
]

# Regulon: HNF4A activates the 4 down metabolic genes; DECOY regulates only up genes.
REGULON_EDGES = [
    ('HNF4A', 'ACADS', 1), ('HNF4A', 'ALDH2', 1),
    ('HNF4A', 'CPT1A', 1), ('HNF4A', 'APOB', 1),
    ('HNF4A', 'FILLER1', 1), ('HNF4A', 'FILLER2', 1),
    ('DECOY', 'CDK1', 1), ('DECOY', 'UP2', 1), ('DECOY', 'UP3', 1),
]

STRUCTURES = [
    {
        'pathway': 'Fatty acid degradation',
        'enrichment_score': -3.4,
        'enrichment_direction': 'downregulated',
        'p_value_fdr': 1.4e-2,
        'mapped_de_genes': [
            {'gene_symbol': 'ACADS'}, {'gene_symbol': 'CPT1A'}, {'gene_symbol': 'ALDH2'},
        ],
    },
    {
        'pathway': 'Cell cycle',
        'enrichment_score': +1.8,
        'enrichment_direction': 'upregulated',
        'p_value_fdr': 3.76e-4,
        'mapped_de_genes': [{'gene_symbol': 'CDK1'}],
    },
]


def _step_with_regulon():
    step = _bare_step03()
    step.regulon_service = RegulonService.from_edges(REGULON_EDGES)
    return step


def test_regulon_surfaces_non_de_master_tf():
    step = _step_with_regulon()
    regs = step._identify_upstream_regulators(STRUCTURES, GENES)
    by_tf = {r['tf']: r for r in regs}

    assert 'HNF4A' in by_tf, 'the master TF of the down axis was not recovered'
    hnf4a = by_tf['HNF4A']
    assert hnf4a['is_de'] is False                     # not in DE list, yet surfaced
    assert hnf4a['target_direction'] == 'down'
    assert hnf4a['overlap_count'] == 4
    assert set(hnf4a['targets']) == {'ACADS', 'ALDH2', 'CPT1A', 'APOB'}
    assert hnf4a['inferred_tf_activity'] == 'decreased'
    assert hnf4a['evidence_source'] == 'collectri'
    assert hnf4a['mode'] == 'database'
    assert hnf4a['enrichment_fdr'] is not None
    # Evidence pathways: the enriched DOWN pathway containing its targets.
    assert 'Fatty acid degradation' in hnf4a['evidence_pathways']


def test_down_pathways_for_targets_filters_up_and_weak():
    step = _bare_step03()
    structs = [
        {'pathway': 'StrongDown', 'enrichment_score': -3.0, 'p_value_fdr': 1e-3,
         'mapped_de_genes': [{'gene_symbol': 'ACADS'}]},
        {'pathway': 'WeakDown', 'enrichment_score': -1.2, 'p_value_fdr': 0.4,   # fdr > 0.25
         'mapped_de_genes': [{'gene_symbol': 'ACADS'}]},
        {'pathway': 'UpPathway', 'enrichment_score': +2.0, 'p_value_fdr': 1e-4,  # up-regulated
         'mapped_de_genes': [{'gene_symbol': 'ACADS'}]},
    ]
    got = step._find_down_pathways_for_targets(['ACADS'], structs)
    assert got == ['StrongDown']       # weak-FDR and up-regulated pathways excluded


def test_down_pathways_caps_dedups_and_uses_direction_fallback():
    step = _bare_step03()
    # 4 down pathways contain the target (one via enrichment_direction, NES absent);
    # expect the strongest 3 by |NES|, deduped, capped at max_pathways=3.
    structs = [
        {'pathway': 'D1', 'enrichment_score': -3.5, 'p_value_fdr': 1e-3,
         'mapped_de_genes': [{'gene_symbol': 'ACADS'}]},
        {'pathway': 'D2', 'enrichment_score': -2.0, 'p_value_fdr': 1e-3,
         'mapped_de_genes': [{'gene_symbol': 'ACADS'}]},
        {'pathway': 'D1', 'enrichment_score': -3.5, 'p_value_fdr': 1e-3,   # duplicate name
         'mapped_de_genes': [{'gene_symbol': 'ACADS'}]},
        {'pathway': 'D3', 'enrichment_score': -1.0, 'p_value_fdr': 1e-3,
         'mapped_de_genes': [{'gene_symbol': 'ACADS'}]},
        {'pathway': 'D4_dirfallback', 'enrichment_score': None,            # NES absent
         'enrichment_direction': 'downregulated', 'p_value_fdr': 1e-3,
         'mapped_de_genes': [{'gene_symbol': 'ACADS'}]},
    ]
    got = step._find_down_pathways_for_targets(['ACADS'], structs)
    assert len(got) == 3               # capped
    assert got == ['D1', 'D2', 'D3']   # strongest |NES| first, deduped; D4 (|NES|=0) drops off
    # the direction-fallback pathway IS eligible (proven by loosening the cap)
    assert 'D4_dirfallback' in step._find_down_pathways_for_targets(
        ['ACADS'], structs, max_pathways=10)


def test_off_program_tf_is_dropped():
    # DECOY regulates only up-regulated genes -> zero overlap with the down set -> absent.
    step = _step_with_regulon()
    regs = step._identify_upstream_regulators(STRUCTURES, GENES)
    assert 'DECOY' not in {r['tf'] for r in regs}


def test_no_down_genes_returns_empty():
    step = _step_with_regulon()
    up_only = [{'geneSymbol': 'CDK1', 'foldChange': +2.4}]
    assert step._identify_upstream_regulators(STRUCTURES, up_only) == []


def test_llm_fallback_when_regulon_unavailable():
    step = _bare_step03()
    step.regulon_service = RegulonService(path='/nonexistent/regulon.tsv')
    assert step.regulon_service.available is False

    def fake_complete(prompt, system_prompt=None, **kwargs):
        return json.dumps({'regulators': [
            {'tf': 'HNF4A', 'rationale': 'liver master regulator',
             'inferred_activity': 'decreased',
             'example_targets': ['ACADS', 'CPT1A', 'MADEUP']},   # MADEUP not in down set
        ]})

    step.llm = SimpleNamespace(complete=fake_complete)
    regs = step._identify_upstream_regulators(STRUCTURES, GENES)
    assert len(regs) == 1
    r = regs[0]
    assert r['tf'] == 'HNF4A'
    assert r['evidence_source'] == 'llm_hypothesis'
    assert r['mode'] == 'llm'
    assert r['enrichment_fdr'] is None
    # example_targets are filtered to genes actually in the down set.
    assert set(r['targets']) == {'ACADS', 'CPT1A'}


@pytest.mark.parametrize('payload', [
    '{"regulators": ["HNF4A", "PPARA"]}',      # list of bare strings, not dicts
    '{"regulators": {"tf": "HNF4A"}}',          # object instead of a list
    '{"regulators": [null, 3, "x"]}',           # null / wrong-typed entries
    '{"regulators": null}',                     # null
    '{"regulators": [{"tf": 123}]}',            # dict row with non-string tf
    'not json at all',                          # unparseable
])
def test_llm_fallback_survives_malformed_output(payload):
    # A malformed fallback response must degrade to [] — never crash Step 3 (the caller
    # re-raises, so a crash here would abort the whole pathway-mechanisms step).
    step = _bare_step03()
    step.regulon_service = RegulonService(path='/nonexistent/regulon.tsv')
    step.llm = SimpleNamespace(complete=lambda *a, **k: payload)
    assert step._identify_upstream_regulators(STRUCTURES, GENES) == []


def test_llm_fallback_tolerates_wrongtyped_dict_fields():
    # A valid tf but a non-list example_targets must not crash — it yields a row with no
    # example targets rather than a TypeError that aborts Step 3.
    step = _bare_step03()
    step.regulon_service = RegulonService(path='/nonexistent/regulon.tsv')
    step.llm = SimpleNamespace(complete=lambda *a, **k: json.dumps(
        {'regulators': [{'tf': 'HNF4A', 'inferred_activity': 'decreased',
                         'example_targets': 5}]}))       # non-iterable example_targets
    regs = step._identify_upstream_regulators(STRUCTURES, GENES)
    assert len(regs) == 1
    assert regs[0]['tf'] == 'HNF4A'
    assert regs[0]['targets'] == []


def test_step06_renders_regulon_columns():
    step6 = Step06ReportGeneration.__new__(Step06ReportGeneration)
    regs = _step_with_regulon()._identify_upstream_regulators(STRUCTURES, GENES)
    block = step6._format_upstream_regulators(regs)
    assert 'Upstream Regulator Candidates' in block
    assert 'HNF4A' in block
    assert 'not DE' in block
    assert 'Inferred TF Activity' in block
    assert 'Enrichment FDR' in block
    assert 'CollecTRI' in block
    # empty input -> no section
    assert step6._format_upstream_regulators([]) == ''


def test_step06_renders_truncation_fdr_and_dash():
    step6 = Step06ReportGeneration.__new__(Step06ReportGeneration)
    rows = [
        {  # DB row with >6 targets -> truncation + (overlap/regulon_size) + numeric FDR cell
            'tf': 'HNF4A', 'is_de': False, 'inferred_tf_activity': 'decreased',
            'targets': ['A', 'B', 'C', 'D', 'E', 'F', 'G'], 'overlap_count': 7,
            'regulon_size': 280, 'enrichment_fdr': 1.14e-11, 'evidence_pathways': ['P1'],
            'evidence_source': 'collectri', 'mode': 'database',
        },
        {  # LLM row -> '(overlap)' only and an em-dash FDR cell
            'tf': 'PPARA', 'is_de': False, 'inferred_tf_activity': 'decreased',
            'targets': ['A', 'B'], 'overlap_count': 2, 'regulon_size': None,
            'enrichment_fdr': None, 'evidence_pathways': [],
            'evidence_source': 'llm_hypothesis', 'mode': 'llm', 'fallback_reason': 'db_unavailable',
        },
    ]
    block = step6._format_upstream_regulators(rows)
    assert ', …' in block                         # >6 targets truncated
    assert '(7/280)' in block                      # overlap/regulon_size suffix
    assert '1.14e-11' in block                     # numeric FDR cell (DB row)
    assert '(2)' in block                          # LLM row: overlap only, no regulon_size
    assert '| — |' in block                        # em-dash for the None FDR (LLM row)


def _llm_row(reason):
    return {
        'tf': 'HNF4A', 'is_de': False, 'inferred_tf_activity': 'decreased',
        'targets': ['ACADS', 'CPT1A'], 'overlap_count': 2, 'regulon_size': None,
        'enrichment_fdr': None, 'evidence_pathways': [],
        'evidence_source': 'llm_hypothesis', 'mode': 'llm', 'fallback_reason': reason,
    }


def test_step06_flags_llm_hypothesis_rows():
    step6 = Step06ReportGeneration.__new__(Step06ReportGeneration)
    block = step6._format_upstream_regulators([_llm_row('db_unavailable')])
    assert 'LLM-proposed hypotheses' in block
    assert 'LLM hypothesis' in block


def test_step06_fallback_note_matches_reason():
    # The disclaimer must NOT falsely claim the DB was unavailable when it was loaded but
    # produced no significant hit.
    step6 = Step06ReportGeneration.__new__(Step06ReportGeneration)
    unavail = step6._format_upstream_regulators([_llm_row('db_unavailable')])
    assert 'regulon database was unavailable' in unavail

    no_sig = step6._format_upstream_regulators([_llm_row('no_significant_enrichment')])
    assert 'no transcription factor reached regulon-enrichment significance' in no_sig
    assert 'unavailable' not in no_sig


def test_step03_no_significant_enrichment_uses_llm_with_reason():
    # Regulon available but the down set doesn't clear the gate -> LLM fallback tagged
    # 'no_significant_enrichment' (not 'db_unavailable').
    step = _bare_step03()
    step.regulon_service = RegulonService.from_edges([('IRRELEVANT', 'ZZZ1', 1)])
    assert step.regulon_service.available is True

    step.llm = SimpleNamespace(complete=lambda *a, **k: json.dumps(
        {'regulators': [{'tf': 'HNF4A', 'inferred_activity': 'decreased',
                         'example_targets': ['ACADS']}]}))
    regs = step._identify_upstream_regulators(STRUCTURES, GENES)
    assert regs and regs[0]['fallback_reason'] == 'no_significant_enrichment'
