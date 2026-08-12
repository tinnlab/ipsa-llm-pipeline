"""
Tests for Part C: when Step 3 retrieved pathway structures but its mechanism-
interpretation LLM step produced no usable output (unparseable JSON, or an empty
list), the report's "Pathway Mechanisms and Interactions" section must be populated
directly from the retrieved structures — NOT rendered blank, and NOT emit the
misleading "No pathway structures could be retrieved from KEGG" message.
"""

import types

from unittest.mock import MagicMock

from src.pipeline.steps.step03_pathway_mechanisms import Step03PathwayMechanisms
from src.pipeline.steps.step04_hypothesis_generation import Step04HypothesisGeneration
from src.pipeline.services.pathway_query_service import PathwayQueryService
from src.pipeline.services.regulon_service import RegulonService


def _bare_step03():
    step = Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)
    step.step_number = 3
    step.step_name = 'Pathway Mechanisms and Interactions'
    step.llm = MagicMock()
    # execute() runs upstream-regulator inference; give it an empty (unavailable) regulon
    # so these Part-C mechanism tests don't depend on the bundled DB.
    step.regulon_service = RegulonService(path='/nonexistent/regulon.tsv')
    return step


def _structure(pathway='Butanoate metabolism'):
    """A retrieved structure in the internal (pre-serialization) shape execute() uses."""
    return {
        'pathway': pathway,
        'pathway_id': 'hsa00650',
        'source': 'kegg',
        'confidence': 'high',
        'structure': types.SimpleNamespace(id='hsa00650'),
        'mapped_de_genes': [
            types.SimpleNamespace(gene_symbol='ACAT1', fold_change=-1.2, p_value=0.01, direction='down'),
            types.SimpleNamespace(gene_symbol='DBT', fold_change=-1.06, p_value=0.02, direction='down'),
        ],
        'de_relations': [
            types.SimpleNamespace(source='ACAT1', target='DBT', type='ECrel', subtype='activation'),
        ],
        'p_value': 1e-3,
        'p_value_fdr': 1.75e-3,
        'enrichment_score': -1.9,
        'enrichment_direction': 'downregulated',
    }


# --------------------------------------------------------------------------- #
# Unit: the synthesis helpers
# --------------------------------------------------------------------------- #

def test_synthesize_mechanisms_from_structures():
    step = _bare_step03()
    mechs = step._synthesize_mechanisms_from_structures([_structure()])
    assert len(mechs) == 1
    m = mechs[0]
    assert m['pathway'] == 'Butanoate metabolism'
    assert {g['gene'] for g in m['deGeneInvolvement']} == {'ACAT1', 'DBT'}
    assert len(m['curatedRelations']) == 1
    rel = m['curatedRelations'][0]
    assert (rel['source'], rel['target'], rel['type']) == ('ACAT1', 'DBT', 'activation')


def test_synthesize_filters_relations_missing_endpoints():
    step = _bare_step03()
    struct = _structure()
    struct['de_relations'] = [
        types.SimpleNamespace(source='ACAT1', target='', type='ECrel', subtype='activation'),
    ]
    mechs = step._synthesize_mechanisms_from_structures([struct])
    assert mechs[0]['curatedRelations'] == []


def test_generate_summary_from_structures_is_factual():
    step = _bare_step03()
    summary = step._generate_summary_from_structures(step._synthesize_mechanisms_from_structures([_structure()]))
    assert 'Mechanism interpretation was unavailable' in summary
    assert 'Butanoate metabolism' in summary


# --------------------------------------------------------------------------- #
# Integration via execute(): both failure routes synthesize instead of blanking
# --------------------------------------------------------------------------- #

def _run_execute(step):
    return step.execute(
        pathways=[{'name': 'Butanoate metabolism', 'source': 'KEGG', 'genes': ['ACAT1', 'DBT']}],
        genes=[{'geneSymbol': 'ACAT1', 'foldChange': -1.2}],
        analyses=[],
        themes=None,
        hub_genes=[],
    )


def test_execute_synthesizes_on_unparseable_json():
    """Interpretation LLM returns unparseable JSON -> synthesize from structures."""
    step = _bare_step03()
    step._get_pathway_structures = MagicMock(return_value=[_structure()])
    step.llm.chat.return_value = 'this is not valid json {{{'

    result = _run_execute(step)

    assert len(result.pathway_mechanisms) == 1
    assert result.pathway_mechanisms[0]['pathway'] == 'Butanoate metabolism'
    assert result.pathway_mechanisms[0]['curatedRelations'][0]['source'] == 'ACAT1'
    # Report reflects real data, NOT the misleading empty message.
    assert 'Butanoate metabolism' in result.report_section
    assert 'ACAT1' in result.report_section
    assert 'No pathway structures could be retrieved' not in result.report_section
    assert 'Mechanism interpretation was unavailable' in result.mechanistic_summary


def test_execute_synthesizes_on_empty_mechanisms():
    """Interpretation LLM returns valid JSON with an empty list -> synthesize."""
    step = _bare_step03()
    step._get_pathway_structures = MagicMock(return_value=[_structure()])
    step.llm.chat.return_value = '{"pathwayMechanisms": [], "mechanisticSummary": ""}'

    result = _run_execute(step)

    assert len(result.pathway_mechanisms) == 1
    assert result.pathway_mechanisms[0]['pathway'] == 'Butanoate metabolism'
    assert 'No pathway structures could be retrieved' not in result.report_section


def test_get_pathway_structures_forwards_organism_to_kegg():
    """The organism must reach kegg_service.get_pathway_structure — completes the
    orchestrator -> execute -> _get_pathway_structures -> KEGG organism chain so a
    non-human dataset isn't queried against human KEGG."""
    step = _bare_step03()
    step.kegg_service = MagicMock()
    step.kegg_service.get_pathway_structure.return_value = None  # miss -> stop before mapping
    step.pc_service = MagicMock()
    step.pc_service.get_interactions_between.return_value = []
    step.llm.chat.return_value = (
        '{"biologicalFunction": "x", "inferredRelations": [], '
        '"deGeneRoles": [], "functionalConsequences": "y"}'
    )
    step._get_pathway_structures(
        [{'name': 'Butanoate metabolism', 'source': 'KEGG', 'genes': ['ACAT1']}],
        [],
        'Mus musculus',
    )
    step.kegg_service.get_pathway_structure.assert_called_once_with(
        'Butanoate metabolism', 'Mus musculus'
    )


def test_execute_extracts_organism_from_analyses():
    """The execute() seam: analyses[0]['organismId'] must reach _get_pathway_structures,
    so a non-human dataset isn't silently queried against human KEGG."""
    step = _bare_step03()
    captured = {}

    def fake_get_structs(pathways, genes, organism):
        captured['organism'] = organism
        return []

    step._get_pathway_structures = fake_get_structs
    step.execute(
        pathways=[{'name': 'P', 'source': 'KEGG', 'genes': []}],
        genes=[],
        analyses=[{'organismId': 'Mus musculus'}],
        themes=None,
        hub_genes=[],
    )
    assert captured['organism'] == 'Mus musculus'


def test_execute_still_empty_when_no_structures():
    """Genuinely no structures retrieved -> keep the explicit empty report (unchanged)."""
    step = _bare_step03()
    step._get_pathway_structures = MagicMock(return_value=[])

    result = _run_execute(step)

    assert result.pathway_mechanisms == []
    assert 'No pathway structures could be retrieved' in result.report_section


def test_execute_synthesizes_on_empty_mechanisms_full():
    """Strengthened: empty-list interpretation -> synthesized entries, honest summary,
    real content in the report, and no misleading message."""
    step = _bare_step03()
    step._get_pathway_structures = MagicMock(return_value=[_structure()])
    step.llm.chat.return_value = '{"pathwayMechanisms": [], "mechanisticSummary": ""}'

    result = _run_execute(step)

    assert len(result.pathway_mechanisms) == 1
    assert 'Mechanism interpretation was unavailable' in result.mechanistic_summary
    assert 'ACAT1' in result.report_section
    assert 'No pathway structures could be retrieved' not in result.report_section


def test_execute_backfills_missing_pathways_keeps_llm_summary():
    """Partial failure: some pathways interpreted, others dropped -> the dropped ones
    are backfilled from structures while the genuine LLM summary is preserved."""
    step = _bare_step03()
    step._get_pathway_structures = MagicMock(
        return_value=[_structure('Butanoate metabolism'), _structure('Propanoate metabolism')]
    )
    # LLM returns a mechanism for only ONE of the two structures, plus a real summary.
    step.llm.chat.return_value = (
        '{"pathwayMechanisms": [{"pathway": "Butanoate metabolism", '
        '"biologicalFunction": "real bio", "deGeneInvolvement": [], '
        '"curatedRelations": [], "crosstalk": [], "functionalConsequences": ""}], '
        '"mechanisticSummary": "A genuine interpretation summary."}'
    )

    result = _run_execute(step)

    pathways = {m['pathway'] for m in result.pathway_mechanisms}
    assert pathways == {'Butanoate metabolism', 'Propanoate metabolism'}  # missing one backfilled
    assert result.mechanistic_summary == 'A genuine interpretation summary.'  # LLM summary kept


def test_execute_backfill_matches_pathway_name_case_insensitively():
    """An LLM-echoed pathway name differing only in case/whitespace must NOT be treated
    as missing (which would append a duplicate structure-derived card)."""
    step = _bare_step03()
    step._get_pathway_structures = MagicMock(return_value=[_structure('Butanoate metabolism')])
    step.llm.chat.return_value = (
        '{"pathwayMechanisms": [{"pathway": "  butanoate METABOLISM ", '
        '"biologicalFunction": "real", "deGeneInvolvement": [], '
        '"curatedRelations": [], "crosstalk": [], "functionalConsequences": ""}], '
        '"mechanisticSummary": "s"}'
    )
    result = _run_execute(step)
    assert len(result.pathway_mechanisms) == 1  # no duplicate backfilled


def test_execute_synthesizes_on_batch_all_fail():
    """Batch path (>15 pathways) where every batch fails JSON parsing -> all pathways
    are backfilled and the misleading 'processing errors' summary is replaced."""
    step = _bare_step03()
    structures = [_structure(f'Pathway {i}') for i in range(16)]  # > BATCH_THRESHOLD (15)
    step._get_pathway_structures = MagicMock(return_value=structures)
    step.llm.chat.return_value = 'not valid json {{'  # every batch fails to parse

    result = _run_execute(step)

    assert len(result.pathway_mechanisms) == 16
    assert 'Mechanism interpretation was unavailable' in result.mechanistic_summary
    assert 'processing errors' not in result.mechanistic_summary
    assert 'No pathway structures could be retrieved' not in result.report_section


# --------------------------------------------------------------------------- #
# Synthesizer detail: real-DE filtering, isDE, relation type mapping, summary branch
# --------------------------------------------------------------------------- #

def _structure_geneset(pathway='Gene Set Pathway'):
    """A structure that resolved but mapped only gene-set placeholders (no real DE)."""
    s = _structure(pathway)
    s['confidence'] = 'gene_set'
    s['mapped_de_genes'] = [
        types.SimpleNamespace(gene_symbol='GENEX', fold_change=0.0, p_value=1.0, direction='unknown'),
        types.SimpleNamespace(gene_symbol='GENEY', fold_change=0.0, p_value=1.0, direction='unknown'),
    ]
    s['de_relations'] = []
    return s


def test_synthesize_excludes_placeholder_genes():
    """Gene-set placeholder genes (direction 'unknown') must NOT appear as DE genes —
    this is what otherwise trips the Step 3 hallucination validator."""
    step = _bare_step03()
    mechs = step._synthesize_mechanisms_from_structures([_structure_geneset()])
    assert mechs[0]['deGeneInvolvement'] == []


def test_synthesize_filters_empty_gene_symbol():
    step = _bare_step03()
    struct = _structure()
    struct['mapped_de_genes'] = [
        types.SimpleNamespace(gene_symbol='', fold_change=-1.0, p_value=0.01, direction='down'),
        types.SimpleNamespace(gene_symbol='ACAT1', fold_change=-1.2, p_value=0.01, direction='down'),
    ]
    mechs = step._synthesize_mechanisms_from_structures([struct])
    assert {g['gene'] for g in mechs[0]['deGeneInvolvement']} == {'ACAT1'}


def test_synthesize_isDE_reflects_membership():
    """isDE is True only when BOTH relation endpoints are real DE genes."""
    step = _bare_step03()
    struct = _structure()
    struct['mapped_de_genes'] = [
        types.SimpleNamespace(gene_symbol='ACAT1', fold_change=-1.2, p_value=0.01, direction='down'),
    ]
    struct['de_relations'] = [
        types.SimpleNamespace(source='ACAT1', target='DBT', type='ECrel', subtype='activation'),
    ]
    rel = step._synthesize_mechanisms_from_structures([struct])[0]['curatedRelations'][0]
    assert rel['isDE'] is False  # DBT is not a mapped DE gene here


def test_synthesize_mockrelation_type_mapping():
    """LLM 'other'-route relations (type='inferred', subtype='regulation') map to the subtype."""
    step = _bare_step03()
    struct = _structure()
    struct['de_relations'] = [
        types.SimpleNamespace(source='ACAT1', target='DBT', type='inferred', subtype='regulation'),
    ]
    rel = step._synthesize_mechanisms_from_structures([struct])[0]['curatedRelations'][0]
    assert rel['type'] == 'regulation'


def test_synthesize_relation_type_falls_through_to_type():
    step = _bare_step03()
    struct = _structure()
    struct['de_relations'] = [
        types.SimpleNamespace(source='ACAT1', target='DBT', type='PPrel', subtype=''),
    ]
    rel = step._synthesize_mechanisms_from_structures([struct])[0]['curatedRelations'][0]
    assert rel['type'] == 'PPrel'


def test_generate_summary_from_structures_more_than_five():
    step = _bare_step03()
    mechs = [{'pathway': f'P{i}'} for i in range(7)]
    summary = step._generate_summary_from_structures(mechs)
    assert 'and 2 more' in summary
    assert 'P4' in summary and 'P5' not in summary  # only first 5 listed


def test_synthesized_output_flows_to_step4_consumers():
    """Synthesized Step 3 output must be usable by Step 4's gate and query tools."""
    step = _bare_step03()
    structs = [_structure()]
    step3_out = {
        'pathway_mechanisms': step._synthesize_mechanisms_from_structures(structs),
        'pathway_structures': [step._pathway_structure_to_dict(ps) for ps in structs],
        'pathway_overlaps': [],
    }
    assert Step04HypothesisGeneration._has_pathway_data(step3_out) is True
    rels = PathwayQueryService(step3_out).get_pathway_mechanism('Butanoate metabolism')['curatedRelations']
    assert (rels[0]['source'], rels[0]['target']) == ('ACAT1', 'DBT')
