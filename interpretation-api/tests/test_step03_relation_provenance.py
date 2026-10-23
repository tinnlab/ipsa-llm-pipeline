"""Step 3 relation provenance: the report heading for a pathway's relations names only the
provenance that pathway has, and the mechanism prompt only asks the model to cite KEGG when
every pathway in the batch came from KEGG."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.pipeline.steps.step03_pathway_mechanisms import (
    Step03PathwayMechanisms,
    _relations_heading,
)
from src.pipeline.steps.step06_report_generation import (
    Step06ReportGeneration,
    _pathway_databases,
    _step3_methods_text,
)

ORIGINAL_KEGG_INTRO = 'Interpret these curated KEGG pathway structures in light of differentially expressed genes:'


def _bare_step03():
    step = Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)
    step.step_number = 3
    step.step_name = 'Pathway Mechanisms and Interactions'
    step.llm = MagicMock()
    return step


@pytest.mark.parametrize('source,confidence,heading', [
    ('kegg', 'high', '**KEGG Curated Regulatory Relations:**'),
    ('reactome', 'high', '**REACTOME Curated Regulatory Relations:**'),
    ('other', 'pc_grounded', '**Pathway Commons Curated Interactions:**'),
    ('other', 'inferred', '**Inferred Regulatory Relations (LLM-derived):**'),
    ('other', 'gene_set', '**Regulatory Relations:**'),
    ('kegg', 'gene_set', '**Regulatory Relations:**'),
    ('KEGG', 'unknown', '**Regulatory Relations:**'),
])
def test_relations_heading_mapping(source, confidence, heading):
    assert _relations_heading(source, confidence) == heading


def _section(source, confidence):
    result = {'pathwayMechanisms': [{
        'pathway': 'Mitochondrial complex I',
        'biologicalFunction': 'Electron transport',
        'curatedRelations': [{'source': 'NDUFA4', 'target': 'NDUFB8', 'type': 'binding'}],
    }]}
    structures = [{'pathway': 'Mitochondrial complex I', 'source': source, 'confidence': confidence}]
    return _bare_step03()._generate_report_section(result, pathway_structures=structures)


def test_kegg_curated_heading_unchanged():
    section = _section('kegg', 'high')
    assert '**Source:** KEGG' in section
    assert '**KEGG Curated Regulatory Relations:**' in section


@pytest.mark.parametrize('confidence,heading', [
    ('pc_grounded', '**Pathway Commons Curated Interactions:**'),
    ('inferred', '**Inferred Regulatory Relations (LLM-derived):**'),
    ('gene_set', '**Regulatory Relations:**'),
])
def test_non_kegg_pathway_is_not_labelled_kegg(confidence, heading):
    section = _section('other', confidence)
    assert 'KEGG' not in section
    assert heading in section
    assert '- NDUFA4 --[binding]--> NDUFB8' in section


def test_unmatched_pathway_makes_no_provenance_claim():
    result = {'pathwayMechanisms': [{
        'pathway': 'Not in structures',
        'biologicalFunction': 'x',
        'curatedRelations': [{'source': 'A1', 'target': 'B2', 'type': 'activation'}],
    }]}
    section = _bare_step03()._generate_report_section(result, pathway_structures=[])
    assert '**Regulatory Relations:**' in section
    assert 'Curated' not in section


def test_relation_gene_symbol_never_becomes_the_database_name():
    # Two pathways: the first one's last relation source must not leak into the second's heading.
    result = {'pathwayMechanisms': [
        {'pathway': 'P1', 'biologicalFunction': 'x',
         'curatedRelations': [{'source': 'TP53', 'target': 'MDM2', 'type': 'inhibition'}]},
        {'pathway': 'P2', 'biologicalFunction': 'y',
         'curatedRelations': [{'source': 'CDK1', 'target': 'CCNB1', 'type': 'binding'}]},
    ]}
    structures = [{'pathway': 'P1', 'source': 'kegg', 'confidence': 'high'},
                  {'pathway': 'P2', 'source': 'kegg', 'confidence': 'high'}]
    section = _bare_step03()._generate_report_section(result, pathway_structures=structures)
    assert section.count('**KEGG Curated Regulatory Relations:**') == 2
    assert 'TP53 Curated' not in section and 'CDK1 Curated' not in section


def _ps(source, confidence):
    gene = SimpleNamespace(gene_symbol='NDUFA4', direction='up', fold_change=1.5, p_value=1e-4)
    return {'pathway': f'{source} pathway', 'source': source, 'confidence': confidence,
            'structure': SimpleNamespace(id='X1'), 'mapped_de_genes': [gene], 'de_relations': []}


def test_all_kegg_batch_keeps_original_prompt_wording():
    prompt = _bare_step03()._build_user_prompt([_ps('kegg', 'high'), _ps('kegg', 'gene_set')], [])
    assert prompt.startswith(ORIGINAL_KEGG_INTRO)
    assert 'Summarize what this pathway does (from KEGG structure)' in prompt
    assert 'Which KEGG relations involve DE genes' in prompt
    assert 'citing KEGG relations and DE gene patterns' in prompt


def test_non_kegg_batch_prompt_does_not_assert_kegg():
    prompt = _bare_step03()._build_user_prompt([_ps('other', 'inferred'), _ps('other', 'gene_set')], [])
    assert 'KEGG' not in prompt
    assert '"curatedRelations"' in prompt  # schema field kept for downstream consumers


def test_mixed_batch_uses_source_neutral_wording():
    prompt = _bare_step03()._build_user_prompt([_ps('kegg', 'high'), _ps('other', 'pc_grounded')], [])
    assert not prompt.startswith(ORIGINAL_KEGG_INTRO)
    assert 'from KEGG structure' not in prompt
    assert 'citing KEGG relations' not in prompt
    assert '[CURATED]' in prompt  # the KEGG pathway is still shown as curated


# ---- report-visible provenance strings outside the relation heading ----------------

def test_empty_step3_report_does_not_name_kegg():
    assert 'KEGG' not in _bare_step03()._generate_empty_report()


def _bare_step06():
    step = Step06ReportGeneration.__new__(Step06ReportGeneration)
    step.llm = MagicMock()
    return step


def _appendix(pathways, step3):
    section = _bare_step06()._generate_appendices(
        input_data={'metadata': {'organism': 'Mus musculus'}, 'pathways': pathways},
        step1={}, step2={'metadata': {'skipped': True}}, step3=step3, step4={})
    return section.content


def test_methods_name_the_custom_database_not_kegg():
    content = _appendix(
        [{'name': 'Complex I', 'source': 'MitoPathways'}, {'name': 'OXPHOS', 'source': 'MitoPathways'}],
        {'pathway_structures': [{'source': 'other', 'confidence': 'inferred'},
                                {'source': 'other', 'confidence': 'pc_grounded'}]})
    assert '- **Pathway database:** MitoPathways\n' in content
    assert ('- **Step 3:** Pathway mechanism analysis '
            '(LLM-inferred relations; Pathway Commons interactions)') in content
    assert 'KEGG' not in content and 'Reactome' not in content


def test_methods_name_kegg_when_kegg_was_used():
    content = _appendix([{'name': 'Cell cycle', 'source': 'KEGG'}],
                        {'pathway_structures': [{'source': 'kegg', 'confidence': 'high'}]})
    assert '- **Pathway database:** KEGG\n' in content
    assert '(curated KEGG pathway structures)' in content


def test_methods_fall_back_to_neutral_wording():
    assert _pathway_databases({'pathways': [{'source': 'Unknown'}, {'source': ''}, {}]}) == []
    content = _appendix([{'name': 'x', 'source': 'Unknown'}], {})
    assert '- **Pathway database:** not specified in the input\n' in content
    assert '- **Step 3:** Pathway mechanism analysis\n' in content


def test_methods_lists_each_database_once_in_order():
    assert _pathway_databases({'pathways': [{'source': 'GO_BP'}, {'source': 'KEGG'}, {'source': 'GO_BP'}]}) \
        == ['GO_BP', 'KEGG']


def test_methods_step3_skipped_without_genes():
    assert _step3_methods_text({'metadata': {'skipped': True}}) == 'Skipped (no gene expression data)'


def test_step6_mechanisms_fallback_does_not_claim_curated():
    section = _bare_step06()._generate_mechanisms_section(
        {'pathway_mechanisms': [{'pathway': 'P', 'curatedRelations': []}]})
    assert 'curated' not in section.content.lower()
