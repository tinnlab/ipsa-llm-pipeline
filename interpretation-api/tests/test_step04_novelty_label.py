"""
Novelty label rename: "builds-on-known" -> "extended".

The novelty value is free text returned by the LLM, so the prompt change alone is a
request, not a guarantee. These tests pin the prompt vocabulary and the deterministic
normalisation that maps the legacy label (and variants) onto the canonical one before
anything is stored or rendered (no LLM required).
"""

import pytest

from src.pipeline.steps.step04_hypothesis_generation import (
    Step04HypothesisGeneration,
    normalize_novelty,
    NOVELTY_LABELS,
)
from src.pipeline.steps.step06_report_generation import Step06ReportGeneration


GENES = [
    {'geneSymbol': 'ACADS', 'foldChange': -1.4},
    {'geneSymbol': 'CDK1', 'foldChange': +2.4},
]
PATHWAYS = [
    {'name': 'Fatty acid degradation', 'NES': -3.4, 'genes': ['ACADS']},
    {'name': 'Cell cycle', 'NES': 2.1, 'genes': ['CDK1']},
]


def _bare_step04():
    return Step04HypothesisGeneration.__new__(Step04HypothesisGeneration)


@pytest.mark.parametrize('raw, expected', [
    ('builds-on-known', 'extended'),
    ('Builds-On-Known', 'extended'),
    ('builds on known', 'extended'),
    ('builds_on_known', 'extended'),
    ('  builds-on-known  ', 'extended'),
    ('extended', 'extended'),
    ('Extended', 'extended'),
    ('novel', 'novel'),
    ('confirmatory', 'confirmatory'),
    ('speculative', 'speculative'),
    ('', 'unknown'),
    ('   ', 'unknown'),
    (None, 'unknown'),
    (3, 'unknown'),
    (['novel'], 'unknown'),
])
def test_normalize_novelty_table(raw, expected):
    out = normalize_novelty(raw)
    assert isinstance(out, str)
    assert out == expected


def test_canonical_labels_are_the_prompt_vocabulary():
    assert NOVELTY_LABELS == ('novel', 'extended', 'confirmatory')


@pytest.mark.parametrize('context', ['HCC', ''])
def test_prompt_asks_for_extended_not_builds_on_known(context):
    prompt = _bare_step04()._build_user_prompt(
        genes=GENES, pathways=PATHWAYS, themes=None, hub_genes_result=None,
        mechanisms_result=None, experiment_context=context,
    )
    assert '"novelty": "novel | extended | confirmatory"' in prompt
    assert 'extended = adds detail to established biology' in prompt
    assert 'builds-on-known' not in prompt
    if context:
        # The disease-biology instruction is only emitted with an experiment context.
        assert 'label it "extended"' in prompt


def test_stored_hypotheses_are_canonicalised_in_place():
    result = {'hypotheses': [
        {'hypothesis': 'a', 'novelty': 'builds-on-known'},
        {'hypothesis': 'b', 'novelty': 'novel'},
        {'hypothesis': 'c'},
        'not-a-dict',
    ]}
    Step04HypothesisGeneration._canonicalize_novelty_labels(result)
    assert result['hypotheses'][0]['novelty'] == 'extended'
    assert result['hypotheses'][1]['novelty'] == 'novel'
    assert 'novelty' not in result['hypotheses'][2]
    assert result['hypotheses'][3] == 'not-a-dict'


def test_step04_report_section_renders_extended():
    result = {'hypotheses': [
        {'hypothesis': 'ACADS loss', 'confidence': 'high', 'novelty': 'builds-on-known'},
        {'hypothesis': 'CDK1 gain', 'confidence': 'low'},
    ]}
    md = _bare_step04()._generate_report_section(result, 'HCC')
    assert '*Confidence: HIGH | Novelty: extended*' in md
    assert '*Confidence: LOW | Novelty: unknown*' in md
    assert 'builds-on-known' not in md


def test_step06_fallback_renders_extended():
    step6 = Step06ReportGeneration.__new__(Step06ReportGeneration)
    section = step6._generate_hypotheses_section({'hypotheses': [
        {'hypothesis': 'ACADS loss', 'confidence': 'medium', 'novelty': 'builds-on-known'},
    ]})
    assert '*Confidence: MEDIUM | Novelty: extended*' in section.content
    assert 'builds-on-known' not in section.content


def test_step06_prefers_stored_report_section_verbatim():
    """A pre-rendered report_section is used as-is: normalisation does not rewrite
    markdown that was already produced (stored reports keep what they contain)."""
    step6 = Step06ReportGeneration.__new__(Step06ReportGeneration)
    stored = '*Confidence: HIGH | Novelty: builds-on-known*\n'
    section = step6._generate_hypotheses_section(
        {'hypotheses': [], 'report_section': stored})
    assert section.content == stored
