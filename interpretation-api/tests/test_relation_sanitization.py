"""
Tests for report bug 6: model scratch text leaked into the report, e.g.
"ALLD2 --[compound]--> AKR1A1: Typo corrected: ALDH2 to AKR1A1...". Two data-driven
guards: (1) drop relations whose gene-shaped endpoint is not a real node in the gene
universe (removes the hallucinated "ALLD2"); (2) strip self-correction / meta-commentary
from free-text fields.
"""

from types import SimpleNamespace

from src.pipeline.fc_utils import sanitize_llm_text
from src.pipeline.steps.step03_pathway_mechanisms import Step03PathwayMechanisms


def _bare_step03():
    return Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)


def _entry(symbol):
    return SimpleNamespace(id=f'e_{symbol}', gene_symbol=symbol, name=symbol, names=[symbol])


def _structure_ps(compounds=('cAMP',)):
    structure = SimpleNamespace(
        id='hsa00620',
        genes=[_entry('ALDH2'), _entry('ADH1A'), _entry('AKR1A1'), _entry('PRKACA')],
        compounds=[_entry(c) for c in compounds],
    )
    return {
        'pathway': 'Pyruvate metabolism',
        'mapped_de_genes': [SimpleNamespace(gene_symbol='ALDH2')],
        'structure': structure,
        'de_relations': [SimpleNamespace(source='ALDH2', target='ADH1A', subtype='compound')],
    }


def _inferred_ps():
    """An inferred/PC-grounded pathway: MockStructure with no gene/compound inventory."""
    structure = SimpleNamespace(id='inferred1')  # no .genes / .compounds
    return {
        'pathway': 'Inferred signaling',
        'mapped_de_genes': [SimpleNamespace(gene_symbol='TP53')],
        'structure': structure,
        'de_relations': [],
    }


# ---------------------------------------------------------------------------
# sanitize_llm_text
# ---------------------------------------------------------------------------

def test_sanitize_removes_typo_correction_clause():
    text = 'Typo corrected: ALDH2 to AKR1A1; down-regulation may affect aldehyde reduction.'
    assert sanitize_llm_text(text) == 'down-regulation may affect aldehyde reduction.'


def test_sanitize_removes_first_person_metacommentary():
    text = 'This gene is key. I previously stated it was down, but up. It drives growth.'
    assert sanitize_llm_text(text) == 'This gene is key. It drives growth.'


def test_sanitize_leaves_clean_text_untouched():
    text = 'Promotes IKKalpha activation and NF-kappaB transcription.'
    assert sanitize_llm_text(text) == text


# ---------------------------------------------------------------------------
# _validate_against_kegg: drop hallucinated endpoints + sanitize
# ---------------------------------------------------------------------------

def test_relation_with_unknown_gene_endpoint_is_dropped():
    step = _bare_step03()
    result = {'pathwayMechanisms': [{
        'pathway': 'Pyruvate metabolism',
        'curatedRelations': [
            {'source': 'ALLD2', 'target': 'AKR1A1', 'type': 'compound',
             'interpretation': 'Typo corrected: ALDH2 to AKR1A1; may affect reduction.'},
            {'source': 'ALDH2', 'target': 'ADH1A', 'type': 'compound',
             'interpretation': 'Reduced ALDH2 may decrease acetaldehyde conversion.'},
        ],
    }]}
    out = step._validate_against_kegg(result, [_structure_ps()])
    rels = out['pathwayMechanisms'][0]['curatedRelations']
    assert len(rels) == 1
    assert rels[0]['source'] == 'ALDH2'
    # The leaked self-correction text is gone with the dropped relation.
    assert all('Typo corrected' not in r.get('interpretation', '') for r in rels)


def test_metabolite_endpoints_are_not_dropped():
    step = _bare_step03()
    result = {'pathwayMechanisms': [{
        'pathway': 'Pyruvate metabolism',
        'curatedRelations': [
            {'source': 'acetyl-CoA', 'target': 'ALDH2', 'type': 'compound',
             'interpretation': 'Metabolite feeds the enzyme.'},
        ],
    }]}
    out = step._validate_against_kegg(result, [_structure_ps()])
    # 'acetyl-CoA' is not gene-shaped, so it is not judged as a bogus gene.
    assert len(out['pathwayMechanisms'][0]['curatedRelations']) == 1


def test_free_text_fields_are_sanitized():
    step = _bare_step03()
    result = {'pathwayMechanisms': [{
        'pathway': 'Pyruvate metabolism',
        'biologicalFunction': 'Central metabolism. Let me revise: it is glycolytic.',
        'curatedRelations': [],
    }]}
    out = step._validate_against_kegg(result, [_structure_ps()])
    bf = out['pathwayMechanisms'][0]['biologicalFunction']
    assert 'Let me revise' not in bf
    assert 'Central metabolism.' in bf


def test_gene_shaped_metabolite_in_compounds_is_kept():
    # cAMP -> "CAMP" is gene-shaped but is a real compound node; must NOT be dropped.
    step = _bare_step03()
    result = {'pathwayMechanisms': [{
        'pathway': 'Pyruvate metabolism',
        'curatedRelations': [
            {'source': 'PRKACA', 'target': 'cAMP', 'type': 'compound',
             'interpretation': 'Kinase responds to cAMP.'},
        ],
    }]}
    out = step._validate_against_kegg(result, [_structure_ps(compounds=('cAMP',))])
    assert len(out['pathwayMechanisms'][0]['curatedRelations']) == 1


def test_real_gene_in_structure_inventory_is_kept():
    step = _bare_step03()
    result = {'pathwayMechanisms': [{
        'pathway': 'Pyruvate metabolism',
        'curatedRelations': [
            {'source': 'PRKACA', 'target': 'ADH1A', 'type': 'activation',
             'interpretation': 'Both are pathway members.'},
        ],
    }]}
    out = step._validate_against_kegg(result, [_structure_ps()])
    assert len(out['pathwayMechanisms'][0]['curatedRelations']) == 1


def test_inferred_pathway_without_inventory_is_not_judged():
    # No gene/compound inventory → we can't validate → we must NOT drop (keep TP53→MDM2).
    step = _bare_step03()
    result = {'pathwayMechanisms': [{
        'pathway': 'Inferred signaling',
        'curatedRelations': [
            {'source': 'TP53', 'target': 'MDM2', 'type': 'inhibition',
             'interpretation': 'p53 represses MDM2.'},
        ],
    }]}
    out = step._validate_against_kegg(result, [_inferred_ps()])
    assert len(out['pathwayMechanisms'][0]['curatedRelations']) == 1


def test_endpoint_valid_only_via_de_relations_is_kept():
    # An endpoint present only in the real KEGG de_relations (not genes/compounds) is a
    # legitimate node and must be kept.
    structure = SimpleNamespace(id='hsa00620', genes=[_entry('ALDH2')], compounds=[])
    ps = {
        'pathway': 'Pyruvate metabolism',
        'mapped_de_genes': [SimpleNamespace(gene_symbol='ALDH2')],
        'structure': structure,
        'de_relations': [SimpleNamespace(source='ALDH2', target='FOO9', subtype='activation')],
    }
    step = _bare_step03()
    result = {'pathwayMechanisms': [{
        'pathway': 'Pyruvate metabolism',
        'curatedRelations': [
            {'source': 'ALDH2', 'target': 'FOO9', 'type': 'activation', 'interpretation': 'ok'},
            {'source': 'ALDH2', 'target': 'BOGUS1', 'type': 'activation', 'interpretation': 'x'},
        ],
    }]}
    out = step._validate_against_kegg(result, [ps])
    kept = out['pathwayMechanisms'][0]['curatedRelations']
    targets = {r['target'] for r in kept}
    assert 'FOO9' in targets      # only in de_relations, still valid
    assert 'BOGUS1' not in targets  # nowhere in the inventory → dropped


def test_mechanistic_summary_is_sanitized():
    step = _bare_step03()
    result = {'mechanisticSummary': 'Key programs up. I previously miscounted them. Done.',
              'pathwayMechanisms': []}
    out = step._validate_against_kegg(result, [_structure_ps()])
    assert 'previously miscounted' not in out['mechanisticSummary']


def test_sanitize_leaves_midsentence_trigger_words_intact():
    # Anchoring guard: a correction-like substring mid-sentence must not be stripped.
    assert sanitize_llm_text('The corrected AKT to MTOR axis is intact.') == \
        'The corrected AKT to MTOR axis is intact.'
    assert sanitize_llm_text('Signaling flows from RAS to RAF.') == \
        'Signaling flows from RAS to RAF.'


def test_sanitize_decimal_inside_correction_clause():
    # A decimal inside the removed clause must not split it and leave a misleading residue.
    out = sanitize_llm_text('Typo corrected: ALDH2 was 1.5 fold to AKR1A1. Real content.')
    assert out == 'Real content.'


# ---------------------------------------------------------------------------
# Unresolved placeholder / scratch text (the "e-?? not listed" leak)
# ---------------------------------------------------------------------------

def test_sanitize_strips_placeholder_pvalue_parenthetical():
    # The exact leak from the TCGA-LIHC report: a value the model could not source,
    # emitted as a stub. Keep the sourced fold change; drop the placeholder parenthetical.
    text = 'CDCA8 up-regulated 2.72-fold (p=2.85e-?? not listed but hub score high)'
    assert sanitize_llm_text(text) == 'CDCA8 up-regulated 2.72-fold'


def test_sanitize_strips_not_listed_parenthetical():
    assert sanitize_llm_text('STEAP3 down-regulated (evidence not listed)') == \
        'STEAP3 down-regulated'


def test_sanitize_preserves_real_pvalue_and_fold_change_parentheticals():
    # A legitimate p-value has no '?' and must survive; so must signed-FC parentheticals.
    for text in (
        'CDK1 up-regulated 2.41-fold (p=2.78e-157)',
        'NHP2 (^1.18) pseudouridylates rRNA',
        'FBL (FC: +1.12) methylates rRNA',
        'ACAT1 (v1.02) thiolysis of acetoacetyl-CoA',
        'Ribosome pathway NES = +1.81, FDR = 3.76e-04',
    ):
        assert sanitize_llm_text(text) == text


def test_sanitize_strips_bare_placeholder_pvalue_token():
    # Even without surrounding parentheses, a placeholder p-value token is removed.
    out = sanitize_llm_text('MKI67 up-regulated 3.29-fold, p=e-?? and hub score high')
    assert 'e-??' not in out
    assert 'MKI67 up-regulated 3.29-fold' in out


def test_sanitize_does_not_strip_word_ending_in_p_before_unknown():
    # The 'p unknown' marker is \b-anchored: "top unknown" must not trigger a strip.
    text = 'Its role (top unknown-function gene) remains under study.'
    assert sanitize_llm_text(text) == text


def test_sanitize_keeps_tentative_question_parentheticals():
    # A SINGLE rhetorical/tentative '?' is preserved; only a value-placeholder '??' run or
    # a '=?' assignment is treated as scratch. This must not over-strip hedged parentheticals
    # that merely end in a digit/'e' before the '?'.
    for text in (
        'The regulator (HNF4A?) may drive this axis.',
        'Its net effect (direction unclear?) needs testing.',
        'Marginally significant (p<0.05?) association.',
        'Shown earlier (see Fig 2e?) in the panel.',
    ):
        assert sanitize_llm_text(text) == text
    # ...but a value placeholder inside parens is still stripped.
    assert sanitize_llm_text('CDK1 up (p=e-?? here) drives it.') == 'CDK1 up drives it.'
    assert sanitize_llm_text('Value (= ?) not resolved.') == 'Value not resolved.'


def test_sanitize_keeps_complete_pvalue_with_trailing_hedge():
    # A COMPLETE p-value with a trailing rhetorical '?' has a real value — it is not a
    # placeholder and must survive intact (no strip, no dangling empty paren).
    assert sanitize_llm_text('Marginal significance (p=0.05?).') == \
        'Marginal significance (p=0.05?).'
    assert sanitize_llm_text('reported p = 0.03? in cohort') == 'reported p = 0.03? in cohort'
    # ...whereas a p-value whose value/exponent is a placeholder is still stripped.
    assert sanitize_llm_text('MKI67 3.29-fold, p=e-?? and high') == 'MKI67 3.29-fold and high'


def test_sanitize_leaves_no_dangling_empty_parens():
    # A single-'?' exponent placeholder that is the sole paren content is stripped WITHOUT
    # leaving an empty "()" behind.
    assert sanitize_llm_text('CDK1 up (p=2.5e-?) drives it.') == 'CDK1 up drives it.'
    assert sanitize_llm_text('x (p=e-?) y') == 'x y'
