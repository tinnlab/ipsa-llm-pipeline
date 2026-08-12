"""
Tests for report bugs 1 + 2 (single root cause): the LLM echoes a themed pathway name
with the KEGG-id suffix it was shown ("Cell cycle (hsa04110)"). Left unnormalized that
suffix (a) makes Part-C backfill re-add the pathway as a duplicate bare stub, and (b)
misses the confidence lookup so the real card reads "Unverified" while the stub reads
"High Confidence" — the same pathway rendered twice with two confidences.

The fix is a shared id-stripping key used for canonicalization, coverage, the confidence
lookup, and de-dup.
"""

from src.pipeline.steps.step03_pathway_mechanisms import (
    _strip_pathway_id_suffix,
    _pathway_key,
    _canonicalize_mechanism_names,
    _dedupe_mechanisms,
    _structures_missing_mechanisms,
    Step03PathwayMechanisms,
)


def _bare_step03():
    return Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)


def _structure(name, confidence='high', nes=1.77):
    return {
        'pathway': name,
        'source': 'kegg',
        'confidence': confidence,
        'enrichment_score': nes,
        'enrichment_metric': 'NES',
        'enrichment_direction': 'upregulated',
    }


# ---------------------------------------------------------------------------
# id-suffix normalization
# ---------------------------------------------------------------------------

def test_strip_kegg_id_suffix():
    assert _strip_pathway_id_suffix('Cell cycle (hsa04110)') == 'Cell cycle'
    assert _strip_pathway_id_suffix('Ribosome (KEGG: hsa03008)') == 'Ribosome'
    assert _strip_pathway_id_suffix('Foo (REACTOME: R-HSA-69278)') == 'Foo'
    assert _strip_pathway_id_suffix('Bar (R-HSA-123)') == 'Bar'


def test_strip_preserves_legitimate_parentheticals():
    # An id-format match only — a real name with a non-id parenthetical is untouched.
    name = 'Glycosylphosphatidylinositol (GPI)-anchor biosynthesis'
    assert _strip_pathway_id_suffix(name) == name


def test_pathway_key_collapses_variants():
    assert _pathway_key('Cell cycle (hsa04110)') == _pathway_key('cell cycle') == 'cell cycle'


# ---------------------------------------------------------------------------
# canonicalization + de-dup
# ---------------------------------------------------------------------------

def test_canonicalize_rewrites_echoed_id_name_to_structure_name():
    mechs = [{'pathway': 'Cell cycle (hsa04110)', 'biologicalFunction': 'x'}]
    _canonicalize_mechanism_names(mechs, [_structure('Cell cycle')])
    assert mechs[0]['pathway'] == 'Cell cycle'


def test_dedupe_collapses_idsuffixed_and_clean_duplicate():
    full = {'pathway': 'Cell cycle (hsa04110)', 'biologicalFunction': 'Coordinates division',
            'functionalConsequences': 'proliferation'}
    stub = {'pathway': 'Cell cycle', 'biologicalFunction': ''}
    # After canonicalization both share the structure name, so de-dup keeps the rich one.
    _canonicalize_mechanism_names([full, stub], [_structure('Cell cycle')])
    out = _dedupe_mechanisms([full, stub])
    assert len(out) == 1
    assert out[0]['biologicalFunction'] == 'Coordinates division'


# ---------------------------------------------------------------------------
# end-to-end render: one card, one confidence
# ---------------------------------------------------------------------------

def test_report_renders_pathway_once_with_consistent_confidence():
    step = _bare_step03()
    structures = [_structure('Cell cycle', confidence='high')]
    mechs = [
        {'pathway': 'Cell cycle (hsa04110)', 'biologicalFunction': 'Coordinates division',
         'deGeneInvolvement': [{'gene': 'CDK1', 'foldChange': 2.41}],
         'functionalConsequences': 'drives proliferation'},
        {'pathway': 'Cell cycle', 'biologicalFunction': '',
         'deGeneInvolvement': [{'gene': 'CDK1', 'foldChange': 2.41}]},
    ]
    _canonicalize_mechanism_names(mechs, structures)
    md = step._generate_report_section(
        {'pathwayMechanisms': mechs, 'mechanisticSummary': 'summary'}, structures)

    assert md.count('#### Cell cycle') == 1
    assert 'hsa04110' not in md                      # id suffix normalized away
    assert 'High Confidence' in md
    assert 'Unverified' not in md


# ---------------------------------------------------------------------------
# Part-C backfill coverage: an id-suffixed echo must not trigger a duplicate stub
# ---------------------------------------------------------------------------

def test_idsuffixed_echo_is_not_treated_as_missing():
    # This is the coverage check inside execute() that prevents the duplicate bare stub.
    structures = [_structure('Cell cycle'), _structure('Ribosome')]
    mechs = [{'pathway': 'Cell cycle (hsa04110)', 'biologicalFunction': 'x'}]
    _canonicalize_mechanism_names(mechs, structures)
    missing = _structures_missing_mechanisms(mechs, structures)
    # Cell cycle is covered (despite the id suffix echo); only Ribosome is missing.
    assert [m['pathway'] for m in missing] == ['Ribosome']


def test_missing_detects_genuinely_uncovered_structure():
    structures = [_structure('Cell cycle')]
    missing = _structures_missing_mechanisms([], structures)
    assert len(missing) == 1
