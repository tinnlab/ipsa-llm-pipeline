"""
The engine, not the model, names each Step 3 card. Every retrieved pathway gets a tag
(P1, P2, ...) shown in the prompt; the model returns it with each card as pathwayRef and
the card's title is set to that pathway's retrieved name. A reworded name ("Ribosome
biogenesis" for "Ribosome biogenesis in eukaryotes") therefore still links the card to
the engine's data, so the relation, gene and crosstalk checks run on it and no second
backfilled card appears.
"""

from types import SimpleNamespace

from src.pipeline.steps.step03_pathway_mechanisms import (
    Step03PathwayMechanisms,
    _structures_missing_mechanisms,
    assign_pathway_refs,
    resolve_mechanism_pathways,
)


def _ps(name):
    return {'pathway': name, 'source': 'kegg', 'confidence': 'high', 'mapped_de_genes': [],
            'de_relations': [], 'structure': SimpleNamespace(id='hsa00000', genes=[])}


def _structures():
    s = [_ps('Ribosome'), _ps('Ribosome biogenesis in eukaryotes'), _ps('Cell cycle')]
    assign_pathway_refs(s)
    return s


def test_refs_are_assigned_in_order_and_kept():
    s = _structures()
    assert [p['ref'] for p in s] == ['P1', 'P2', 'P3']
    assign_pathway_refs(s)
    assert [p['ref'] for p in s] == ['P1', 'P2', 'P3']


def test_reworded_name_is_replaced_by_the_engine_name():
    s = _structures()
    mechs = [{'pathwayRef': 'P2', 'pathway': 'Ribosome biogenesis'}]
    counts = resolve_mechanism_pathways(mechs, s)
    assert mechs[0]['pathway'] == 'Ribosome biogenesis in eukaryotes'
    assert 'pathwayRef' not in mechs[0]
    assert counts['by_ref'] == 1
    # The real pathway is covered, so no second (backfilled) card is added for it.
    missing = {p['pathway'] for p in _structures_missing_mechanisms(mechs, s)}
    assert 'Ribosome biogenesis in eukaryotes' not in missing


def test_tag_formats_accepted():
    s = _structures()
    for ref in ('P3', 'p3', '[P3]', '3', ' P3 '):
        mechs = [{'pathwayRef': ref, 'pathway': 'something else'}]
        resolve_mechanism_pathways(mechs, s)
        assert mechs[0]['pathway'] == 'Cell cycle', ref


def test_missing_or_unknown_tag_falls_back_to_exact_name():
    s = _structures()
    mechs = [{'pathway': 'cell cycle (hsa04110)'}, {'pathwayRef': 'P99', 'pathway': 'Ribosome'}]
    counts = resolve_mechanism_pathways(mechs, s)
    assert [m['pathway'] for m in mechs] == ['Cell cycle', 'Ribosome']
    assert counts['by_name'] == 2


def test_name_exactly_matching_another_pathway_wins_over_tag():
    s = _structures()
    mechs = [{'pathwayRef': 'P1', 'pathway': 'Cell cycle'}]
    counts = resolve_mechanism_pathways(mechs, s)
    assert mechs[0]['pathway'] == 'Cell cycle'
    assert counts['ref_name_conflict'] == 1


def test_unmatched_card_is_counted_and_left_alone():
    s = _structures()
    mechs = [{'pathway': 'Made-up pathway'}]
    counts = resolve_mechanism_pathways(mechs, s)
    assert mechs[0]['pathway'] == 'Made-up pathway' and counts['unmatched'] == 1


def test_prompt_shows_the_tag_and_asks_for_it():
    step = Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)
    s = _structures()
    block = step._format_pathway_structure_for_llm(s[1])
    assert block.startswith('**Pathway P2: Ribosome biogenesis in eukaryotes** (KEGG: hsa00000)')
    prompt = step._build_user_prompt(s, [], None)
    assert '"pathwayRef"' in prompt


def test_untagged_structure_prompt_unchanged():
    step = Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)
    block = step._format_pathway_structure_for_llm(_ps('Cell cycle'))
    assert block.startswith('**Pathway: Cell cycle** (KEGG: hsa00000)')
