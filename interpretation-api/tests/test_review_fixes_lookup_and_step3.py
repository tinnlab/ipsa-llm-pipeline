"""
Fixes from the independent review of the paper-code branch:
- pathway name lookups (KEGG find, Step 4 tool) take the exact name before any partial
  match, so "Ribosome" and "Ribosome biogenesis in eukaryotes" never swap or share a map;
- Step 4 coverage compares normalized pathway names;
- plain `chat()` flattens tool-call history (no tools are sent there);
- Step 3: crosstalk ignores gene-like words inside pathway names, overlaps are
  case-insensitive, KEGG misses get the input-gene filter for LLM relations, a GMT file
  for another organism is never used, gene-set cards list every gene of a box.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.pipeline.services.kegg_service import PathwayEntry
from src.agents.llm_client import UnifiedLLMClient as LLMClient
from src.pipeline.services.kegg_service import _choose_kegg_hit
from src.pipeline.services.pathway_membership_service import PathwayMembershipService
from src.pipeline.services.pathway_query_service import PathwayQueryService
from src.pipeline.steps.step03_pathway_mechanisms import (
    Step03PathwayMechanisms,
    check_crosstalk,
)
from tests.test_bug4_tool_gating import _bare_step04, _structures_only_step3

# ---- KEGG find: exact title first ----------------------------------------------------

FIND_RIBOSOME = ['path:map03008\tRibosome biogenesis in eukaryotes', 'path:map03010\tRibosome']


def test_kegg_find_prefers_exact_title_over_first_hit():
    assert _choose_kegg_hit(FIND_RIBOSOME, 'Ribosome', 'hsa') == 'map03010'
    assert _choose_kegg_hit(FIND_RIBOSOME, 'Ribosome biogenesis in eukaryotes', 'hsa') == 'map03008'


def test_kegg_find_exact_title_ignores_organism_suffix_and_prefers_org_id():
    lines = ['hsa03008\tRibosome biogenesis in eukaryotes - Homo sapiens (human)',
             'map03010\tRibosome', 'hsa03010\tRibosome - Homo sapiens (human)']
    assert _choose_kegg_hit(lines, 'ribosome', 'hsa') == 'hsa03010'


def test_kegg_find_without_exact_title_keeps_old_rule():
    lines = ['map04010\tMAPK signaling pathway', 'hsa04010\tMAPK signaling pathway - Homo sapiens (human)']
    assert _choose_kegg_hit(lines, 'MAPK cascade', 'hsa') == 'hsa04010'
    assert _choose_kegg_hit(['map04010\tMAPK signaling pathway'], 'MAPK', 'hsa') == 'map04010'
    assert _choose_kegg_hit([], 'x', 'hsa') is None


# ---- Step 4 tool lookup ----------------------------------------------------------------

def _service(mechs, structs):
    return PathwayQueryService({
        'pathway_mechanisms': [{'pathway': m, 'biologicalFunction': f'{m} function'} for m in mechs],
        'pathway_structures': [{'pathway': s, 'mapped_de_genes': []} for s in structs],
        'pathway_overlaps': []})


def test_tool_lookup_exact_structure_name_beats_partial_mechanism_name():
    svc = _service(['Ribosome biogenesis in eukaryotes'], ['Ribosome', 'Ribosome biogenesis in eukaryotes'])
    assert svc._fuzzy_match_pathway('Ribosome') == 'ribosome'
    assert svc.get_pathway_mechanism('Ribosome')['pathway'] == 'Ribosome'


def test_tool_lookup_long_name_does_not_return_the_short_pathway():
    svc = _service(['Ribosome'], ['Ribosome', 'Ribosome biogenesis in eukaryotes'])
    out = svc.get_pathway_mechanism('Ribosome biogenesis in eukaryotes')
    assert out['pathway'] == 'Ribosome biogenesis in eukaryotes'
    assert out['biologicalFunction'] == ''  # not the Ribosome card's text


def test_tool_lookup_partial_match_takes_closest_name():
    svc = _service(['Ribosome biogenesis in eukaryotes', 'Ribosome'], [])
    assert svc._fuzzy_match_pathway('ribosomes') == 'ribosome'


def test_tool_lookup_normalizes_spacing_and_echoed_id():
    svc = _service(['Cell cycle'], ['Cell cycle'])
    assert svc._fuzzy_match_pathway('Cell  Cycle (hsa04110)') == 'cell cycle'


# ---- Step 4 coverage compares normalized names ----------------------------------------

FINAL = json.dumps({'hypotheses': [], 'centralMechanisticModel': '', 'keyPredictions': [],
                    'hypothesesSummary': ''})


def test_coverage_counts_a_pathway_queried_under_different_casing():
    step = _bare_step04()
    step.llm.chat.return_value = FINAL
    step.llm.chat_with_tools.side_effect = [
        {'content': None, 'finish_reason': 'tool_calls', 'tool_calls': [
            {'id': 'c1', 'type': 'function', 'function': {
                'name': 'get_pathway_mechanism',
                'arguments': json.dumps({'pathway_name': 'butanoate metabolism'})}}]},
        {'content': FINAL, 'tool_calls': None, 'finish_reason': 'stop'},
    ]
    mechanisms = {'pathway_mechanisms': [{'pathway': 'BUTANOATE  Metabolism'}],
                  'pathway_overlaps': [],
                  'pathway_structures': _structures_only_step3()['pathway_structures']}
    result = step.execute(genes=[], pathways=[], themes=None, hub_genes_result=None,
                          mechanisms_result=mechanisms, analyses=None, context=None)
    log = result.metadata['tool_call_log']
    assert log['nudged'] is False
    assert log['filled_by_engine'] == [] and log['missing_after_loop'] == []


def test_table_pathways_reset_when_a_run_has_no_structures():
    step = _bare_step04()
    step._table_pathways = ['Leftover from an earlier run']
    step._build_user_prompt(genes=[], pathways=[], themes=None, hub_genes_result=None,
                            mechanisms_result={'pathway_mechanisms': [], 'pathway_structures': []},
                            experiment_context='')
    assert step._table_pathways == []


# ---- chat() flattens tool history -------------------------------------------------------

def _client(provider):
    client = LLMClient.__new__(LLMClient)
    client.provider, client.model, client.temperature = provider, 'm', 0.0
    client.max_tokens, client.reasoning_effort, client.api_key = 100, None, None
    sent = {}

    def fake_post(payload, headers, **kwargs):
        sent['messages'] = payload['messages']
        return 'ok'
    client._post = fake_post
    return client, sent


TOOL_HISTORY = [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'c1', 'type': 'function',
         'function': {'name': 'get_pathway_mechanism', 'arguments': '{"pathway_name": "X"}'}}]},
    {'role': 'tool', 'tool_call_id': 'c1', 'content': '{"pathway": "X"}'},
    {'role': 'user', 'content': 'final please'},
]


def test_chat_flattens_tool_history_for_every_provider():
    for provider in ('local', 'groq'):
        client, sent = _client(provider)
        assert client.chat([dict(m) for m in TOOL_HISTORY]) == 'ok'
        roles = [m['role'] for m in sent['messages']]
        assert 'tool' not in roles
        assert not any(m.get('tool_calls') for m in sent['messages'])
        assert '[Called tool: get_pathway_mechanism' in sent['messages'][1]['content']
        assert '[Result from get_pathway_mechanism]' in sent['messages'][2]['content']


def test_chat_without_tool_history_is_unchanged():
    client, sent = _client('local')
    msgs = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'}]
    client.chat(msgs)
    assert sent['messages'] is msgs


# ---- Step 3 ----------------------------------------------------------------------------

def _kegg_struct(name, *genes):
    return {'pathway': name, 'source': 'kegg', 'confidence': 'high', 'mapped_de_genes': [],
            'structure': SimpleNamespace(genes=[SimpleNamespace(gene_symbols=[g], gene_symbol=g)
                                                for g in genes])}


def test_crosstalk_keeps_line_whose_partner_name_contains_a_gene_symbol():
    structures = [_kegg_struct('Apoptosis', 'NFKB1'), _kegg_struct('TNF signaling pathway', 'TNF', 'NFKB1')]
    overlaps = [SimpleNamespace(pathway1='Apoptosis', pathway2='TNF signaling pathway', shared_genes=['NFKB1'])]
    mech = {'pathway': 'Apoptosis', 'crosstalk': ['Apoptosis shares NFKB1 with TNF signaling pathway.']}
    counts = check_crosstalk([mech], overlaps, structures, [{'geneSymbol': 'NFKB1'}])
    assert mech['crosstalk'] == ['Apoptosis shares NFKB1 with TNF signaling pathway.']
    assert counts['lines_dropped'] == 0


def test_crosstalk_still_drops_a_wrong_gene_outside_pathway_names():
    structures = [_kegg_struct('Apoptosis', 'NFKB1'), _kegg_struct('TNF signaling pathway', 'TNF', 'NFKB1')]
    overlaps = [SimpleNamespace(pathway1='Apoptosis', pathway2='TNF signaling pathway', shared_genes=['NFKB1'])]
    mech = {'pathway': 'Apoptosis', 'crosstalk': ['TNF is the shared gene with TNF signaling pathway.']}
    check_crosstalk([mech], overlaps, structures, [{'geneSymbol': 'NFKB1'}])
    assert mech['crosstalk'] == []


def test_pathway_overlaps_ignore_symbol_case():
    step = Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)

    def gene(sym):
        return SimpleNamespace(gene_symbol=sym, direction='up')
    overlaps = step._calculate_pathway_overlaps(
        [{'pathway': 'Cell cycle', 'mapped_de_genes': [gene('Mcm2'), gene('Cdk1')]},
         {'pathway': 'Replication set', 'mapped_de_genes': [gene('MCM2')]}],
        hub_genes=['mcm2'])
    assert len(overlaps) == 1
    assert overlaps[0].shared_genes == ['Mcm2'] and overlaps[0].shared_hub_genes == ['Mcm2']


def test_kegg_miss_fallback_filters_llm_relations_by_input_gene():
    step = Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)
    step.step_number = 3
    step.membership_service = None
    step.kegg_service = MagicMock()
    step.kegg_service.get_pathway_structure.return_value = None  # KEGG miss
    step._detect_pathway_source = lambda pathway: 'kegg'
    rels = [SimpleNamespace(source='TFAM', target='POLRMT', subtype='inferred'),
            SimpleNamespace(source='POLRMT', target='SSBP1', subtype='inferred')]
    step._build_inferred_structure = lambda pathway, genes, organism: (
        SimpleNamespace(id='x'), [], list(rels), 'inferred')
    out = step._get_pathway_structures([{'name': 'Some KEGG map'}], [{'geneSymbol': 'Tfam'}], 'Mus musculus')
    assert [r.source for r in out[0]['de_relations']] == ['TFAM']
    assert out[0]['source'] == 'other'


def test_gmt_for_another_organism_is_not_used(tmp_path):
    (tmp_path / 'Mouse.Mito.gmt').write_text(
        'OXPHOS\tOXPHOS\tNdufs1\tNdufs2\nComplex I\tComplex I\tNdufs1\nTCA cycle\tTCA cycle\tCs\n')
    svc = PathwayMembershipService(gmt_dir=tmp_path)
    pathways = [{'name': n, 'source': 'mito', 'genes': []} for n in ('OXPHOS', 'Complex I', 'TCA cycle')]
    assert svc.associate_sources(pathways, 'Homo sapiens') == {}
    assert svc.members(pathways[0]) is None
    assert svc.associate_sources(pathways, 'Mus musculus') == {'mito': 'Mouse.Mito.gmt'}
    assert svc.members(pathways[0]) == ['Ndufs1', 'Ndufs2']


def test_gene_set_card_lists_every_gene_of_a_box():
    step = Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)
    step.step_number = 3
    step.membership_service = None
    box = PathwayEntry(id='1', name='hsa:4171 hsa:4172', type='gene',
                       names=['hsa:4171', 'hsa:4172'], gene_symbol='MCM2')
    box.gene_symbols = ['MCM2', 'MCM3']
    structure = SimpleNamespace(id='hsa04110', genes=[box], relations=[], compounds=[])
    step.kegg_service = MagicMock()
    step.kegg_service.get_pathway_structure.return_value = structure
    step.kegg_service.map_de_genes_to_pathway.return_value = []
    step.kegg_service.find_de_gene_relations.return_value = []
    step._detect_pathway_source = lambda pathway: 'kegg'
    out = step._get_pathway_structures([{'name': 'Cell cycle'}], [], 'Homo sapiens')
    assert [g.gene_symbol for g in out[0]['mapped_de_genes']] == ['MCM2', 'MCM3']
    assert out[0]['confidence'] == 'gene_set'
