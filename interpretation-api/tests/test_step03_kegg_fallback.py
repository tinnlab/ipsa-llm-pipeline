"""
Tests for Step 3 resilience: a KEGG/KGML miss must NOT blank the mechanisms section.

When `kegg_service.get_pathway_structure` returns None (e.g. KGML unavailable), Step 3
should fall back to the Pathway Commons/LLM route for that pathway instead of silently
dropping it, and emit a dev-only provenance log so a developer can tell at a glance
whether a mechanism came from curated KGML or the fallback.
"""

import json
import types

from unittest.mock import MagicMock, patch

import src.pipeline.services.kegg_service as kegg_mod
from src.pipeline.services.kegg_service import KEGGService
from src.pipeline.steps.step03_pathway_mechanisms import Step03PathwayMechanisms


LLM_STRUCTURE_JSON = json.dumps({
    "biologicalFunction": "Butanoate metabolism produces short-chain fatty acids.",
    "inferredRelations": [
        {"source": "ACAT1", "target": "ACAT2", "type": "regulation",
         "confidence": "medium", "rationale": "Both act in acetyl-CoA handling."}
    ],
    "deGeneRoles": [
        {"gene": "ACAT1", "foldChange": -1.2, "inferredRole": "thiolase"}
    ],
    "functionalConsequences": "Reduced SCFA catabolism.",
})


def _bare_step():
    """Construct Step03 without running its network-touching __init__."""
    step = Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)
    step.step_number = 3
    step.step_name = 'Pathway Mechanisms and Interactions'
    step.llm = MagicMock()
    step.kegg_service = MagicMock()
    step.pc_service = MagicMock()
    return step


def _kegg_pathway():
    return {
        'name': 'Butanoate metabolism',
        'source': 'KEGG',
        'pathwayId': 'hsa00650',
        'genes': ['ACAT1', 'ACAT2'],
        'ES': -1.9,
        'pValue': 1e-3,
        'pValueFDR': 1.75e-3,
    }


def _de_genes():
    return [
        {'geneSymbol': 'ACAT1', 'foldChange': -1.2, 'pValue': 0.01},
        {'geneSymbol': 'ACAT2', 'foldChange': -0.9, 'pValue': 0.02},
    ]


def test_kegg_miss_falls_back_to_pc_llm(capsys):
    """KEGG returns None -> pathway is kept via PC/LLM fallback, not dropped."""
    step = _bare_step()
    # Simulate a KGML miss for every KEGG pathway.
    step.kegg_service.get_pathway_structure.return_value = None
    # No curated PC interactions -> fallback relies on the LLM (confidence 'inferred').
    step.pc_service.get_interactions_between.return_value = []
    step.llm.chat.return_value = LLM_STRUCTURE_JSON

    structures = step._get_pathway_structures([_kegg_pathway()], _de_genes(), 'Homo sapiens')

    # Pathway survived via fallback rather than being dropped.
    assert len(structures) == 1
    entry = structures[0]
    assert entry['source'] == 'other'          # provenance: not curated KEGG
    assert entry['confidence'] in {'pc_grounded', 'inferred', 'gene_set'}
    assert entry['confidence'] == 'inferred'    # DE genes present, no PC data

    # Dev provenance logs are emitted.
    out = capsys.readouterr().out
    assert '[KEGG-MISS]' in out
    assert 'Mechanism provenance:' in out
    assert 'curated_KGML=0' in out
    assert 'fallback_PC/LLM=1' in out


def test_curated_kgml_path_still_works(capsys):
    """When KEGG resolves, the pathway is curated (source=kegg, confidence=high)."""
    step = _bare_step()

    fake_structure = MagicMock()
    fake_structure.id = 'hsa00650'
    fake_structure.genes = [MagicMock(), MagicMock()]
    step.kegg_service.get_pathway_structure.return_value = fake_structure
    step.kegg_service.map_de_genes_to_pathway.return_value = [
        types.SimpleNamespace(gene_symbol='ACAT1'),
        types.SimpleNamespace(gene_symbol='ACAT2'),
    ]
    step.kegg_service.find_de_gene_relations.return_value = []

    structures = step._get_pathway_structures([_kegg_pathway()], _de_genes(), 'Homo sapiens')

    assert len(structures) == 1
    entry = structures[0]
    assert entry['source'] == 'kegg'
    assert entry['confidence'] == 'high'

    # The LLM/PC fallback was never invoked on the happy path.
    step.llm.chat.assert_not_called()
    step.pc_service.get_interactions_between.assert_not_called()

    out = capsys.readouterr().out
    assert 'Mechanism provenance:' in out
    assert 'curated_KGML=1' in out
    assert 'fallback_PC/LLM=0' in out


def test_native_other_pathway_unaffected(capsys):
    """A natively non-KEGG pathway still goes straight through the PC/LLM route."""
    step = _bare_step()
    step.pc_service.get_interactions_between.return_value = []
    step.llm.chat.return_value = LLM_STRUCTURE_JSON

    pathway = {
        'name': 'Some Reactome Pathway',
        'source': 'Reactome',
        'genes': ['ACAT1', 'ACAT2'],
        'ES': 1.1,
    }
    structures = step._get_pathway_structures([pathway], _de_genes(), 'Homo sapiens')

    assert len(structures) == 1
    assert structures[0]['source'] == 'other'
    # KEGG service is never consulted for a non-KEGG pathway.
    step.kegg_service.get_pathway_structure.assert_not_called()
    out = capsys.readouterr().out
    assert '[KEGG-MISS]' not in out


def test_kgml_404_integration_falls_back(capsys):
    """End-to-end seam: a REAL KEGGService whose KGML 404s must trigger step03 fallback.

    Unlike test_kegg_miss_falls_back_to_pc_llm (which mocks kegg_service to return None),
    this wires the real KEGGService into step03 with only the network mocked, proving the
    full chain: find -> resolve hsa00650 -> KGML 404 -> get_pathway_structure None ->
    step03 PC/LLM fallback. This is exactly the boundary the two fixes meet at.
    """
    step = _bare_step()
    step.kegg_service = KEGGService()  # real service, real _find_pathway_id/_normalize_pathway_id
    step.pc_service.get_interactions_between.return_value = []
    step.llm.chat.return_value = LLM_STRUCTURE_JSON

    requested = []

    def fake_get(url, *args, **kwargs):
        requested.append(url)
        r = MagicMock()
        if '/find/pathway/' in url:
            r.ok, r.text = True, 'map00650\tButanoate metabolism'
        elif '/kgml' in url:
            r.ok, r.text = False, ''      # KGML 404 even after resolving hsa00650
        else:
            r.ok, r.text = False, ''
        return r

    with patch.object(kegg_mod.requests, 'get', side_effect=fake_get):
        structures = step._get_pathway_structures([_kegg_pathway()], _de_genes(), 'Homo sapiens')

    assert len(structures) == 1
    assert structures[0]['source'] == 'other'          # dropped KEGG -> PC/LLM fallback
    # Real service resolved the org-specific id before the KGML attempt.
    assert any('/get/hsa00650/kgml' in u for u in requested)
    out = capsys.readouterr().out
    assert '[KEGG-MISS]' in out
    assert 'Mechanism provenance:' in out
