"""
Tests for KEGGService pathway-ID resolution.

Regression context: KEGG's `find/pathway/<name>` REST operation dropped the historical
`path:` prefix and now returns a bare reference ID like `map00650`. The old code only
converted IDs that `startswith('path:map')` into the organism-specific form, so a bare
`map00650` flowed through unconverted and `get/map00650/kgml` returned HTTP 404 (KGML
exists only for `hsa00650`). That blanked Step 3 (Pathway Mechanisms) and cascaded to
zero hypotheses in Step 4.

These tests pin the fix.
"""

import importlib

import pytest
from unittest.mock import MagicMock, patch

KEGG_MODULES = ['src.pipeline.services.kegg_service']

# Minimal but valid KGML: 2 gene entries + 1 relation. ACAT1 ends in a digit on purpose
# so the symbol assertions also guard against the trailing-digit-stripping bug.
KGML_FIXTURE = """<?xml version="1.0"?>
<pathway name="path:hsa00650" org="hsa" number="00650" title="Butanoate metabolism">
  <entry id="1" name="hsa:38" type="gene">
    <graphics name="ACAT1, ACAT, MAT, T2" />
  </entry>
  <entry id="2" name="hsa:1629" type="gene">
    <graphics name="DBT" />
  </entry>
  <relation entry1="1" entry2="2" type="ECrel">
    <subtype name="compound" value="C00024" />
  </relation>
</pathway>
"""


@pytest.fixture(params=KEGG_MODULES)
def kegg_mod(request):
    """The kegg_service module under test (one per copy)."""
    return importlib.import_module(request.param)


def _resp(ok=True, text=""):
    r = MagicMock()
    r.ok = ok
    r.text = text
    return r


def _make_fake_get(find_text, kgml_text=KGML_FIXTURE,
                   find_ok=True, kgml_ok=True, requested=None):
    """Build a fake requests.get that routes by URL and records requested URLs.

    `kgml_ok` may be a bool or a predicate `fn(url) -> bool`, letting a test make KGML
    available only for the organism-specific id (so the structure assertions actually
    catch a regression that fetches the reference `map*` id).
    """
    def _get(url, *args, **kwargs):
        if requested is not None:
            requested.append(url)
        if '/find/pathway/' in url:
            return _resp(ok=find_ok, text=find_text)
        if '/kgml' in url:
            ok = kgml_ok(url) if callable(kgml_ok) else kgml_ok
            return _resp(ok=ok, text=kgml_text if ok else "")
        return _resp(ok=False, text="")
    return _get


# --------------------------------------------------------------------------- #
# _find_pathway_id: organism-specific ID resolution across all observed shapes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("find_text, organism, expected", [
    # The regression: new prefix-less reference ID must convert to organism form.
    ("map00650\tButanoate metabolism", "Homo sapiens", "hsa00650"),
    # Back-compat: legacy 'path:map' form must still convert.
    ("path:map00650\tButanoate metabolism", "Homo sapiens", "hsa00650"),
    # Already organism-specific (bare) — pass through.
    ("hsa00650\tButanoate metabolism - Homo sapiens", "Homo sapiens", "hsa00650"),
    # Already organism-specific (prefixed) — strip prefix, pass through.
    ("path:hsa00650\tButanoate metabolism", "Homo sapiens", "hsa00650"),
    # Organism awareness: mouse maps onto 'mmu'.
    ("map00650\tButanoate metabolism", "Mus musculus", "mmu00650"),
])
def test_find_pathway_id_resolves_org_specific(kegg_mod, find_text, organism, expected):
    svc = kegg_mod.KEGGService()
    with patch.object(kegg_mod.requests, 'get',
                      side_effect=_make_fake_get(find_text)):
        result = svc._find_pathway_id("Butanoate metabolism", organism)
    assert result == expected


def test_find_pathway_id_prefers_org_specific_line(kegg_mod):
    """When find returns both a generic and an org-specific line, prefer org-specific."""
    svc = kegg_mod.KEGGService()
    find_text = "map00650\tButanoate metabolism\nhsa00650\tButanoate metabolism - human"
    with patch.object(kegg_mod.requests, 'get',
                      side_effect=_make_fake_get(find_text)):
        assert svc._find_pathway_id("Butanoate metabolism", "Homo sapiens") == "hsa00650"


def test_find_pathway_id_empty_response_returns_none(kegg_mod):
    svc = kegg_mod.KEGGService()
    with patch.object(kegg_mod.requests, 'get', side_effect=_make_fake_get("")):
        assert svc._find_pathway_id("Nonexistent pathway", "Homo sapiens") is None


def test_find_pathway_id_not_ok_returns_none(kegg_mod):
    svc = kegg_mod.KEGGService()
    with patch.object(kegg_mod.requests, 'get',
                      side_effect=_make_fake_get("map00650\tx", find_ok=False)):
        assert svc._find_pathway_id("Butanoate metabolism", "Homo sapiens") is None


def test_find_pathway_id_unknown_organism_returns_none_without_query(kegg_mod):
    """Unknown organism must NOT silently resolve to human — skip KEGG entirely.

    Guards the regression where a non-model organism (not in organism_map) defaulted to
    org_code 'hsa' and surfaced human curated data for a non-human study.
    """
    svc = kegg_mod.KEGGService()
    fake = MagicMock(side_effect=_make_fake_get("map00650\tButanoate metabolism"))
    with patch.object(kegg_mod.requests, 'get', fake):
        assert svc._find_pathway_id("Butanoate metabolism", "Sus scrofa") is None
    fake.assert_not_called()  # no network call for an unsupported organism


def test_find_pathway_id_unknown_organism_warns_only_once(kegg_mod, capsys):
    """The unsupported-organism warning is emitted once per organism, not per call."""
    svc = kegg_mod.KEGGService()
    fake = MagicMock(side_effect=_make_fake_get("map00650\tButanoate metabolism"))
    with patch.object(kegg_mod.requests, 'get', fake):
        assert svc._find_pathway_id("Butanoate metabolism", "Sus scrofa") is None
        assert svc._find_pathway_id("Propanoate metabolism", "Sus scrofa") is None
    fake.assert_not_called()  # no network for an unsupported organism
    warns = [ln for ln in capsys.readouterr().out.splitlines()
             if 'Organism not supported' in ln]
    assert len(warns) == 1   # cached after first; un-cached would warn twice


def test_find_pathway_id_caches_search(kegg_mod):
    """A repeated lookup must hit the cache, not re-query KEGG."""
    svc = kegg_mod.KEGGService()
    fake = MagicMock(side_effect=_make_fake_get("map00650\tButanoate metabolism"))
    with patch.object(kegg_mod.requests, 'get', fake):
        first = svc._find_pathway_id("Butanoate metabolism", "Homo sapiens")
        second = svc._find_pathway_id("Butanoate metabolism", "Homo sapiens")
    assert first == second == "hsa00650"
    assert fake.call_count == 1


# --------------------------------------------------------------------------- #
# _normalize_pathway_id: direct unit coverage of the conversion helper
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw, org_code, expected", [
    ("map00650", "hsa", "hsa00650"),
    ("path:map00650", "hsa", "hsa00650"),
    ("00650", "hsa", "hsa00650"),
    ("hsa00650", "hsa", "hsa00650"),
    ("path:hsa00650", "hsa", "hsa00650"),
    ("map00650", "mmu", "mmu00650"),
    ("map01100", "hsa", "hsa01100"),          # global/overview map
    # Non-pathway id: db prefix stripped, no numeric match -> returned as-is (then 404s).
    ("cpd:C00024", "hsa", "C00024"),
])
def test_normalize_pathway_id(kegg_mod, raw, org_code, expected):
    svc = kegg_mod.KEGGService()
    assert svc._normalize_pathway_id(raw, org_code) == expected


@pytest.mark.parametrize("raw, org_code, expected", [
    ("map0650", "hsa", "map0650"),      # 4 digits -> not a reference-pathway shape
    ("map001000", "hsa", "map001000"),  # 6 digits -> unchanged
    ("1234", "hsa", "1234"),            # bare 4-digit -> unchanged (loosened \\d+ would convert)
])
def test_normalize_pathway_id_requires_exactly_five_digits(kegg_mod, raw, org_code, expected):
    """KEGG pathway numbers are exactly 5 digits; only those convert to the org form."""
    svc = kegg_mod.KEGGService()
    assert svc._normalize_pathway_id(raw, org_code) == expected


# --------------------------------------------------------------------------- #
# _extract_gene_symbol: must preserve trailing digits (gene-symbol corruption bug)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("graphics_name, expected", [
    ("ACAT1, ACAT, MAT, T2", "ACAT1"),   # primary symbol is the first comma-token
    ("BDH1, BDH", "BDH1"),
    ("TP53, p53", "TP53"),
    ("COX4I2", "COX4I2"),                 # trailing digit must be kept
    ("NDUFA4L2", "NDUFA4L2"),
    ("HMGCS1, HMGCS", "HMGCS1"),
    ("DBT", "DBT"),                       # no trailing digit
    (None, None),
    ("", None),
])
def test_extract_gene_symbol_preserves_trailing_digits(kegg_mod, graphics_name, expected):
    svc = kegg_mod.KEGGService()
    assert svc._extract_gene_symbol(graphics_name) == expected


# --------------------------------------------------------------------------- #
# _get_kgml
# --------------------------------------------------------------------------- #

def test_get_kgml_404_returns_none(kegg_mod):
    svc = kegg_mod.KEGGService()
    with patch.object(kegg_mod.requests, 'get',
                      side_effect=_make_fake_get("", kgml_ok=False)):
        assert svc._get_kgml("map00650") is None


# --------------------------------------------------------------------------- #
# End-to-end wire-through: the resolved org-specific ID must be used for KGML
# --------------------------------------------------------------------------- #

def test_get_pathway_structure_uses_org_specific_kgml(kegg_mod):
    """find returns bare map00650 -> KGML MUST be fetched for hsa00650.

    KGML is made available ONLY for the hsa id, so a regression that fetched map00650
    would yield structure=None and fail here — the assertions are bug-sensitive.
    """
    svc = kegg_mod.KEGGService()
    requested = []
    fake = _make_fake_get(
        "map00650\tButanoate metabolism",
        kgml_ok=lambda u: '/get/hsa00650/kgml' in u,  # 404 for anything but hsa00650
        requested=requested,
    )
    with patch.object(kegg_mod.requests, 'get', side_effect=fake):
        structure = svc.get_pathway_structure("Butanoate metabolism", "Homo sapiens")

    # Structure parsed from the KGML fixture, with correct, distinct gene symbols
    # (ACAT1 keeps its trailing digit — guards the gene-symbol corruption bug).
    assert structure is not None
    assert {g.gene_symbol for g in structure.genes} == {"ACAT1", "DBT"}
    assert len(structure.relations) == 1
    rel = structure.relations[0]
    assert (rel.source, rel.target) == ("ACAT1", "DBT")

    # Regression guard: KGML was fetched for the organism-specific ID, never map*.
    kgml_urls = [u for u in requested if u.endswith('/kgml')]
    assert any('/get/hsa00650/kgml' in u for u in kgml_urls)
    assert not any('/get/map00650/kgml' in u for u in kgml_urls)


def test_get_pathway_structure_legacy_prefix_end_to_end(kegg_mod):
    """Legacy 'path:map00650' must also resolve to hsa00650 end-to-end."""
    svc = kegg_mod.KEGGService()
    requested = []
    fake = _make_fake_get(
        "path:map00650\tButanoate metabolism",
        kgml_ok=lambda u: '/get/hsa00650/kgml' in u,
        requested=requested,
    )
    with patch.object(kegg_mod.requests, 'get', side_effect=fake):
        structure = svc.get_pathway_structure("Butanoate metabolism", "Homo sapiens")
    assert structure is not None
    assert any('/get/hsa00650/kgml' in u for u in requested if u.endswith('/kgml'))


def test_get_pathway_structure_none_when_kgml_404(kegg_mod):
    """If KGML 404s even after resolving the org id, return None (caller falls back)."""
    svc = kegg_mod.KEGGService()
    requested = []
    fake = _make_fake_get(
        "map00650\tButanoate metabolism",
        kgml_ok=False,  # every KGML fetch 404s
        requested=requested,
    )
    with patch.object(kegg_mod.requests, 'get', side_effect=fake):
        assert svc.get_pathway_structure("Butanoate metabolism", "Homo sapiens") is None
    # Even on failure, the id was resolved correctly before the KGML attempt.
    assert any('/get/hsa00650/kgml' in u for u in requested)
