"""
Step 03: Pathway Mechanisms and Interactions

This step interprets pathway structures from multiple sources (KEGG, Reactome, GO,
MitoCarta, MSigDB, custom) in the context of differentially expressed genes, using
LLM to provide biological interpretation.

Key features:
- Retrieves curated pathway structures from KEGG/Reactome (high confidence)
- Infers pathway mechanisms from non-curated sources using LLM knowledge (inferred)
- Maps DE genes to pathways
- Extracts regulatory relationships involving DE genes (curated or inferred)
- Uses LLM to interpret biological consequences
- Marks confidence level: 'high' for curated, 'inferred' for LLM-based
"""

import sys
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.groq_client import GroqClient
from src.pipeline.services.kegg_service import KEGGService, PathwayStructure, MappedDEGene, entry_symbols
from src.pipeline.services.reactome_service import ReactomeService
from src.pipeline.services.pathway_commons_service import PathwayCommonsService
from src.pipeline.services.pathway_membership_service import PathwayMembershipService
from src.pipeline.services.regulon_service import RegulonService
from src.pipeline.json_utils import parse_llm_json
from src.pipeline.fc_utils import (
    fc_direction, fc_arrow, has_fc, to_float, FC_EPSILON, sanitize_llm_text,
    gene_fc_lookup, NES_STRONG, NES_MODERATE,
)


def _enrichment_metric(pathway: Dict):
    """Return (value, metric) for a pathway's enrichment score.

    Prefer NES (normalized, unbounded) when present; fall back to the legacy `ES`
    field only when NES is absent. Upstream (the IPSA website) historically forwarded the
    NES value under the name `ES`, so callers that don't yet send `NES` still work.
    """
    nes = to_float(pathway.get('NES'))
    if nes is not None:
        return nes, 'NES'
    es = to_float(pathway.get('ES'))
    # Classic fgsea ES is bounded to [-1, 1]. A value beyond that magnitude can only be
    # an NES that upstream forwarded under the `ES` name (the historical mislabelling),
    # so treat it as NES to get the correct magnitude scale even before callers send NES.
    if es is not None and abs(es) > 1.0:
        return es, 'NES'
    return es, 'ES'


def _magnitude_label(value: float, metric: str = 'NES') -> str:
    """Magnitude descriptor calibrated to the score's scale.

    NES is unbounded (typically +/-1 to +/-4); classic ES is bounded [-1, 1].
    Using ES-scale thresholds on NES values made almost everything read 'strong'.
    """
    a = abs(value)
    if metric == 'NES':
        return 'strong' if a >= NES_STRONG else 'moderate' if a >= NES_MODERATE else 'weak'
    return 'strong' if a > 0.6 else 'moderate' if a > 0.4 else 'weak'


# A trailing KEGG/Reactome pathway-id parenthetical the LLM may fold into the pathway
# name it echoes back. The interpretation prompt shows "**Pathway: <name>** (KEGG: <id>)"
# (see _format_pathway_structure_for_llm), and the model sometimes emits the name as
# "Cell cycle (hsa04110)". Left in place, that suffix breaks name-based matching against
# the retrieved structures — causing a duplicate backfilled stub, an inconsistent
# confidence badge, and a missed de-dup for the very same pathway. Matching on the id
# *format* keeps legitimate parentheticals (e.g. "... (GPI)-anchor biosynthesis") intact.
_PATHWAY_ID_SUFFIX_RE = re.compile(
    r'\s*\(\s*(?:KEGG:\s*|REACTOME:\s*)?'
    r'(?:[a-z]{2,4}\d{4,6}|R-[A-Z]{3}-\d+)'
    r'\s*\)\s*$',
    re.IGNORECASE,
)


def _strip_pathway_id_suffix(name: str) -> str:
    """Remove a trailing KEGG/Reactome id parenthetical from a pathway name."""
    if not name:
        return ''
    return _PATHWAY_ID_SUFFIX_RE.sub('', name).strip()


def _pathway_key(name: str) -> str:
    """Normalized key for matching a pathway name across mechanisms and structures.

    Strips an echoed KEGG/Reactome id suffix, then lowercases/whitespace-normalizes so
    "Cell cycle (hsa04110)" and "Cell cycle" resolve to the same pathway.
    """
    return _strip_pathway_id_suffix(name).strip().lower()


# KEGG relation subtypes with no direction ("---", "-+-", shared compound). For these a
# listed relation matches in either orientation; every other KEGG subtype is directed
# and must match source -> target (item 16).
_UNDIRECTED_KEGG_SUBTYPES = {
    '', 'binding/association', 'dissociation', 'compound', 'missing interaction',
    'state change',
}


def _sym(value: Any) -> str:
    """Upper-case, trimmed gene symbol for comparisons (casing differs across sources:
    KGML human symbols, mouse symbols, uppercase Pathway Commons rows)."""
    return str(value or '').strip().upper()


def _relation_reference(ps: Dict) -> Optional[Tuple[str, set, set]]:
    """The retrieved relation set for one pathway, or None when nothing was retrieved.

    Returns (kind, directed_pairs, undirected_pairs), with symbols upper-cased.
    - KEGG pathway with a KGML structure: its DE relations (main L236, L240). A KGML
      pathway always has a retrieved set, even if it holds no DE relation.
    - Pathway with Pathway Commons interactions: those interactions, in either direction.
    - Anything else (inferred, no retrieval): None. Its relations are kept (decision D3).
    """
    if ps.get('source') == 'kegg' and ps.get('confidence') in ('high', 'gene_set'):
        directed, undirected = set(), set()
        for rel in ps.get('de_relations', []) or []:
            src = _sym(getattr(rel, 'source', None) if not isinstance(rel, dict) else rel.get('source'))
            tgt = _sym(getattr(rel, 'target', None) if not isinstance(rel, dict) else rel.get('target'))
            if not src or not tgt:
                continue
            subtype = (getattr(rel, 'subtype', None) if not isinstance(rel, dict)
                       else rel.get('subtype')) or ''
            if str(subtype).strip().lower() in _UNDIRECTED_KEGG_SUBTYPES:
                undirected.add((src, tgt))
            else:
                directed.add((src, tgt))
        return 'kegg', directed, undirected
    pc = ps.get('pc_interactions') or []
    if pc:
        undirected = set()
        for inter in pc:
            src = _sym(getattr(inter, 'source', None) if not isinstance(inter, dict) else inter.get('source'))
            tgt = _sym(getattr(inter, 'target', None) if not isinstance(inter, dict) else inter.get('target'))
            if src and tgt:
                undirected.add((src, tgt))
        return 'pathway_commons', set(), undirected
    return None


def _relation_in_reference(source: Any, target: Any, ref: Tuple[str, set, set]) -> bool:
    """True if the (source, target) pair is among the retrieved relations."""
    _, directed, undirected = ref
    s, t = _sym(source), _sym(target)
    return ((s, t) in directed or (s, t) in undirected or (t, s) in undirected)


def _rel_end(rel: Any, attr: str) -> Any:
    return rel.get(attr) if isinstance(rel, dict) else getattr(rel, attr, None)


def _input_gene_set(genes: Optional[List[Dict]]) -> set:
    """Upper-case symbols of the analysis's input gene list."""
    out = {_sym(g.get('geneSymbol') or g.get('gene')) for g in genes or [] if isinstance(g, dict)}
    out.discard('')
    return out


def filter_llm_relations(relations: List[Any], input_upper: set) -> Tuple[List[Any], List[Any]]:
    """LLM-proposed relations (pathways with nothing retrieved, D3 keep): keep a relation
    only if at least one end is an input gene (owner decision 2026-11-02). Comparison is
    upper-case on both sides. Returns (kept, dropped). With no input genes there is
    nothing to check against, so every relation is kept."""
    if not input_upper:
        return list(relations or []), []
    kept, dropped = [], []
    for rel in relations or []:
        if _sym(_rel_end(rel, 'source')) in input_upper or _sym(_rel_end(rel, 'target')) in input_upper:
            kept.append(rel)
        else:
            dropped.append(rel)
    return kept, dropped


# ---------------------------------------------------------------------------
# Pathway Commons relation extraction for non-KEGG pathways (item 19, main L236/L240)
# ---------------------------------------------------------------------------

# Directed, mechanistic SIF types first; complex membership last. Ranking decides which
# interactions survive PC_INTERACTION_CAP.
_PC_TYPE_RANK = {
    'controls-state-change-of': 0, 'controls-expression-of': 0,
    'controls-phosphorylation-of': 0, 'controls-transport-of': 1,
    'controls-production-of': 1, 'consumption-controlled-by': 1,
    'catalysis-precedes': 1, 'used-to-produce': 2, 'reacts-with': 2,
    'in-complex-with': 3,
}
# Max Pathway Commons interactions kept per pathway (after the DE and membership
# filters). A bound is needed because a full-membership pathsbetween query can return
# thousands of rows (MitoPathways "Metabolism": 3,226 mechanistic rows).
PC_INTERACTION_CAP = 60


@dataclass
class PCRelation:
    """A retrieved Pathway Commons interaction used as a pathway relation. Symbols are
    stored upper-case; display_names maps them back to the input casing."""
    source: str
    target: str
    type: str
    subtype: str


@dataclass
class PCStructure:
    """Structure placeholder for a pathway grounded on Pathway Commons interactions."""
    id: str
    name: str
    organism: str


def filter_pc_interactions(interactions, members_upper: set, de_upper: set,
                           cap: int = PC_INTERACTION_CAP):
    """Keep interactions with a DE gene at one end and both ends in the pathway
    membership, rank them (directed types first, both-ends-DE first), then cap.

    Returns (kept, stats). Symbols in `kept` are upper-case PCRelation objects.
    Far-end partners outside the queried membership are dropped rather than given a
    guessed display name (pathsbetween returns intermediates outside the query).
    """
    stats = {'retrieved': 0, 'no_de_end': 0, 'outside_membership': 0, 'capped': 0}
    seen, candidates = set(), []
    for inter in interactions or []:
        stats['retrieved'] += 1
        s = _sym(getattr(inter, 'source', None))
        t = _sym(getattr(inter, 'target', None))
        itype = getattr(inter, 'interaction_type', '') or ''
        if not s or not t or s == t:
            continue
        if s not in members_upper or t not in members_upper:
            stats['outside_membership'] += 1
            continue
        if s not in de_upper and t not in de_upper:
            stats['no_de_end'] += 1
            continue
        key = (s, t, itype)
        if key in seen:
            continue
        seen.add(key)
        both_de = s in de_upper and t in de_upper
        candidates.append(((_PC_TYPE_RANK.get(itype, 2), 0 if both_de else 1, s, t, itype),
                           PCRelation(source=s, target=t, type='pathway_commons', subtype=itype)))
    candidates.sort(key=lambda c: c[0])
    kept = [rel for _, rel in candidates[:cap]]
    stats['capped'] = max(0, len(candidates) - cap)
    stats['kept'] = len(kept)
    return kept, stats


# Item 11/20 (main L240): remove the description of every gene that is not a DE gene and
# not in a retrieved relation. Switched on by the owner (2026-11-02) after reading the
# measured flags; the flags are recorded on each card (geneCheckFlags) either way.
# On cards whose pathway has mapped DE genes, item 21 already drops every gene this
# would remove; it matters for cards with no mapped DE gene (gene-set cards).
# STEP3_GENE_CHECK_ENABLED = False  # shipped off pending the owner's decision
STEP3_GENE_CHECK_ENABLED = True


def _retrieved_endpoints(ps: Optional[Dict]) -> set:
    """Upper-case endpoints of the relations retrieved for a pathway (KGML / PC)."""
    ref = _relation_reference(ps) if ps else None
    if ref is None:
        return set()
    _, directed, undirected = ref
    return {g for pair in directed | undirected for g in pair}


def normalize_card_genes(mechanisms: List[Dict], pathway_structures: List[Dict],
                         genes: List[Dict], gene_check_enabled: Optional[bool] = None) -> Dict:
    """Make each Step 3 card's gene list match the engine's data, in place.

    1. Item 11/20 gene check (main L240): a listed gene that is neither a DE gene nor an
       endpoint of a relation retrieved for that pathway is flagged (geneCheckFlags) and,
       when enabled, removed. Runs inside Step 3, so it reaches the printed card, and
       does not depend on the orchestrator's enable_validation flag.
    2. Item 21 (main L263, L269): the card's DE genes and fold changes are the engine's
       mapped DE genes of that pathway. The model's role text is kept for genes it named;
       model-listed genes not mapped to the pathway are dropped (droppedGenes).

    Returns counts for logging/measurement.
    """
    if gene_check_enabled is None:
        gene_check_enabled = STEP3_GENE_CHECK_ENABLED
    de_upper = {_sym(g.get('geneSymbol') or g.get('gene')) for g in genes or []}
    de_upper.discard('')
    by_key = {_pathway_key(ps.get('pathway', '')): ps for ps in pathway_structures or []}
    counts = {'flagged': 0, 'removed_by_gene_check': 0, 'dropped_unmapped': 0,
              'added_mapped': 0, 'fc_replaced': 0}

    for mech in mechanisms or []:
        ps = by_key.get(_pathway_key(mech.get('pathway', '')))
        listed = [g for g in (mech.get('deGeneInvolvement') or []) if isinstance(g, dict)]

        # 1) gene check (main L240)
        allowed = de_upper | _retrieved_endpoints(ps)
        flags = [g.get('gene') for g in listed if g.get('gene') and _sym(g.get('gene')) not in allowed]
        if flags:
            mech['geneCheckFlags'] = flags
            counts['flagged'] += len(flags)
            if gene_check_enabled:
                flagged = {_sym(f) for f in flags}
                listed = [g for g in listed if _sym(g.get('gene')) not in flagged]
                counts['removed_by_gene_check'] += len(flags)
                mech['deGeneInvolvement'] = listed

        # 2) engine genes and fold changes (item 21)
        if ps is None:
            continue
        mapped = [g for g in ps.get('mapped_de_genes', []) or []
                  if getattr(g, 'gene_symbol', '') and getattr(g, 'direction', '') in ('up', 'down')]
        if not mapped:
            continue  # no DE data for this pathway: nothing computed to print
        mapped_by_sym = {}
        for g in mapped:
            mapped_by_sym.setdefault(_sym(g.gene_symbol), g)
        roles, order = {}, []
        for g in listed:
            s = _sym(g.get('gene'))
            if s in mapped_by_sym and s not in roles:
                roles[s] = g.get('roleInPathway', '') or ''
                order.append(s)
                fc = to_float(g.get('foldChange'))
                if fc is None or round(fc, 2) != round(mapped_by_sym[s].fold_change, 2):
                    counts['fc_replaced'] += 1
        dropped = [g.get('gene') for g in listed
                   if g.get('gene') and _sym(g.get('gene')) not in mapped_by_sym]
        rest = sorted((s for s in mapped_by_sym if s not in roles),
                      key=lambda s: -abs(mapped_by_sym[s].fold_change or 0))
        counts['added_mapped'] += len(rest)
        mech['deGeneInvolvement'] = [
            {'gene': mapped_by_sym[s].gene_symbol, 'foldChange': mapped_by_sym[s].fold_change,
             'roleInPathway': roles.get(s, '')}
            for s in order + rest
        ]
        if dropped:
            mech['droppedGenes'] = dropped
            counts['dropped_unmapped'] += len(dropped)
    return counts


_GENE_TOKEN = re.compile(r'\b[A-Za-z][A-Za-z0-9\-]{1,14}\b')


def _listed_only(text: str, gene: str) -> bool:
    """True if every occurrence of `gene` is an item of a list or parenthetical, with list
    punctuation on both sides ("(A, B)", "e.g., A, B)", "A/B/C"), so removing it leaves a
    grammatical sentence. A clause comma ("however, A's ...") does not count."""
    found = False
    for m in re.finditer(rf'(?<![\w-]){re.escape(gene)}(?![\w-])', text):
        found = True
        before = text[:m.start()].rstrip()[-1:]
        after = text[m.end():].lstrip()[:1]
        if before not in ('(', ',', '/') or after not in (',', ')', '/'):
            return False
    return found


def _remove_gene_tokens(text: str, genes: set) -> str:
    """Remove named gene tokens from crosstalk text and tidy the separators left behind."""
    for g in genes:
        text = re.sub(rf'(?<![\w-]){re.escape(g)}(?![\w-])', '', text)
    text = re.sub(r'\(\s*(?:e\.g\.,?\s*)?[,;/\s]*\)', '', text)       # "( , )" / "(e.g., )"
    text = re.sub(r'\s*,\s*(?=[,)])', '', text)                      # ", ," / ", )"
    text = re.sub(r'\(\s*,\s*', '(', text)                           # "( ,"
    text = re.sub(r'\s+and\s*(?=[),.;])', '', text)                  # "X and )"
    text = re.sub(r'\(\s*and\s+', '(', text)                         # "( and X"
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s+([,.;)])', r'\1', text)
    return text.strip()


def check_crosstalk(mechanisms: List[Dict], pathway_overlaps, pathway_structures: List[Dict],
                    genes: List[Dict]) -> Dict:
    """Keep the crosstalk text consistent with the computed DE-gene overlaps (item 22).

    For each crosstalk line of a card, the partner pathways it names (by name) and the
    gene symbols it names are found. A named gene that is not in the computed overlap of
    this pathway with a named partner (or, if no partner is named, with any partner) is
    removed when it is an item of a list, and drops the whole line when it is part of
    the sentence (deleting it there would leave broken prose). A line naming only
    partners with which no DE gene is shared is dropped. Counts written in the text are
    not changed: the partner a count refers to cannot be told reliably. Changes are
    recorded on the card (crosstalkCorrections).
    """
    shared = {}
    for o in pathway_overlaps or []:
        a, b = _pathway_key(o.pathway1), _pathway_key(o.pathway2)
        genes_upper = {_sym(g) for g in o.shared_genes}
        shared[(a, b)] = shared[(b, a)] = genes_upper
    names = {_pathway_key(ps.get('pathway', '')): ps.get('pathway', '') for ps in pathway_structures or []}
    # Gene symbols as written in the inputs. Tokens are matched with their exact casing so
    # English words ("set", "max", "cat") are never read as genes (SET, MAX, CAT).
    vocabulary = {str(g.get('geneSymbol') or g.get('gene') or '').strip() for g in genes or []}
    for ps in pathway_structures or []:
        for g in ps.get('mapped_de_genes', []) or []:
            vocabulary.add(str(getattr(g, 'gene_symbol', '') or '').strip())
        structure = ps.get('structure')
        for entry in getattr(structure, 'genes', []) or []:
            for s in getattr(entry, 'gene_symbols', None) or [getattr(entry, 'gene_symbol', None)]:
                vocabulary.add(str(s or '').strip())
    vocabulary.discard('')
    counts = {'lines': 0, 'genes_removed': 0, 'lines_dropped': 0}

    for mech in mechanisms or []:
        me = _pathway_key(mech.get('pathway', ''))
        lines = mech.get('crosstalk')
        if not isinstance(lines, list):
            continue
        kept, corrections = [], []
        for line in lines:
            if not isinstance(line, str):
                kept.append(line)
                continue
            counts['lines'] += 1
            low = line.lower()
            partners = [k for k, nm in names.items()
                        if k != me and nm and nm.lower() in low]
            # Prefer the longest names (e.g. "Ribosome biogenesis in eukaryotes" over "Ribosome").
            partners = [k for k in partners
                        if not any(k != o and names[k].lower() in names[o].lower() for o in partners)]
            if partners and all(not shared.get((me, p)) for p in partners):
                corrections.append({'line': line, 'action': 'dropped',
                                    'reason': 'no DE gene shared with the named pathway(s)'})
                counts['lines_dropped'] += 1
                continue
            allowed = set()
            for p in partners or [k for k in names if k != me]:
                allowed |= shared.get((me, p), set())
            # Pathway names are removed before looking for genes: "TNF signaling pathway"
            # names a pathway, not the gene TNF.
            # named = {t for t in _GENE_TOKEN.findall(line) if t in vocabulary}
            genes_text = line
            for nm in sorted((n for n in names.values() if n), key=len, reverse=True):
                genes_text = re.sub(re.escape(nm), ' ', genes_text, flags=re.IGNORECASE)
            named = {t for t in _GENE_TOKEN.findall(genes_text) if t in vocabulary}
            wrong = {t for t in named if _sym(t) not in allowed}
            new = line
            if wrong:
                # Remove in place only from lists; a wrong gene that is part of the
                # sentence itself ("DBT is the sole shared gene ...") drops the line.
                if not all(_listed_only(line, g) for g in wrong):
                    corrections.append({'line': line, 'action': 'dropped',
                                        'reason': 'names a gene outside the computed overlap',
                                        'removed_genes': sorted(wrong)})
                    counts['lines_dropped'] += 1
                    continue
                new = _remove_gene_tokens(new, wrong)
                counts['genes_removed'] += len(wrong)
            if new != line:
                corrections.append({'line': line, 'action': 'corrected', 'to': new,
                                    'removed_genes': sorted(wrong)})
            kept.append(new)
        mech['crosstalk'] = kept
        if corrections:
            mech['crosstalkCorrections'] = corrections
    return counts


def _display(symbol: str, display_names: Optional[Dict[str, str]]) -> str:
    """The input-cased name for an upper-case symbol (mouse names stay mouse-cased)."""
    if not display_names:
        return symbol
    return display_names.get(_sym(symbol), symbol)


def _canonicalize_mechanism_names(mechanisms: List[Dict],
                                  pathway_structures: List[Dict]) -> None:
    """Rewrite each mechanism's ``pathway`` to its retrieved structure name, in place.

    The LLM can echo a pathway name with the id suffix it was shown in the prompt; this
    maps that echo back to the canonical structure name so the downstream coverage
    check, confidence lookup, and de-duplication all key on the same string.
    """
    canon_by_key = {}
    for ps in pathway_structures:
        name = ps.get('pathway') if isinstance(ps, dict) else None
        if name:
            canon_by_key.setdefault(_pathway_key(name), name)
    for mech in mechanisms or []:
        canon = canon_by_key.get(_pathway_key(mech.get('pathway') or ''))
        if canon:
            mech['pathway'] = canon


_PATHWAY_REF_RE = re.compile(r'^\[?\s*P?\s*(\d+)\s*\]?$', re.IGNORECASE)


def assign_pathway_refs(pathway_structures: List[Dict]) -> None:
    """Give every retrieved pathway a short reference tag (P1, P2, ...), in place.

    The tag is shown with the pathway in the prompt and the model returns it with each
    card, so the engine, not the model's spelling of the name, decides which pathway a
    card belongs to. Tags are global (assigned before batching) so they stay unique.
    """
    for i, ps in enumerate(pathway_structures or [], 1):
        if isinstance(ps, dict) and not ps.get('ref'):
            ps['ref'] = f'P{i}'


def resolve_mechanism_pathways(mechanisms: List[Dict],
                               pathway_structures: List[Dict]) -> Dict:
    """Set each card's ``pathway`` to the retrieved pathway it belongs to, in place.

    Order of evidence:
    1. the returned ``pathwayRef`` tag -> that pathway's retrieved name (the card title
       is then always the engine's name);
    2. no or unknown tag -> the name the model wrote, matched exactly (case, whitespace
       and an echoed id suffix ignored), as before;
    3. a tag that disagrees with a name matching a *different* pathway exactly: the name
       is used (a full exact name is the stronger signal) and the conflict is counted.
    Cards matching nothing keep the model's name and are counted as unmatched.
    The ``pathwayRef`` field is removed from the card after use.
    """
    by_ref, by_key = {}, {}
    for ps in pathway_structures or []:
        if not isinstance(ps, dict) or not ps.get('pathway'):
            continue
        if ps.get('ref'):
            by_ref[str(ps['ref']).upper()] = ps['pathway']
        by_key.setdefault(_pathway_key(ps['pathway']), ps['pathway'])
    counts = {'by_ref': 0, 'by_name': 0, 'ref_name_conflict': 0, 'unmatched': 0}
    for mech in mechanisms or []:
        if not isinstance(mech, dict):
            continue
        raw_ref = mech.pop('pathwayRef', None)
        m = _PATHWAY_REF_RE.match(str(raw_ref or '').strip())
        ref_name = by_ref.get(f'P{int(m.group(1))}') if m else None
        name_match = by_key.get(_pathway_key(mech.get('pathway') or ''))
        if ref_name and name_match and name_match != ref_name:
            counts['ref_name_conflict'] += 1
            mech['pathway'] = name_match
        elif ref_name:
            counts['by_ref'] += 1
            mech['pathway'] = ref_name
        elif name_match:
            counts['by_name'] += 1
            mech['pathway'] = name_match
        else:
            counts['unmatched'] += 1
    return counts


def _relations_heading(source: str, confidence: str) -> str:
    """Report heading for a pathway's relation list, naming only the provenance it has.

    Item 18: high -> "<SOURCE> Curated" (only KEGG sets 'high', so "KEGG Curated"),
    pc_grounded -> "Pathway Commons", inferred -> "LLM-Proposed", anything else (gene-set
    only, unmatched) -> plain "Regulatory Relations" with no provenance claim.
    """
    if confidence == 'high':
        return f'**{(source or "").upper()} Curated Regulatory Relations:**'
    if confidence == 'pc_grounded':
        # return '**Pathway Commons Curated Interactions:**'  # item 18: owner-specified label
        return '**Pathway Commons Regulatory Relations:**'
    if confidence == 'inferred':
        # return '**Inferred Regulatory Relations (LLM-derived):**'  # item 18: owner-specified label
        return '**LLM-Proposed Regulatory Relations:**'
    return '**Regulatory Relations:**'


def _structures_missing_mechanisms(mechanisms: List[Dict],
                                   pathway_structures: List[Dict]) -> List[Dict]:
    """Structures with no mechanism entry, matched by id-stripped normalized name.

    Central to the double-render fix: an LLM echo like "Cell cycle (hsa04110)" must be
    recognized as covering the "Cell cycle" structure, or Part-C would re-add it as a
    duplicate bare stub. Keying both sides through :func:`_pathway_key` prevents that.
    """
    covered = {_pathway_key(m.get('pathway') or '') for m in mechanisms or []}
    return [ps for ps in pathway_structures
            if _pathway_key(ps.get('pathway') or '') not in covered]


def _mechanism_richness(mech: Dict) -> int:
    """Score how 'complete' a mechanism entry is, for de-duplication."""
    score = 0
    if (mech.get('biologicalFunction') or '').strip():
        score += 2
    if (mech.get('functionalConsequences') or '').strip():
        score += 2
    if mech.get('deGeneInvolvement'):
        score += 1
    if mech.get('curatedRelations'):
        score += 1
    if mech.get('crosstalk'):
        score += 1
    return score


def _dedupe_mechanisms(mechanisms: List[Dict]) -> List[Dict]:
    """Collapse duplicate pathway-mechanism entries by pathway name.

    Step 3 can produce two entries for the same pathway — a rich LLM-interpreted one
    and an empty synthesized/backfilled stub — which render as duplicate, contradictory
    sections. Keep the richest entry per normalized pathway name, preserving order.
    """
    best = {}
    order = []
    for mech in mechanisms or []:
        key = _pathway_key(mech.get('pathway') or '')
        if not key:
            order.append(id(mech))
            best[id(mech)] = mech
            continue
        if key not in best:
            best[key] = mech
            order.append(key)
        elif _mechanism_richness(mech) > _mechanism_richness(best[key]):
            best[key] = mech
    return [best[k] for k in order]


def _dedupe_de_genes(de_genes: List[Dict]) -> List[Dict]:
    """De-duplicate a deGeneInvolvement list by gene symbol (keep strongest |FC|)."""
    best = {}
    order = []
    for g in de_genes or []:
        sym = (g.get('gene') or '').strip().upper()
        if not sym:
            continue
        if sym not in best:
            best[sym] = g
            order.append(sym)
        else:
            # to_float guards against string/None foldChange from the LLM
            cur = abs(to_float(best[sym].get('foldChange')) or 0)
            new = abs(to_float(g.get('foldChange')) or 0)
            if new > cur:
                best[sym] = g
    return [best[s] for s in order]


@dataclass
class PathwayOverlap:
    """Represents overlap between two pathways"""
    pathway1: str
    pathway2: str
    shared_genes_count: int
    shared_genes: List[str]
    shared_hub_genes: List[str]


@dataclass
class PathwayMechanismResult:
    """Result of pathway mechanism analysis"""
    pathway_mechanisms: List[Dict[str, Any]]
    mechanistic_summary: str
    pathway_structures: List[Dict[str, Any]]
    pathway_overlaps: List[Dict[str, Any]]
    report_section: str
    # Candidate upstream regulators (TFs) of the down-regulated program, from TF→target
    # regulon (CollecTRI) enrichment. Surfaces driver TFs that are not themselves in the
    # DE list (so Step 2's DE-based PPI-hub scoring misses them). LLM-proposed fallback
    # when the regulon DB is unavailable / yields no significant hit.
    upstream_regulators: List[Dict[str, Any]] = field(default_factory=list)


class Step03PathwayMechanisms:
    """
    Step 03: Pathway Mechanisms and Interactions

    Analyzes pathway mechanisms using KEGG curated data and LLM interpretation.
    """

    def __init__(self):
        self.step_number = 3
        self.step_name = 'Pathway Mechanisms and Interactions'
        self.dependencies = [1, 2]  # Depends on Step 1 (themes) and Step 2 (hub genes)

        # Initialize services
        self.llm = GroqClient()
        self.kegg_service = KEGGService()
        self.reactome_service = ReactomeService()
        self.pc_service = PathwayCommonsService()
        # Full membership of non-KEGG pathways from bundled GMT files (item 19).
        self.membership_service = PathwayMembershipService()
        # TF->target regulon for upstream-regulator inference. Lazily tolerant of a
        # missing data file (available=False) so the pipeline degrades to an LLM fallback.
        self.regulon_service = RegulonService()

    def execute(
        self,
        pathways: List[Dict],
        genes: List[Dict],
        analyses: List[Dict],
        themes: Optional[List[Dict]] = None,
        hub_genes: Optional[List[str]] = None
    ) -> PathwayMechanismResult:
        """
        Execute pathway mechanism analysis

        Args:
            pathways: List of enriched pathways from previous analysis
            genes: List of differentially expressed genes (can be empty)
            analyses: Analysis metadata (including organism)
            themes: Optional pathway themes from Step 1 for biological context
            hub_genes: Optional list of hub genes from Step 2 for network analysis

        Returns:
            PathwayMechanismResult with mechanisms, structures, and report
        """
        print(f'\n[Step {self.step_number}] {self.step_name}')

        # Get organism
        organism = analyses[0].get('organismId', 'Homo sapiens') if analyses else 'Homo sapiens'
        print(f'  Organism: {organism}')

        # Check if genes are provided
        has_genes = genes and len(genes) > 0
        if not has_genes:
            print(f'  Note: No DE genes provided - pathway mechanisms will be analyzed without gene mapping')

        # Use pathways already filtered and prioritized by Step 1
        # (orchestrator passes all filtered pathways: themed first, then ungrouped)
        # Pathways are already sorted by the orchestrator (theme significance + FDR)
        sorted_pathways = pathways

        print(f'  Analyzing {len(sorted_pathways)} pathways from Step 1 (tissue-filtered)')
        print(f'  Note: Pathways prioritized by theme significance, then FDR')

        # Get curated pathway structures from KEGG or Reactome
        print('\n  Retrieving curated pathway structures...')
        pathway_structures = self._get_pathway_structures(
            sorted_pathways,
            genes,
            organism
        )

        print(f'  Retrieved {len(pathway_structures)} pathway structures')
        # The engine, not the model, names each card: every pathway gets a tag the model
        # returns with its card (resolve_mechanism_pathways).
        assign_pathway_refs(pathway_structures)

        # Calculate pathway overlaps using actual mapped DE genes
        print('\n  Calculating pathway overlaps from mapped DE genes...')
        pathway_overlaps = self._calculate_pathway_overlaps(
            pathway_structures,
            hub_genes or []
        )

        if pathway_overlaps:
            print(f'  Found {len(pathway_overlaps)} pathway overlaps')
            for overlap in pathway_overlaps[:5]:
                print(f'    - {overlap.pathway1} ↔ {overlap.pathway2}: {overlap.shared_genes_count} shared genes')
        else:
            print('  No significant pathway overlaps detected')

        if not pathway_structures:
            print('  ⚠️  No pathway structures retrieved. Cannot proceed with mechanism analysis.')
            return PathwayMechanismResult(
                pathway_mechanisms=[],
                # Retrieval covers KEGG, Pathway Commons and LLM inference, so naming KEGG was wrong.
                # mechanistic_summary='No pathway structures could be retrieved from KEGG.',
                mechanistic_summary='No pathway structures could be retrieved.',
                pathway_structures=[],
                pathway_overlaps=[asdict(o) for o in pathway_overlaps],
                report_section=self._generate_empty_report()
            )

        # Build LLM prompt with curated data
        print('\n  Generating LLM interpretation of pathway mechanisms...')

        # Use batch processing for large pathway sets to avoid truncation
        BATCH_THRESHOLD = 15  # Process in batches if more than this many pathways
        # Smaller batches keep each JSON response short: local models slip on JSON far more
        # often on long payloads (a missing comma anywhere fails json.loads for the whole
        # batch). 5 roughly halves per-batch output vs 10; the salvage-retry below recovers
        # the rest when a batch still comes back malformed.
        BATCH_SIZE = 5        # Pathways per batch

        try:
            response = None  # bound before use so the except handler can't NameError
            if len(pathway_structures) > BATCH_THRESHOLD:
                print(f'  Large pathway set detected ({len(pathway_structures)} pathways)')
                print(f'  Using batch processing to prevent response truncation...')
                result = self._process_pathways_in_batches(
                    pathway_structures,
                    pathway_overlaps,
                    themes,
                    batch_size=BATCH_SIZE
                )
            else:
                print(f'  Small pathway set ({len(pathway_structures)} pathways) - using single-shot processing')
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_user_prompt(pathway_structures, pathway_overlaps, themes)

                # Get LLM interpretation with increased token limit
                response = self.llm.chat(
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    max_tokens=60000,  # Large limit for comprehensive analysis
                    temperature=0.0,
                    seed=42
                )

                # Parse JSON response
                result = json.loads(response)

            # The engine names each card from the tag the model returned, before any check
            # that looks a card's pathway up by name (relation checks, genes, crosstalk).
            name_counts = resolve_mechanism_pathways(
                (result or {}).get('pathwayMechanisms') or [] if isinstance(result, dict) else [],
                pathway_structures)
            print(f'  Card pathway names: {name_counts}')

            # Validate against KEGG data
            validated_result = self._validate_against_kegg(result, pathway_structures, genes)

            # Canonicalize LLM-echoed pathway names back to the retrieved structure
            # names. The model sometimes echoes the KEGG-id suffix it was shown in the
            # prompt (e.g. "Cell cycle (hsa04110)"); left as-is that name misses the
            # coverage check below (→ duplicate backfilled stub), the confidence lookup
            # (→ "Unverified" on the real card), and de-dup — all for the same pathway.
            _canonicalize_mechanism_names(
                validated_result.get('pathwayMechanisms') or [], pathway_structures)

            # Items 11/20 and 21: gene check and engine genes/fold changes on every card,
            # before the report section is built so the printed cards carry the result.
            gene_counts = normalize_card_genes(
                validated_result.get('pathwayMechanisms') or [], pathway_structures, genes)
            print(f'  Card genes: {gene_counts}')
            # Item 22: crosstalk genes must come from the computed DE-gene overlaps.
            crosstalk_counts = check_crosstalk(
                validated_result.get('pathwayMechanisms') or [], pathway_overlaps,
                pathway_structures, genes)
            print(f'  Crosstalk check: {crosstalk_counts}')

            # Part C: ensure every retrieved structure has a mechanism entry. The
            # interpretation LLM may return nothing (unparseable/empty) or drop
            # pathways (failed batches); backfill the missing ones from the structures
            # so the report matches the retrieved data instead of rendering blank.
            mechs = list(validated_result.get('pathwayMechanisms') or [])
            # Match on id-stripped, normalized names so an LLM-echoed name that differs
            # only in case/whitespace/id-suffix isn't treated as a missing pathway
            # (which would append a duplicate structure-derived card).
            covered = {_pathway_key(m.get('pathway') or '') for m in mechs}
            missing = _structures_missing_mechanisms(mechs, pathway_structures)
            if missing:
                print(f'  [Step {self.step_number}] Backfilling {len(missing)} pathway(s) '
                      f'with no interpretation from retrieved structures')
                mechs.extend(self._synthesize_mechanisms_from_structures(missing))
                validated_result['pathwayMechanisms'] = mechs
                # If interpretation produced nothing at all, the existing summary is
                # empty or a generic error string — replace it with an honest,
                # structure-derived overview.
                if not covered:
                    validated_result['mechanisticSummary'] = \
                        self._generate_summary_from_structures(mechs)

            print(f'  Pathway Mechanisms Interpreted: {len(validated_result.get("pathwayMechanisms", []))}')

            # Generate report section
            report_section = self._generate_report_section(
                validated_result,
                pathway_structures
            )

            structure_dicts = [self._pathway_structure_to_dict(ps) for ps in pathway_structures]
            return PathwayMechanismResult(
                pathway_mechanisms=validated_result.get('pathwayMechanisms', []),
                mechanistic_summary=validated_result.get('mechanisticSummary', ''),
                pathway_structures=structure_dicts,
                pathway_overlaps=[asdict(o) for o in pathway_overlaps],
                report_section=report_section,
                upstream_regulators=self._identify_upstream_regulators(structure_dicts, genes)
            )

        except json.JSONDecodeError as e:
            print(f'  ⚠️  Error parsing JSON response: {e}')
            if response is not None:
                print(f'  Response length: {len(response)} characters')
                print(f'  First 500 chars: {response[:500]}')
                print(f'  Last 500 chars: {response[-500:]}')

            # Part C: the interpretation LLM returned unparseable JSON, but structures
            # WERE retrieved — synthesize mechanisms from them rather than emitting the
            # misleading "No pathway structures could be retrieved from KEGG" report.
            print('  Interpretation JSON unparseable — synthesizing mechanisms from '
                  'retrieved structures')
            synthesized = self._synthesize_mechanisms_from_structures(pathway_structures)
            fallback_result = {
                'pathwayMechanisms': synthesized,
                'mechanisticSummary': self._generate_summary_from_structures(synthesized),
            }
            structure_dicts = [self._pathway_structure_to_dict(ps) for ps in pathway_structures]
            return PathwayMechanismResult(
                pathway_mechanisms=synthesized,
                mechanistic_summary=fallback_result['mechanisticSummary'],
                pathway_structures=structure_dicts,
                pathway_overlaps=[asdict(o) for o in pathway_overlaps],
                report_section=self._generate_report_section(fallback_result, pathway_structures),
                upstream_regulators=self._identify_upstream_regulators(structure_dicts, genes)
            )
        except Exception as e:
            print(f'  Error in Step {self.step_number}: {e}')
            raise

    def _create_theme_batches(
        self,
        pathway_structures: List[Dict],
        themes: Optional[List[Dict]] = None,
        max_batch_size: int = 10
    ) -> List[tuple]:
        """
        Group pathway structures by their theme membership for coherent processing

        This creates batches where pathways from the same biological theme are
        processed together, preserving thematic coherence and improving cross-pathway
        connection detection.

        Args:
            pathway_structures: All pathway structures to batch
            themes: Optional themes from Step 1 with pathway groupings
            max_batch_size: Maximum pathways per batch (for splitting large themes)

        Returns:
            List of (batch_name, pathways) tuples
        """
        if not themes:
            # Fallback to sequential batching if no themes provided
            batches = []
            for i in range(0, len(pathway_structures), max_batch_size):
                batch = pathway_structures[i:i + max_batch_size]
                batches.append((f"Sequential Batch", batch))
            return batches

        # Build pathway name -> theme mapping
        pathway_to_theme = {}
        theme_info = {}  # theme_name -> theme metadata

        for theme in themes:
            theme_name = theme.get('name', f"Theme {theme.get('cluster_number', 'Unknown')}")
            theme_info[theme_name] = theme

            for pathway in theme.get('pathways', []):
                pathway_name = pathway.get('name', pathway.get('pathway', ''))
                pathway_to_theme[pathway_name] = theme_name

        # Group pathway_structures by theme
        theme_groups = {}
        ungrouped_pathways = []

        for ps in pathway_structures:
            pathway_name = ps['pathway']
            theme_name = pathway_to_theme.get(pathway_name)

            if theme_name:
                if theme_name not in theme_groups:
                    theme_groups[theme_name] = []
                theme_groups[theme_name].append(ps)
            else:
                ungrouped_pathways.append(ps)

        # Create batches: one batch per theme (split if too large)
        batches = []

        # Sort themes by the order they appear in the themes list
        theme_order = {t.get('name', f"Theme {t.get('cluster_number')}"): i
                      for i, t in enumerate(themes)}

        for theme_name in sorted(theme_groups.keys(), key=lambda t: theme_order.get(t, 999)):
            pathways = theme_groups[theme_name]

            if len(pathways) > max_batch_size:
                # Split large theme into sub-batches
                for sub_idx, i in enumerate(range(0, len(pathways), max_batch_size), 1):
                    sub_batch = pathways[i:i + max_batch_size]
                    batch_name = f"{theme_name} (Part {sub_idx})"
                    batches.append((batch_name, sub_batch))
            else:
                batches.append((theme_name, pathways))

        # Add ungrouped pathways as final batch(es)
        if ungrouped_pathways:
            if len(ungrouped_pathways) > max_batch_size:
                # Split ungrouped into sub-batches
                for sub_idx, i in enumerate(range(0, len(ungrouped_pathways), max_batch_size), 1):
                    sub_batch = ungrouped_pathways[i:i + max_batch_size]
                    batch_name = f"Ungrouped Pathways (Part {sub_idx})"
                    batches.append((batch_name, sub_batch))
            else:
                batches.append(("Ungrouped Pathways", ungrouped_pathways))

        return batches

    def _process_pathways_in_batches(
        self,
        pathway_structures: List[Dict],
        pathway_overlaps: List[PathwayOverlap],
        themes: Optional[List[Dict]] = None,
        batch_size: int = 10
    ) -> Dict:
        """
        Process pathways in theme-aware batches to avoid response truncation

        This method groups pathways by their biological themes and processes each
        theme independently. This preserves thematic coherence and improves cross-pathway
        connection detection while preventing JSON truncation errors.

        Args:
            pathway_structures: All pathway structures to process
            pathway_overlaps: Pathway overlap data
            themes: Optional themes from Step 1 for theme-aware batching
            batch_size: Maximum pathways per batch (default: 10)

        Returns:
            Combined result dict with all pathway mechanisms
        """
        total_pathways = len(pathway_structures)

        # Create theme-based batches
        batches = self._create_theme_batches(pathway_structures, themes, batch_size)
        total_batches = len(batches)

        print(f'  Processing {total_pathways} pathways in {total_batches} theme-aware batches...')
        if themes:
            print(f'  Using hierarchical batching: grouping by biological themes')
        else:
            print(f'  No themes provided - using sequential batching')

        all_mechanisms = []
        failed_pathways = []

        # Process each theme-based batch
        for batch_num, (batch_name, batch) in enumerate(batches, 1):
            batch_pathway_names = [ps['pathway'] for ps in batch]

            print(f'\n  Theme Batch {batch_num}/{total_batches}: {batch_name} ({len(batch)} pathways)')
            # Show first 3 pathway names
            preview = batch_pathway_names[:3] + (['...'] if len(batch) > 3 else [])
            print(f'    Pathways: {", ".join(preview)}')

            # Interpret the batch; on a malformed-JSON response, salvage by splitting rather
            # than dropping all its pathways (see _interpret_pathway_batch).
            batch_mechanisms, batch_failed = self._interpret_pathway_batch(
                batch, batch_name, pathway_overlaps, themes)
            all_mechanisms.extend(batch_mechanisms)
            failed_pathways.extend(batch_failed)

        # Summary
        print(f'\n  Batch processing complete:')
        print(f'    ✅ Successful: {len(all_mechanisms)}/{total_pathways} mechanisms')
        if failed_pathways:
            print(f'    ❌ Failed: {len(failed_pathways)} pathways')
            print(f'    Failed pathways: {", ".join(failed_pathways[:5])}{"..." if len(failed_pathways) > 5 else ""}')

        # Generate overall summary from all mechanisms
        mechanistic_summary = self._generate_summary_from_mechanisms(all_mechanisms)

        return {
            'pathwayMechanisms': all_mechanisms,
            'mechanisticSummary': mechanistic_summary,
            'failed_pathways': failed_pathways,
            'total_processed': len(all_mechanisms),
            'total_requested': total_pathways
        }

    # Max recursive splits when salvaging a malformed batch. From BATCH_SIZE=5 this reaches
    # single pathways well within the cap (5 -> 2/3 -> 1); the cap just bounds the recursion.
    _BATCH_SPLIT_MAX_DEPTH = 3

    def _interpret_pathway_batch(
        self,
        batch: List[Dict],
        batch_name: str,
        pathway_overlaps: List['PathwayOverlap'],
        themes: Optional[List[Dict]],
        seed: int = 42,
        depth: int = 0,
    ) -> Tuple[List[Dict], List[str]]:
        """Interpret one batch of pathways into mechanisms, salvaging on malformed JSON.

        Local models intermittently emit syntactically invalid JSON on a long batch
        response (a single missing comma fails ``json.loads`` for the whole batch). Instead
        of dropping all its pathways, we:
          1. parse tolerantly (``parse_llm_json`` strips fences/prose);
          2. on failure with >1 pathway, split the batch in half and retry each half with a
             fresh LLM call — a shorter prompt yields a shorter, far-likelier-valid response;
          3. on failure with a lone pathway, retry once with a different seed (which changes
             the sampling even at temperature 0), then skip only that one pathway.

        Returns ``(mechanisms, failed_pathway_names)``.
        """
        batch_names = [ps.get('pathway', '') for ps in batch]

        try:
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(batch, pathway_overlaps, themes)
            response = self.llm.chat(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                max_tokens=20000,  # Smaller per batch for reliability
                temperature=0.0,
                seed=seed,
            )
        except Exception as e:
            print(f'    ❌ Error calling LLM for batch "{batch_name}": {e}')
            return [], batch_names

        try:
            result = parse_llm_json(response)
            mechanisms = result.get('pathwayMechanisms', []) if isinstance(result, dict) else []
            print(f'    ✅ Success: {len(mechanisms)} mechanisms generated ({batch_name})')
            return mechanisms, []
        except json.JSONDecodeError as e:
            # Salvage path 1: split a multi-pathway batch and retry each half.
            if len(batch) > 1 and depth < self._BATCH_SPLIT_MAX_DEPTH:
                mid = len(batch) // 2
                print(f'    ⚠️  JSON parse error in "{batch_name}" ({e}); '
                      f'salvaging by splitting {len(batch)} → {mid}+{len(batch) - mid}')
                left_m, left_f = self._interpret_pathway_batch(
                    batch[:mid], f'{batch_name} [{depth + 1}A]', pathway_overlaps, themes,
                    seed=seed + 1, depth=depth + 1)
                right_m, right_f = self._interpret_pathway_batch(
                    batch[mid:], f'{batch_name} [{depth + 1}B]', pathway_overlaps, themes,
                    seed=seed + 2, depth=depth + 1)
                return left_m + right_m, left_f + right_f
            # Salvage path 2: a lone pathway (or hit the split cap) — one varied-seed retry.
            if depth <= self._BATCH_SPLIT_MAX_DEPTH:
                print(f'    ⚠️  JSON parse error on "{batch_name}" ({len(batch)} pathway(s)); '
                      f'retrying once with a different seed')
                return self._interpret_pathway_batch(
                    batch, batch_name, pathway_overlaps, themes,
                    seed=seed + 100, depth=self._BATCH_SPLIT_MAX_DEPTH + 1)
            # Give up on just these pathways.
            print(f'    ❌ Still invalid after retry; skipping {len(batch)} pathway(s): '
                  f'{", ".join(batch_names)}')
            return [], batch_names

    def _generate_summary_from_mechanisms(self, mechanisms: List[Dict]) -> str:
        """
        Generate mechanistic summary from individual pathway mechanisms

        Args:
            mechanisms: List of pathway mechanism dicts

        Returns:
            Summary string describing overall findings
        """
        if not mechanisms:
            return "No pathway mechanisms could be generated due to processing errors."

        # Get pathway names for summary
        pathway_names = [m.get('pathway', 'Unknown') for m in mechanisms[:5]]

        # Count up/down regulated
        up_count = sum(1 for m in mechanisms if 'up' in m.get('pathway', '').lower() or
                      'upregulated' in str(m.get('functionalConsequences', '')).lower())
        down_count = sum(1 for m in mechanisms if 'down' in m.get('pathway', '').lower() or
                        'downregulated' in str(m.get('functionalConsequences', '')).lower())

        summary = (
            f"Analyzed {len(mechanisms)} pathway mechanisms involving differentially expressed genes. "
            f"Key pathways include: {', '.join(pathway_names)}. "
            f"Mechanisms reveal coordinated regulation across multiple pathways with significant crosstalk "
            f"and functional interactions between signaling cascades."
        )

        return summary

    def _detect_pathway_source(self, pathway: Dict) -> str:
        """
        Detect if pathway is from KEGG or other sources.

        Only positively identifies KEGG; everything else returns 'other'.
        Non-KEGG pathways use Pathway Commons + LLM for mechanism inference.

        Args:
            pathway: Pathway dict from Step 1 enrichment

        Returns:
            'kegg' or 'other'
        """
        # Check explicit database/source field
        source_field = pathway.get('source', pathway.get('database', ''))
        if source_field:
            if source_field.lower() == 'kegg':
                return 'kegg'
            else:
                return 'other'

        # Check for KEGG ID pattern (hsa04010, mmu04010)
        pathway_id = pathway.get('id', pathway.get('pathwayId', ''))
        if re.match(r'^[a-z]{3}\d{5}$', str(pathway_id)):
            return 'kegg'

        # Check pathway name for KEGG marker
        pathway_name = pathway.get('name', pathway.get('pathwayName', ''))
        if '(KEGG)' in pathway_name.upper():
            return 'kegg'

        return 'other'

    def _get_pathway_structures(
        self,
        pathways: List[Dict],
        genes: List[Dict],
        organism: str
    ) -> List[Dict[str, Any]]:
        """
        Get pathway structures from KEGG (curated) or Pathway Commons + LLM (inferred)

        Routes each pathway to the appropriate handler based on source detection:
        - KEGG: Curated database structures with high confidence
        - Other (Reactome, GO, MitoCarta, MSigDB, custom): Pathway Commons grounded
          LLM interpretation (pc_grounded if PC interactions found, inferred otherwise)
        """
        structures = []

        # Item 19: decide which pathway sources a bundled GMT describes, so the full
        # membership of those pathways can be looked up (see PathwayMembershipService).
        membership = getattr(self, 'membership_service', None)
        if membership is not None:
            associated = membership.associate_sources(pathways, organism)
            if associated:
                print(f'  GMT membership available for source(s): {associated}')

        for pathway in pathways:
            pathway_name = ''  # bound before any use so the except handler can't NameError
            extras = {}
            try:
                pathway_name = pathway.get('name') or pathway.get('pathwayName', '')
                pathway_id = pathway.get('id', pathway.get('pathwayId', ''))

                # Detect pathway source
                source = self._detect_pathway_source(pathway)

                # Route to appropriate handler
                if source == 'kegg':
                    # Curated KEGG pathway
                    print(f'  [KEGG] Processing: {pathway_name}')
                    structure = self.kegg_service.get_pathway_structure(pathway_name, organism)

                    if not structure:
                        # KEGG miss (e.g. KGML unavailable / pathway ID unresolved).
                        # Instead of dropping the pathway, degrade gracefully to the
                        # Pathway Commons/LLM route so a single KEGG failure can never
                        # blank the whole mechanisms section.
                        print(f'  [KEGG-MISS] {pathway_name}: KGML unavailable — '
                              f'using Pathway Commons/LLM fallback')
                        self._last_pc_extras = {}
                        inferred = self._build_inferred_structure(pathway, genes, organism)
                        if not inferred:
                            print(f'  ⚠️  Could not generate fallback structure for: {pathway_name}')
                            continue
                        structure, mapped_de_genes, de_relations, confidence = inferred
                        extras = getattr(self, '_last_pc_extras', {}) or {}
                        source = 'other'  # provenance: mechanism came from PC/LLM, not KEGG
                    else:
                        mapped_de_genes = self.kegg_service.map_de_genes_to_pathway(structure, genes)
                        de_relations = self.kegg_service.find_de_gene_relations(structure, mapped_de_genes)

                        if not mapped_de_genes and structure.genes:
                            # No DE genes but KEGG structure has gene entries — use them as placeholders
                            print(f'    ℹ️  No DE genes mapped — using {len(structure.genes)} KEGG pathway genes')
                            # Every gene of a multi-gene box (item 17), not just the first;
                            # still capped at 30 placeholders, each symbol once.
                            # mapped_de_genes = [MappedDEGene(..., gene_symbol=entry.gene_symbol
                            #     or entry.name, ...) for entry in structure.genes[:30]
                            #     if entry.gene_symbol]  # first gene of each box only
                            mapped_de_genes, seen_syms = [], set()
                            for entry in structure.genes:
                                for sym in entry_symbols(entry):
                                    if len(mapped_de_genes) >= 30 or _sym(sym) in seen_syms:
                                        continue
                                    seen_syms.add(_sym(sym))
                                    mapped_de_genes.append(MappedDEGene(
                                        pathway_entry_id=entry.id,
                                        gene_symbol=sym,
                                        kegg_names=entry.names,
                                        fold_change=0.0,
                                        p_value=1.0,
                                        direction='unknown'))
                            confidence = 'gene_set'  # KEGG structure but no DE data
                        else:
                            confidence = 'high'  # Curated data with DE genes

                else:  # source == 'other'
                    self._last_pc_extras = {}
                    inferred = self._build_inferred_structure(pathway, genes, organism)
                    if not inferred:
                        print(f'  ⚠️  Could not generate LLM structure for: {pathway_name}')
                        continue
                    structure, mapped_de_genes, de_relations, confidence = inferred
                    extras = getattr(self, '_last_pc_extras', {}) or {}

                if source != 'kegg' and not extras.get('pc_interactions'):
                    # LLM-proposed relations (native non-KEGG pathways and KEGG misses):
                    # at least one end must be an input gene.
                    de_relations, dropped_llm = filter_llm_relations(
                        de_relations, _input_gene_set(genes))
                    if dropped_llm:
                        print(f'    ⚠️  Dropped {len(dropped_llm)} LLM-proposed relation(s) '
                              f'with no input gene at either end ({pathway_name})')

                # Capture enrichment score, metric (NES preferred) and direction
                es_value, es_metric = _enrichment_metric(pathway)
                enrichment_direction = None
                if es_value is not None:
                    if es_value > FC_EPSILON:
                        enrichment_direction = 'upregulated'
                    elif es_value < -FC_EPSILON:
                        enrichment_direction = 'downregulated'
                    else:
                        enrichment_direction = 'unchanged'

                structures.append({
                    'pathway': pathway_name,
                    'pathway_id': pathway_id or pathway_name,
                    'source': source,  # Database source (kegg/reactome/other)
                    'confidence': confidence,  # Confidence level (high/inferred)
                    'structure': structure,
                    'mapped_de_genes': mapped_de_genes,
                    'de_relations': de_relations,
                    'p_value': pathway.get('pValue'),
                    'p_value_fdr': pathway.get('pValueFDR'),
                    'enrichment_score': es_value,
                    'enrichment_metric': es_metric,
                    'enrichment_direction': enrichment_direction,
                    # Item 19: the retrieved PC interactions (upper-case, filtered), the
                    # upper -> input-casing map for display, and the retrieval counts.
                    'pc_interactions': extras.get('pc_interactions', []),
                    'display_names': extras.get('display_names', {}),
                    'pc_stats': extras.get('pc_stats', {}),
                })

                print(f'    ✓ [{source.upper()}] {pathway_name}: {len(mapped_de_genes)} DE genes, '
                      f'{len(de_relations)} relations (confidence: {confidence})')

                # Log detailed information
                if mapped_de_genes:
                    print(f'      Mapped DE genes: {", ".join([g.gene_symbol for g in mapped_de_genes[:5]])}{"..." if len(mapped_de_genes) > 5 else ""}')

                if de_relations:
                    print(f'      Key relations:')
                    for rel in de_relations[:3]:
                        print(f'        - {rel.source} --[{rel.subtype}]--> {rel.target}')
                    if len(de_relations) > 3:
                        print(f'        ... and {len(de_relations) - 3} more')

            except Exception as e:
                print(f'    Error processing pathway {pathway_name}: {e}')

        # Dev-only provenance summary (grep-able): shows whether each mechanism came
        # from curated KEGG KGML or the Pathway Commons/LLM fallback. Not surfaced to
        # end users — the report keeps the human-facing "Confidence" badges.
        if structures:
            conf = Counter(s['confidence'] for s in structures)
            kgml_count = sum(1 for s in structures if s['source'] == 'kegg')
            fallback_count = len(structures) - kgml_count
            print(
                f"  [Step {self.step_number}] Mechanism provenance: curated_KGML={kgml_count}, "
                f"fallback_PC/LLM={fallback_count} "
                f"(high={conf.get('high', 0)}, pc_grounded={conf.get('pc_grounded', 0)}, "
                f"inferred={conf.get('inferred', 0)}, gene_set={conf.get('gene_set', 0)}), "
                f"total={len(structures)}"
            )

        return structures

    def _build_inferred_structure(
        self,
        pathway: Dict,
        genes: List[Dict],
        organism: str
    ) -> Optional[Tuple[Any, List, List, str]]:
        """Build a pathway structure for non-curated pathways via Pathway Commons + LLM.

        Used both for natively non-KEGG pathways and as a graceful fallback when a
        KEGG KGML lookup fails.

        Args:
            pathway: Pathway dict from Step 1 enrichment (needs 'name'/'pathwayName',
                'genes').
            genes: List of differentially expressed genes.
            organism: Organism name (e.g. 'Homo sapiens').

        Returns:
            A tuple (structure, mapped_de_genes, de_relations, confidence), or None if
            no structure could be produced.
        """
        pathway_name = pathway.get('name') or pathway.get('pathwayName', '')
        pathway_genes = pathway.get('genes', [])

        # Main L236: retrieve the Pathway Commons interactions for ALL pathway genes.
        # Full membership comes from a bundled GMT when one applies; otherwise the genes
        # the website sent (W6 sends only DE members). Symbols are queried and compared
        # upper-case, because PC's graph endpoint resolves only upper-case symbols (a
        # mouse-cased query returns nothing); input casing is kept for display.
        membership = getattr(self, 'membership_service', None)
        gmt_members = membership.members(pathway) if membership is not None else None
        members = list(gmt_members) if gmt_members else list(pathway_genes)
        de_symbols = [g.get('geneSymbol') or g.get('gene') or '' for g in genes or []]
        display_names = {}
        # The input's DE spelling wins, then the membership's (it may come from a GMT).
        # for name in list(members) + [s for s in de_symbols if s]:  # GMT casing won
        for name in [s for s in de_symbols if s] + list(members):
            display_names.setdefault(_sym(name), name)
        members_upper = {_sym(g) for g in members if g}
        de_upper = {_sym(s) for s in de_symbols if s}
        query = sorted(members_upper)

        raw_interactions = self.pc_service.get_interactions_between(query)
        pc_relations, pc_stats = filter_pc_interactions(raw_interactions, members_upper, de_upper)
        pc_stats.update({'members': len(members_upper),
                         'membership_source': 'gmt' if gmt_members else 'input'})

        if pc_relations:
            # Retrieved relations exist: they are this pathway's relations, extracted the
            # same way as for KEGG (DE gene at one end). No per-pathway inference call.
            print(f'  [PC] Processing: {pathway_name} ({len(pc_relations)} of '
                  f'{pc_stats["retrieved"]} Pathway Commons interactions kept)')
            mapped_de_genes = self._map_de_genes_to_pathway({'genes': members}, genes)
            structure = PCStructure(id=pathway.get('id', pathway_name), name=pathway_name,
                                    organism=organism)
            self._last_pc_extras = {'pc_interactions': pc_relations,
                                    'display_names': display_names, 'pc_stats': pc_stats}
            return structure, mapped_de_genes, list(pc_relations), 'pc_grounded'

        print(f'  [LLM] Processing: {pathway_name} (no Pathway Commons interaction retrieved)')
        self._last_pc_extras = {'pc_interactions': [], 'display_names': display_names,
                                'pc_stats': pc_stats}
        # Nothing retrieved: the inferred path is unchanged (decision D3 keeps the
        # model's inferred relations for these pathways).
        pc_interactions = []
        llm_structure = self._get_llm_pathway_structure(
            pathway, genes, organism, pc_interactions=pc_interactions
        )

        if not llm_structure:
            return None

        # LLM structure already contains mapped genes and inferred relations
        structure = llm_structure['structure']
        mapped_de_genes = llm_structure['mapped_de_genes']
        de_relations = llm_structure['de_relations']
        has_de_genes = llm_structure.get('has_de_genes', True)

        # Assign confidence based on available evidence
        if pc_interactions:
            confidence = 'pc_grounded'  # PC data still valuable with or without DE genes
        elif has_de_genes:
            confidence = 'inferred'
        else:
            confidence = 'gene_set'  # Gene-set-only, no DE genes, no PC data

        return structure, mapped_de_genes, de_relations, confidence

    def _is_reactome_id(self, identifier: str) -> bool:
        """Check if string matches Reactome stable ID pattern"""
        return bool(re.match(r'^R-[A-Z]{3}-\d+$', str(identifier)))

    def _map_de_genes_to_pathway(
        self,
        pathway: Dict,
        genes: List[Dict]
    ) -> List[MappedDEGene]:
        """
        Map DE genes to a pathway based on the pathway's gene list

        Args:
            pathway: Pathway dict with 'genes' field (list of gene symbols)
            genes: List of all DE genes

        Returns:
            List of MappedDEGene objects for genes in this pathway
        """
        # Case-insensitive on both sides (mouse vs upper-case symbols); the DE gene's own
        # casing is kept.
        pathway_genes = {_sym(g) for g in pathway.get('genes', []) or []}
        mapped_genes = []

        for gene in genes:
            gene_symbol = gene.get('geneSymbol', '')
            if gene_symbol and _sym(gene_symbol) in pathway_genes:
                fc = gene.get('foldChange', 0)
                mapped_genes.append(MappedDEGene(
                    pathway_entry_id='',
                    gene_symbol=gene_symbol,
                    kegg_names=[gene_symbol],
                    fold_change=fc,
                    p_value=gene.get('pValue', 1.0),
                    direction=fc_direction(fc)
                ))

        return mapped_genes

    def _get_llm_pathway_structure(
        self,
        pathway: Dict,
        genes: List[Dict],
        organism: str,
        pc_interactions=None
    ) -> Optional[Dict]:
        """
        Build pathway structure using LLM knowledge for non-KEGG pathways.

        When Pathway Commons interactions are available, they are injected into the
        LLM prompt as grounding context to improve accuracy and reduce hallucination.

        Args:
            pathway: Pathway dict from Step 1 enrichment
            genes: List of all DE genes
            organism: Organism name
            pc_interactions: Optional list of PCInteraction from Pathway Commons

        Returns:
            Dict with 'structure', 'mapped_de_genes', 'de_relations' or None if failed
        """
        pathway_name = pathway.get('name') or pathway.get('pathwayName', '')
        pathway_genes = pathway.get('genes', [])

        if not pathway_genes:
            print(f'    ⚠️  No genes provided for pathway: {pathway_name}')
            return None

        # Map DE genes to this pathway (may be empty if no DE genes provided)
        mapped_de_genes = self._map_de_genes_to_pathway(pathway, genes)
        has_de_genes = len(mapped_de_genes) > 0

        if not has_de_genes:
            print(f'    ℹ️  No DE genes mapped for: {pathway_name} — using pathway gene list')
            # Create placeholder gene entries from pathway gene list for downstream compatibility
            mapped_de_genes = [
                MappedDEGene(
                    pathway_entry_id='',
                    gene_symbol=g,
                    kegg_names=[g],
                    fold_change=0.0,
                    p_value=1.0,
                    direction='unknown'
                )
                for g in pathway_genes[:30]  # Limit to avoid huge prompts
            ]

        # Build LLM prompt to infer pathway structure
        system_prompt = """You are a molecular biology expert interpreting pathway mechanisms from gene sets.

You are given:
1. A pathway name (e.g., GO term, MitoCarta, MSigDB, or custom gene set)
2. All genes in the pathway
3. Which genes are differentially expressed (if available)

Use your biological knowledge to infer:
- How these genes likely interact based on their known functions
- What regulatory relationships probably exist between them
- What biological processes they collectively regulate

IMPORTANT:
- Base inferences on established biological knowledge
- Mark relationships as "inferred" (not curated)
- Focus on DE genes when available, otherwise analyze gene set composition
- Be conservative - don't over-interpret

Return ONLY valid JSON, no markdown."""

        # Format DE gene section based on whether DE data is available
        if has_de_genes:
            de_gene_list = []
            for g in mapped_de_genes:
                direction = '↑' if g.direction == 'up' else '↓'
                de_gene_list.append(f'- {g.gene_symbol} (FC: {g.fold_change:.2f} {direction})')
            de_section = f"""**Differentially Expressed Genes:**
{chr(10).join(de_gene_list)}"""
        else:
            gene_sample = pathway_genes[:30]
            de_section = f"""**Pathway Member Genes ({len(pathway_genes)} total, showing {len(gene_sample)}):**
{", ".join(gene_sample)}{"..." if len(pathway_genes) > 30 else ""}

Note: No differential expression data available. Analyze mechanisms based on gene set composition."""

        # Add Pathway Commons grounding if available
        pc_context = ""
        if pc_interactions:
            relations_text = []
            for inter in pc_interactions[:30]:  # Limit to 30
                relations_text.append(
                    f"  {inter.source} --[{inter.interaction_type}]--> {inter.target}"
                )
            if relations_text:
                pc_context = f"""
**Reference: Curated gene interactions from Pathway Commons (KEGG, Reactome, WikiPathways, etc.):**
{chr(10).join(relations_text)}

Use ONLY the genes and interactions listed above.
Do NOT infer relationships not shown in this list.
"""

        user_prompt = f"""**Pathway:** {pathway_name}
**Organism:** {organism}

**All Genes in Pathway:** {", ".join(pathway_genes[:50])}{"..." if len(pathway_genes) > 50 else ""}

{de_section}
{pc_context}
Based on your biological knowledge, infer:
1. What is the biological function of this pathway?
2. How do the {'DE genes' if has_de_genes else 'pathway member genes'} likely interact or relate to each other?
3. What regulatory relationships probably exist?
4. What are the likely functional consequences of these gene {'changes' if has_de_genes else 'set composition'}?

Return JSON:
{{
  "biologicalFunction": "What this pathway/gene set represents (2-3 sentences)",
  "inferredRelations": [
    {{
      "source": "GENE1",
      "target": "GENE2",
      "type": "activation/inhibition/regulation",
      "confidence": "high/medium/low",
      "rationale": "Why this relationship is likely (based on known biology)"
    }}
  ],
  "deGeneRoles": [
    {{
      "gene": "GENE_NAME",
      "foldChange": {"FC_VALUE" if has_de_genes else "0"},
      "inferredRole": "What this gene likely does in the pathway context"
    }}
  ],
  "functionalConsequences": "Predicted biological outcomes (2-3 sentences)"
}}"""

        try:
            # Call LLM
            response = self.llm.chat(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                max_tokens=4000,
                temperature=0.0,
                seed=42
            )

            # Parse response
            llm_result = json.loads(response)

            # Convert LLM response to pathway structure format
            # Create mock PathwayStructure compatible with existing code
            from dataclasses import dataclass
            from typing import List as ListType

            @dataclass
            class MockRelation:
                source: str
                target: str
                type: str
                subtype: str

            @dataclass
            class MockStructure:
                id: str
                name: str
                organism: str

            mock_structure = MockStructure(
                id=pathway.get('id', pathway_name),
                name=pathway_name,
                organism=organism
            )

            # Convert inferred relations to mock relations
            de_relations = []
            for rel in llm_result.get('inferredRelations', []):
                de_relations.append(MockRelation(
                    source=rel.get('source', ''),
                    target=rel.get('target', ''),
                    type='inferred',
                    subtype=rel.get('type', 'regulation')
                ))

            return {
                'structure': mock_structure,
                'mapped_de_genes': mapped_de_genes,
                'de_relations': de_relations,
                'has_de_genes': has_de_genes,
                'llm_interpretation': llm_result  # Store full LLM output
            }

        except json.JSONDecodeError as e:
            print(f'    ⚠️  LLM returned invalid JSON for {pathway_name}: {e}')
            return None
        except Exception as e:
            print(f'    ⚠️  Error getting LLM pathway structure for {pathway_name}: {e}')
            return None

    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM"""
        return """You are a molecular biology expert INTERPRETING pathway data.

You will receive pathways from different sources:

1. **CURATED pathways** (KEGG, Reactome):
   - Provided with curated regulatory relationships from databases
   - Your role: INTERPRET this curated data, NOT generate mechanisms
   - Use ONLY the genes and relationships provided
   - Do NOT infer relationships not shown in the data

2. **PC-GROUNDED pathways** (non-KEGG with Pathway Commons interactions):
   - Provided with curated interactions from Pathway Commons (22+ databases)
   - Your role: INTERPRET these interactions, NOT generate mechanisms
   - Use ONLY the genes and interactions provided
   - Do NOT infer relationships not shown in the data

3. **INFERRED pathways** (GO, MitoCarta, MSigDB, custom without PC data):
   - Provided with gene sets but NO curated relationships
   - Your role: Use biological knowledge to infer likely mechanisms
   - Base inferences on established biology
   - Mark as "inferred" and be conservative

4. **GENE-SET pathways** (gene sets without differential expression data):
   - Provided with pathway member genes but NO fold changes or DE data
   - Your role: Analyze gene set composition to infer pathway function and mechanisms
   - Focus on known gene functions and interactions
   - Be conservative - note the absence of expression data

For ALL pathways:
- Focus on differentially expressed genes when available
- Explain functional consequences
- Identify pathway crosstalk through shared genes

RESPONSE FORMAT:
- Return ONLY valid JSON
- NO markdown code blocks
- NO explanatory text before or after
- Just the raw JSON object"""

    def _build_user_prompt(
        self,
        pathway_structures: List[Dict],
        pathway_overlaps: List[PathwayOverlap],
        themes: Optional[List[Dict]] = None
    ) -> str:
        """Build user prompt for LLM with pathway data"""
        # The KEGG wording below asserts a KEGG structure is in the context and asks the model
        # to cite it. That is only true when every pathway in the batch came from KEGG; for
        # GO / Reactome / custom gene sets / PC / LLM fallbacks it invited fabricated "KEGG"
        # claims. All-KEGG batches keep the original wording unchanged.
        all_kegg = bool(pathway_structures) and all(
            ps.get('source', 'kegg') == 'kegg' for ps in pathway_structures)
        if all_kegg:
            intro = 'Interpret these curated KEGG pathway structures in light of differentially expressed genes:\n\n'
        else:
            intro = ('Interpret these pathways in light of differentially expressed genes. Each pathway is '
                     'marked with its evidence source ([CURATED], [PC-GROUNDED], [INFERRED] or [GENE-SET]); '
                     'cite only the structure or relations actually shown for it:\n\n')
        prompt_parts = [intro]

        # Add biological themes from Step 1 for context
        if themes:
            prompt_parts.append('**Biological Themes (from Step 1 clustering):**\n\n')
            for theme in themes[:5]:  # Limit to top 5 themes
                theme_name = theme.get('name', 'Unknown Theme')
                theme_desc = theme.get('description', '')
                theme_pathways = theme.get('pathways', [])
                prompt_parts.append(f'- **{theme_name}**: {theme_desc}\n')
                if theme_pathways:
                    pathway_names = [p.get('name', p.get('pathway', 'Unknown')) for p in theme_pathways[:3]]
                    prompt_parts.append(f'  Pathways: {", ".join(pathway_names)}\n')
            prompt_parts.append('\nUse these themes to provide biological context for pathway mechanisms.\n\n')
            prompt_parts.append('---\n\n')

        # Add pathway structures
        for ps in pathway_structures:
            prompt_parts.append(self._format_pathway_structure_for_llm(ps))
            prompt_parts.append('\n---\n')

        # Add pathway overlaps: for each pathway in this batch, every partner it shares DE
        # genes with, naming the shared genes (item 22, main L238 / suppl L256). The former
        # global top-10 pairs, as counts only, were the same for every batch.
        # for overlap in pathway_overlaps[:10]:  # replaced: per-pathway lists with gene names
        prompt_parts.append('\n**Pathway Gene Overlaps (shared DE genes; the only basis for crosstalk):**\n')
        for ps in pathway_structures:
            name = ps.get('pathway', '')
            partners = [o for o in pathway_overlaps
                        if _pathway_key(name) in (_pathway_key(o.pathway1), _pathway_key(o.pathway2))]
            if not partners:
                continue
            prompt_parts.append(f'- {name}:\n')
            for o in partners:
                other = o.pathway2 if _pathway_key(o.pathway1) == _pathway_key(name) else o.pathway1
                hub_genes_text = (f' [Hub genes: {", ".join(sorted(o.shared_hub_genes))}]'
                                  if o.shared_hub_genes else '')
                prompt_parts.append(
                    f'  - ↔ {other}: {o.shared_genes_count} shared DE gene(s): '
                    f'{", ".join(sorted(o.shared_genes))}{hub_genes_text}\n')

        # Add instructions. `curatedRelations` keeps its name for every source (renaming it
        # touches step 4, step 6, the validators and the orchestrator); provenance is carried
        # by each pathway's source/confidence instead, and the report heading reads that.
        if not all_kegg:
            prompt_parts.append('''
For each pathway, provide interpretation:
1. **Biological function** - Summarize what this pathway does (from the structure or gene set provided)
2. **DE gene involvement** - Which DE genes participate and their roles
3. **Regulatory relationships** - Which provided relations involve DE genes (activation/inhibition); for [INFERRED] or [GENE-SET] pathways, include only well-established relationships and do not attribute them to a database
4. **Crosstalk mechanisms** - How shared genes connect pathways (use overlap data)
5. **Functional consequences** - What biological effects result from these gene changes

Return JSON:
{
  "pathwayMechanisms": [
    {
      "pathwayRef": "the tag shown for the pathway, e.g. P1",
      "pathway": "pathway name exactly as shown",
      "biologicalFunction": "What it does (2-3 sentences)",
      "deGeneInvolvement": [
        {
          "gene": "GENE_NAME",
          "foldChange": FC_VALUE,
          "roleInPathway": "What this gene does in the pathway (from the data provided)"
        }
      ],
      "curatedRelations": [
        {
          "source": "GENE1",
          "target": "GENE2",
          "type": "activation/inhibition/phosphorylation",
          "isDE": true/false,
          "interpretation": "Functional consequence of this relation"
        }
      ],
      "crosstalk": ["How this pathway connects to others via shared genes"],
      "functionalConsequences": "Predicted biological outcomes (2-3 sentences)"
    }
  ],
  "mechanisticSummary": "Overall interpretation citing the provided relations and DE gene patterns (4-5 sentences)"
}
''')
            return ''.join(prompt_parts)

        prompt_parts.append('''
For each pathway, provide interpretation:
1. **Biological function** - Summarize what this pathway does (from KEGG structure)
2. **DE gene involvement** - Which DE genes participate and their roles
3. **Regulatory relationships** - Which KEGG relations involve DE genes (activation/inhibition)
4. **Crosstalk mechanisms** - How shared genes connect pathways (use overlap data)
5. **Functional consequences** - What biological effects result from these gene changes

Return JSON:
{
  "pathwayMechanisms": [
    {
      "pathwayRef": "the tag shown for the pathway, e.g. P1",
      "pathway": "pathway name exactly as shown",
      "biologicalFunction": "What it does (2-3 sentences)",
      "deGeneInvolvement": [
        {
          "gene": "GENE_NAME",
          "foldChange": FC_VALUE,
          "roleInPathway": "What this gene does in the pathway (from KEGG structure)"
        }
      ],
      "curatedRelations": [
        {
          "source": "GENE1",
          "target": "GENE2",
          "type": "activation/inhibition/phosphorylation",
          "isDE": true/false,
          "interpretation": "Functional consequence of this relation"
        }
      ],
      "crosstalk": ["How this pathway connects to others via shared genes"],
      "functionalConsequences": "Predicted biological outcomes (2-3 sentences)"
    }
  ],
  "mechanisticSummary": "Overall interpretation citing KEGG relations and DE gene patterns (4-5 sentences)"
}
''')

        return ''.join(prompt_parts)

    def _format_pathway_structure_for_llm(self, pathway_structure: Dict) -> str:
        """Format pathway structure for LLM prompt (handles both curated and inferred)"""
        pathway = pathway_structure['pathway']
        mapped_de_genes = pathway_structure['mapped_de_genes']
        de_relations = pathway_structure['de_relations']
        structure = pathway_structure['structure']
        source = pathway_structure.get('source', 'kegg')
        confidence = pathway_structure.get('confidence', 'high')
        es_value = pathway_structure.get('enrichment_score')
        es_metric = pathway_structure.get('enrichment_metric') or (
            'NES' if (es_value is not None and abs(es_value) > 1.0) else 'ES')
        enrichment_dir = pathway_structure.get('enrichment_direction')

        # The tag the model returns with this pathway's card (pathwayRef), so the engine
        # names the card rather than the model's spelling of the pathway.
        ref = pathway_structure.get('ref')
        label = f'Pathway {ref}' if ref else 'Pathway'
        # Format header based on source and confidence
        if source == 'kegg' and confidence != 'gene_set':
            formatted = [f'**{label}: {pathway}** (KEGG: {structure.id}) [CURATED]\n']
        elif confidence == 'pc_grounded':
            formatted = [f'**{label}: {pathway}** ({source.upper()}: {structure.id}) [PC-GROUNDED]\n']
        elif confidence == 'gene_set':
            formatted = [f'**{label}: {pathway}** ({source.upper()}: {structure.id}) [GENE-SET]\n']
        else:  # other (inferred)
            formatted = [f'**{label}: {pathway}** ({source.upper()}: {structure.id}) [INFERRED]\n']

        # Add enrichment score context if available (labelled by actual metric)
        if es_value is not None:
            direction_text = enrichment_dir.upper() if enrichment_dir else "UNKNOWN"
            magnitude = _magnitude_label(es_value, es_metric)

            formatted.append(f'**Pathway Enrichment:** {es_metric} = {es_value:+.3f} ({direction_text}, {magnitude} signal)\n')
            formatted.append(f'**Interpretation:** This pathway shows coordinated {enrichment_dir} in the disease state\n')

        formatted.append('\n')

        # DE genes in pathway
        if mapped_de_genes:
            formatted.append('**Differentially Expressed Genes in Pathway:**\n')
            for g in mapped_de_genes:
                direction = '↑' if g.direction == 'up' else '↓'
                formatted.append(
                    f'- {g.gene_symbol} (FC: {g.fold_change:.2f} {direction}, '
                    f'p={g.p_value:.2e})\n'
                )
            formatted.append('\n')

        # Relations involving DE genes (curated, pc_grounded, or inferred)
        if de_relations:
            if confidence == 'high':
                formatted.append(f'**{source.upper()} Curated Relations (involving DE genes):**\n')
            elif confidence == 'pc_grounded':
                formatted.append(f'**Pathway Commons Interactions (involving DE genes):**\n')
            else:
                formatted.append(f'**Inferred Relations (from LLM knowledge):**\n')

            # Symbols shown in the input casing (e.g. mouse "Ndufs1", not the upper-case
            # form Pathway Commons was queried with), so the gene list and the relations
            # name each gene the same way.
            display_names = pathway_structure.get('display_names') or {}
            for rel in de_relations:
                formatted.append(f'- {_display(rel.source, display_names)} --[{rel.subtype}]--> '
                                 f'{_display(rel.target, display_names)}\n')
            formatted.append('\n')
        else:
            if confidence == 'high':
                formatted.append(
                    f'**Note:** No direct regulatory relations found in {source.upper()} for these DE genes\n\n'
                )
            else:
                formatted.append(
                    f'**Note:** No relations inferred for these DE genes (use biological knowledge to identify likely interactions)\n\n'
                )

        return ''.join(formatted)

    def _calculate_pathway_overlaps(
        self,
        pathway_structures: List[Dict],
        hub_genes: List[str]
    ) -> List[PathwayOverlap]:
        """Calculate gene overlaps between pathways based on mapped DE genes"""
        overlaps = []

        # Create gene sets for each pathway from its mapped DE genes. Gene-set placeholders
        # (direction 'unknown', no DE data) are not DE genes and are excluded, so crosstalk
        # is computed only "through their common DE genes" (main L238, item 22).
        # Compared upper-case: KEGG-mapped genes carry KEGG's spelling ("Mcm2") while
        # other pathways carry the input's ("MCM2"); the first spelling seen is shown.
        pathway_gene_sets = {}
        display = {}
        for ps in pathway_structures:
            pathway_name = ps['pathway']
            # gene_symbols = {g.gene_symbol for g in ps['mapped_de_genes'] ...}  # case-sensitive
            gene_symbols = set()
            for g in ps['mapped_de_genes']:
                if getattr(g, 'direction', '') in ('up', 'down') and g.gene_symbol:
                    gene_symbols.add(_sym(g.gene_symbol))
                    display.setdefault(_sym(g.gene_symbol), g.gene_symbol)
            pathway_gene_sets[pathway_name] = gene_symbols
        hub_upper = {_sym(h) for h in hub_genes or []}

        # Calculate pairwise overlaps
        pathway_names = list(pathway_gene_sets.keys())
        for i, pathway1 in enumerate(pathway_names):
            for pathway2 in pathway_names[i + 1:]:
                shared_upper = pathway_gene_sets[pathway1] & pathway_gene_sets[pathway2]
                shared = {display[s] for s in shared_upper}
                if shared:
                    # shared_hub = [g for g in shared if g in hub_genes]  # case-sensitive
                    shared_hub = [g for g in shared if _sym(g) in hub_upper]

                    overlaps.append(PathwayOverlap(
                        pathway1=pathway1,
                        pathway2=pathway2,
                        shared_genes_count=len(shared),
                        shared_genes=list(shared),
                        shared_hub_genes=shared_hub
                    ))

        # Sort by overlap count
        overlaps.sort(key=lambda o: o.shared_genes_count, reverse=True)

        return overlaps

    def _validate_against_kegg(
        self,
        result: Dict,
        pathway_structures: List[Dict],
        input_genes: Optional[List[Dict]] = None
    ) -> Dict:
        """Validate/clean LLM output against KEGG curated data.

        Two data-driven passes (no report-specific hardcoding):
        1. Drop regulatory relations whose gene-shaped endpoint is absent from the actual
           node inventory — the union of every retrieved structure's GENES and COMPOUNDS,
           plus mapped DE genes and real KEGG relation endpoints. This removes
           hallucinated/typo'd endpoints (the stray "ALLD2") while KEEPING legitimate
           metabolites/second messengers (cAMP→CAMP, ATP, NAD…), which live in the
           structures' compound inventory. Validation is SKIPPED for a pathway whose
           structure has no inventory (inferred / gene-set / PC-grounded MockStructures) —
           we can't judge an endpoint we have no ground truth for, so we don't drop it.
        2. Strip leaked self-correction / meta-commentary from free-text fields.
        """
        # Ground-truth universe of legitimate node names (genes AND compounds) across all
        # retrieved structures. Compounds are included so gene-shaped metabolite symbols
        # (CAMP, ATP, NAD, GTP) are recognized as valid rather than dropped.
        valid_nodes: set = set()
        # Pathways (by normalized name) that actually have a curated node inventory, so we
        # only judge endpoints where we have ground truth.
        pathways_with_inventory: set = set()
        for ps in pathway_structures:
            structure = ps.get('structure')
            genes = list(getattr(structure, 'genes', []) or [])
            compounds = list(getattr(structure, 'compounds', []) or [])
            for entry in genes + compounds:
                for attr in ('gene_symbol', 'name'):
                    val = getattr(entry, attr, None)
                    if val:
                        valid_nodes.add(str(val).upper())
                for nm in getattr(entry, 'names', []) or []:
                    if nm:
                        valid_nodes.add(str(nm).upper())
            for g in ps.get('mapped_de_genes', []):
                if getattr(g, 'gene_symbol', None):
                    valid_nodes.add(g.gene_symbol.upper())
            for rel in ps.get('de_relations', []):
                for ep in (getattr(rel, 'source', None), getattr(rel, 'target', None)):
                    if ep:
                        valid_nodes.add(str(ep).upper())
            if genes or compounds:
                pathways_with_inventory.add(_pathway_key(ps['pathway']))

        # A token that looks like a gene/compound symbol (so free-text names with spaces
        # or lowercase, e.g. "acetyl-CoA", are never judged as a symbol).
        symbol_shape = re.compile(r'^[A-Z][A-Z0-9]{1,14}$')

        def _endpoint_ok(ep: str) -> bool:
            token = (ep or '').strip().upper()
            if not token or not symbol_shape.match(token):
                return True  # empty or not symbol-shaped — nothing to validate
            return token in valid_nodes

        # Clean the top-level mechanistic summary too (self-correction can leak here).
        if result.get('mechanisticSummary'):
            result['mechanisticSummary'] = sanitize_llm_text(result['mechanisticSummary'])

        # Retrieved relation set per pathway (main L240: "removes every relation that is
        # not among the retrieved KEGG relations or Pathway Commons interactions").
        reference_by_key = {}
        llm_proposed_keys = set()
        for ps in pathway_structures:
            ref = _relation_reference(ps)
            if ref is not None:
                reference_by_key[_pathway_key(ps.get('pathway', ''))] = ref
            elif ps.get('source') != 'kegg':
                llm_proposed_keys.add(_pathway_key(ps.get('pathway', '')))
        input_upper = _input_gene_set(input_genes)

        for mech in result.get('pathwayMechanisms') or []:
            # 0) Pair check: keep a listed relation only if its gene pair was retrieved for
            #    this pathway. Pathways with nothing retrieved keep their relations (D3).
            ref = reference_by_key.get(_pathway_key(mech.get('pathway', '')))
            rels = mech.get('curatedRelations')
            # 0b) LLM-proposed relations (nothing retrieved): keep only those with at
            #     least one input gene at either end (owner decision 2026-11-02).
            if (ref is None and isinstance(rels, list)
                    and _pathway_key(mech.get('pathway', '')) in llm_proposed_keys):
                kept, dropped = filter_llm_relations(
                    [r for r in rels if isinstance(r, dict)], input_upper)
                for d in dropped:
                    print(f'    ⚠️  Dropping LLM-proposed relation with no input gene: '
                          f'{d.get("source")} → {d.get("target")} ({mech.get("pathway")})')
                mech['curatedRelations'] = kept
                if dropped:
                    mech['droppedRelations'] = [
                        {'source': d.get('source'), 'target': d.get('target'),
                         'type': d.get('type'), 'reference': 'input_genes'} for d in dropped]
            if ref is not None and isinstance(rels, list):
                kept, dropped = [], []
                for rel in rels:
                    if isinstance(rel, dict) and _relation_in_reference(
                            rel.get('source'), rel.get('target'), ref):
                        kept.append(rel)
                    elif isinstance(rel, dict):
                        dropped.append({'source': rel.get('source'), 'target': rel.get('target'),
                                        'type': rel.get('type'), 'reference': ref[0]})
                for d in dropped:
                    print(f'    ⚠️  Dropping relation not among the retrieved {d["reference"]} '
                          f'relations: {d["source"]} → {d["target"]} ({mech.get("pathway")})')
                mech['curatedRelations'] = kept
                if dropped:
                    mech['droppedRelations'] = dropped

            # 1) Filter hallucinated relation endpoints — but only for pathways whose
            #    structure gave us a real inventory to validate against.
            can_validate = _pathway_key(mech.get('pathway', '')) in pathways_with_inventory
            rels = mech.get('curatedRelations')
            if isinstance(rels, list):
                kept = []
                for rel in rels:
                    if not isinstance(rel, dict):
                        continue
                    if can_validate and not (
                            _endpoint_ok(rel.get('source', '')) and _endpoint_ok(rel.get('target', ''))):
                        print(f'    ⚠️  Dropping relation with unknown endpoint: '
                              f'{rel.get("source")} → {rel.get("target")}')
                        continue
                    # 2a) Clean the relation's interpretation text.
                    if rel.get('interpretation'):
                        rel['interpretation'] = sanitize_llm_text(rel['interpretation'])
                    kept.append(rel)
                mech['curatedRelations'] = kept

            # 2b) Clean the other free-text fields.
            for field in ('biologicalFunction', 'functionalConsequences'):
                if mech.get(field):
                    mech[field] = sanitize_llm_text(mech[field])
            for g in mech.get('deGeneInvolvement') or []:
                if isinstance(g, dict) and g.get('roleInPathway'):
                    g['roleInPathway'] = sanitize_llm_text(g['roleInPathway'])
            ct = mech.get('crosstalk')
            if isinstance(ct, list):
                mech['crosstalk'] = [sanitize_llm_text(c) if isinstance(c, str) else c
                                     for c in ct]

        return result

    def _identify_upstream_regulators(
        self,
        pathway_structures: List[Dict],
        genes: List[Dict],
        top_n: int = 8,
    ) -> List[Dict]:
        """
        Identify candidate upstream regulators (transcription factors) of the
        DOWN-regulated program.

        Motivation: driver TFs of down-regulated, enzyme-centric programs (e.g. the
        hepatocyte metabolic master TFs HNF4A / PPARA / PPARGC1A / CEBPA) are frequently
        NOT differentially expressed AND are NOT reachable through KEGG ``GErel`` edges —
        the enriched metabolic maps encode enzyme<->enzyme (``ECrel``) relations, not
        transcriptional edges, so those TFs never appear. The previous KEGG-GErel scan
        therefore surfaced only TP53 (from the up-regulated p53-signaling map), which is
        misleading for a section meant to explain the down program.

        Primary method: test which TF's target set (regulon, from the bundled CollecTRI
        database) is over-represented among the down-regulated DE genes via a
        hypergeometric test with BH-FDR (see ``RegulonService.rank_tfs``). A TF whose
        regulon barely overlaps the down set (e.g. TP53) fails the overlap/FDR gate and
        drops out naturally — no pathway-specific special-casing. Fallback: if the regulon
        DB is unavailable or yields nothing, ask the LLM to propose candidate TFs, clearly
        labelled as hypotheses (``evidence_source='llm_hypothesis'``).
        """
        de_fc = gene_fc_lookup(genes)
        de_symbols = set(de_fc.keys())
        down_genes = {sym for sym, fc in de_fc.items() if fc < 0}
        if not down_genes:
            return []

        # Primary: regulon-overlap enrichment.
        if self.regulon_service.available:
            hits = self.regulon_service.rank_tfs(down_genes, top_n=top_n)
            if hits:
                print(f'    Upstream regulators: {len(hits)} candidate TF(s) from '
                      f'CollecTRI regulon enrichment')
                return self._format_regulon_hits(hits, de_fc, de_symbols, pathway_structures)
            # DB loaded but nothing cleared the significance gate — a distinct case from a
            # missing DB, so the report note must not falsely claim the DB was unavailable.
            reason = 'no_significant_enrichment'
        else:
            reason = 'db_unavailable'

        # Fallback: LLM-proposed candidates (labelled as hypotheses, not DB-derived).
        return self._propose_upstream_regulators_with_llm(
            de_fc, pathway_structures, top_n, reason)

    def _format_regulon_hits(
        self,
        hits: List[Dict],
        de_fc: Dict[str, float],
        de_symbols: set,
        pathway_structures: List[Dict],
    ) -> List[Dict]:
        """Map ``RegulonService`` hits into the report's upstream-regulator schema."""
        out = []
        for h in hits:
            tf = h['tf']
            # Show the most down-regulated overlapping targets first.
            targets = sorted(h['targets'], key=lambda g: de_fc.get(g, 0.0))
            out.append({
                'tf': tf,
                'num_targets': h['overlap_count'],      # back-compat alias
                'overlap_count': h['overlap_count'],
                'regulon_size': h['regulon_size'],
                'targets': targets,
                'target_direction': 'down',
                'is_de': tf in de_symbols,
                'tf_fold_change': (round(de_fc[tf], 2) if tf in de_fc else None),
                'inferred_tf_activity': h.get('inferred_tf_activity', 'unknown'),
                'p_value': h.get('p_value'),
                'enrichment_fdr': h.get('enrichment_fdr'),
                'evidence_pathways': self._find_down_pathways_for_targets(
                    targets, pathway_structures),
                'evidence_source': 'collectri',
                'mode': 'database',
            })
        return out

    def _find_down_pathways_for_targets(
        self,
        targets: List[str],
        pathway_structures: List[Dict],
        max_pathways: int = 3,
        fdr_max: float = 0.25,
    ) -> List[str]:
        """Enriched DOWN-regulated pathways (strongest |NES| first) that contain any of a
        TF's overlapping targets — evidence context without depending on GErel edges."""
        tset = {t.upper() for t in targets}
        scored = []
        for ps in pathway_structures or []:
            nes = to_float(ps.get('enrichment_score'))
            fdr = to_float(ps.get('p_value_fdr'))
            direction = (ps.get('enrichment_direction') or '').lower()
            is_down = (nes is not None and nes < 0) or direction.startswith('down')
            if not is_down:
                continue
            if fdr is not None and fdr > fdr_max:
                continue
            pw_genes = {(g.get('gene_symbol') or '').upper()
                        for g in ps.get('mapped_de_genes', [])}
            if tset & pw_genes:
                scored.append((abs(nes) if nes is not None else 0.0, ps.get('pathway', '')))
        scored.sort(key=lambda x: x[0], reverse=True)
        names, seen = [], set()
        for _, name in scored:
            if name and name not in seen:
                seen.add(name)
                names.append(name)
            if len(names) >= max_pathways:
                break
        return names

    def _propose_upstream_regulators_with_llm(
        self,
        de_fc: Dict[str, float],
        pathway_structures: List[Dict],
        top_n: int,
        reason: str = 'db_unavailable',
    ) -> List[Dict]:
        """Fallback when the regulon DB is unavailable or yields no significant TF: the LLM
        PROPOSES candidate upstream TFs from its own biology knowledge. Output is explicitly
        labelled ``evidence_source='llm_hypothesis'`` (and carries ``fallback_reason`` so the
        report note is accurate) so it is never confused with DB-derived rows."""
        down = sorted(((s, fc) for s, fc in de_fc.items() if fc < 0), key=lambda kv: kv[1])
        if not down:
            return []
        down_set = {s for s, _ in down}
        gene_list = ', '.join(f'{s} ({fc:+.2f})' for s, fc in down[:40])
        down_pathways = [ps.get('pathway', '') for ps in (pathway_structures or [])
                         if (to_float(ps.get('enrichment_score')) or 0.0) < 0]
        down_pathways = [p for p in down_pathways if p][:8]

        system = ('You are a systems-biology assistant. Propose candidate upstream '
                  'transcription-factor regulators for a set of DOWN-regulated genes, using '
                  'only well-established TF->target biology. Return JSON only.')
        user = (
            f'Down-regulated genes (symbol, log2FC): {gene_list}\n'
            f'Strongest down-regulated pathways: {", ".join(down_pathways)}\n\n'
            f'Propose up to {top_n} transcription factors most likely to drive this '
            'down-regulated program. For each, give example_targets drawn ONLY from the '
            'gene list above. Do not invent p-values.\n'
            'Return JSON: {"regulators": [{"tf": "SYMBOL", "rationale": "one sentence", '
            '"inferred_activity": "decreased|increased|mixed", '
            '"example_targets": ["GENE1", "GENE2"]}]}'
        )
        try:
            resp = self.llm.complete(user, system_prompt=system, temperature=0.0,
                                     response_format={'type': 'json_object'})
            data = parse_llm_json(resp) if isinstance(resp, str) else None
        except Exception as e:
            print(f'    Upstream regulators: LLM fallback unavailable ({e})')
            return []
        if not isinstance(data, dict):
            return []

        # The model can deviate from the requested shape even in JSON mode (a bare list of
        # symbols, a dict, or null entries). Coerce to a list and skip non-dict rows so a
        # malformed fallback degrades to [] rather than crashing the whole Step 3.
        regulators = data.get('regulators')
        if not isinstance(regulators, list):
            return []

        out = []
        for r in regulators[:top_n]:
            if not isinstance(r, dict):
                continue
            tf = r.get('tf')
            if not isinstance(tf, str) or not tf.strip():
                continue
            tf = tf.upper()
            ex_raw = r.get('example_targets')
            examples = ([g.upper() for g in ex_raw
                         if isinstance(g, str) and g.upper() in down_set]
                        if isinstance(ex_raw, list) else [])
            activity = r.get('inferred_activity')
            activity = activity if isinstance(activity, str) and activity.strip() else 'unknown'
            out.append({
                'tf': tf,
                'num_targets': len(examples),
                'overlap_count': len(examples),
                'regulon_size': None,
                'targets': examples,
                'target_direction': 'down',
                'is_de': tf in de_fc,
                'tf_fold_change': (round(de_fc[tf], 2) if tf in de_fc else None),
                'inferred_tf_activity': activity,
                'p_value': None,
                'enrichment_fdr': None,
                'evidence_pathways': [],
                'evidence_source': 'llm_hypothesis',
                'mode': 'llm',
                'fallback_reason': reason,
            })
        if out:
            why = ('regulon DB unavailable' if reason == 'db_unavailable'
                   else 'no significant regulon enrichment')
            print(f'    Upstream regulators: {len(out)} LLM-proposed candidate TF(s) '
                  f'({why})')
        return out

    def _pathway_structure_to_dict(self, ps: Dict) -> Dict:
        """Convert pathway structure to serializable dict"""
        display_names = ps.get('display_names') or {}
        result = {
            'pathway': ps['pathway'],
            'pathway_id': ps['pathway_id'],
            'source': ps.get('source', 'kegg'),  # Database source (kegg/reactome/other)
            'confidence': ps.get('confidence', 'high'),  # Confidence level (high/inferred)
            'p_value': ps['p_value'],
            'p_value_fdr': ps['p_value_fdr'],
            'de_genes_count': len(ps['mapped_de_genes']),
            'de_relations_count': len(ps['de_relations']),
            'mapped_de_genes': [
                {
                    'gene_symbol': g.gene_symbol,
                    'fold_change': g.fold_change,
                    'p_value': g.p_value,
                    'direction': g.direction
                }
                for g in ps['mapped_de_genes']
            ],
            # Shown to Step 4 through the tool results, so in the input casing; every
            # comparison upper-cases both sides.
            'de_relations': [
                {
                    'source': _display(r.source, display_names),
                    'target': _display(r.target, display_names),
                    'type': r.type,
                    'subtype': r.subtype
                }
                for r in ps['de_relations']
            ]
        }
        # Item 19: the retrieved Pathway Commons interactions (canonical upper-case form,
        # the reference for the Step 3 pair check and the Step 4 relation check) and the
        # retrieval counts.
        if ps.get('pc_interactions'):
            result['pc_interactions'] = [
                {'source': r.source, 'target': r.target, 'type': r.subtype}
                for r in ps['pc_interactions']
            ]
        if ps.get('pc_stats'):
            result['pc_stats'] = ps['pc_stats']

        # Add enrichment score fields if present
        if 'enrichment_score' in ps and ps['enrichment_score'] is not None:
            result['enrichment_score'] = ps['enrichment_score']
            result['enrichment_metric'] = ps.get('enrichment_metric', 'NES')
        if 'enrichment_direction' in ps and ps['enrichment_direction'] is not None:
            result['enrichment_direction'] = ps['enrichment_direction']

        return result

    def _generate_report_section(
        self,
        result: Dict,
        pathway_structures: List[Dict]
    ) -> str:
        """Generate markdown report section"""
        lines = [
            '## Pathway Mechanisms and Interactions\n\n',
            '### Overview\n\n',
            result.get('mechanisticSummary', ''),
            '\n\n### Pathway Details\n\n'
        ]

        # Create lookup for pathway structures (to get ES values)
        # Key on the id-stripped, normalized pathway name so a case/whitespace/id-suffix
        # difference between the LLM's echoed pathway name and the structure name doesn't
        # miss the lookup (a miss would drop the confidence badge to "Unverified" and
        # hide the NES line).
        pathway_es_lookup = {_pathway_key(ps['pathway']): ps for ps in pathway_structures}

        for mechanism in _dedupe_mechanisms(result.get('pathwayMechanisms', [])):
            pathway_name = mechanism.get('pathway', 'Unknown')

            # Skip fully-empty entries (no function, genes, relations, crosstalk or
            # consequences) — these are backfilled stubs that render as blank sections.
            has_content = (
                (mechanism.get('biologicalFunction') or '').strip()
                or mechanism.get('deGeneInvolvement')
                or mechanism.get('curatedRelations')
                or mechanism.get('crosstalk')
                or (mechanism.get('functionalConsequences') or '').strip()
            )
            if not has_content:
                continue

            lines.append(f'#### {pathway_name}\n\n')

            # Add pathway source and confidence.
            # Confidence must reflect actual evidence — do NOT default an unmatched
            # pathway to 'high' (that made every pathway read "High Confidence").
            ps = pathway_es_lookup.get(_pathway_key(pathway_name), {})
            source = ps.get('source', 'kegg').upper()
            confidence = ps.get('confidence') or 'unknown'
            confidence_badges = {
                'high': '✓ High Confidence',
                'pc_grounded': '✓ PC-Grounded',
                'gene_set': 'ℹ️ Gene Set Only',
                'inferred': '⚠ Inferred',
            }
            confidence_badge = confidence_badges.get(confidence, 'ℹ️ Unverified')
            lines.append(f'**Source:** {source} | **Confidence:** {confidence_badge}\n\n')

            # Add enrichment score if available (labelled by actual metric, NES-preferred)
            es_value = ps.get('enrichment_score')
            # Derive the metric consistently with _enrichment_metric when absent (legacy data):
            # a |value|>1 can only be NES since classic ES is bounded to [-1, 1].
            es_metric = ps.get('enrichment_metric') or (
                'NES' if (es_value is not None and abs(es_value) > 1.0) else 'ES')
            es_direction = ps.get('enrichment_direction')

            if es_value is not None:
                # Format with metric-calibrated magnitude and direction indicator
                magnitude = _magnitude_label(es_value, es_metric)
                direction_symbol = (
                    '↑' if es_direction == 'upregulated'
                    else '↓' if es_direction == 'downregulated' else '→')

                lines.append(
                    f'**Pathway Enrichment:** {es_metric} = {es_value:+.3f} {direction_symbol} '
                    f'({es_direction}, {magnitude} signal)\n\n'
                )

            # Biological function
            bio_func = mechanism.get('biologicalFunction', '')
            if bio_func:
                lines.append(f'**Biological Function:** {bio_func}\n\n')

            # DE gene involvement (de-duplicated by symbol — KEGG repeats a gene
            # across graphics nodes, which would otherwise list e.g. CYP1A1 x6)
            de_genes = _dedupe_de_genes(mechanism.get('deGeneInvolvement', []))
            # Drop genes whose fold change is explicitly ~0 — not differentially
            # expressed, and previously rendered with a misleading ↓ (e.g.
            # "MIR21 (FC: 0.00 ↓)"). Keep genes with missing/unparseable FC (shown
            # without a value/arrow) rather than discarding their annotation.
            de_genes = [
                g for g in de_genes
                if not (to_float(g.get('foldChange')) is not None
                        and abs(to_float(g.get('foldChange'))) < FC_EPSILON)
            ]
            if de_genes:
                lines.append('**Differentially Expressed Genes:**\n\n')
                for gene_info in de_genes:
                    gene = gene_info.get('gene', '')
                    fc = to_float(gene_info.get('foldChange'))
                    role = (gene_info.get('roleInPathway') or '').strip()
                    suffix = f': {role}' if role else ''
                    if fc is not None:
                        arrow = fc_arrow(fc)
                        fc_txt = f' (FC: {fc:+.2f} {arrow})' if arrow else f' (FC: {fc:+.2f})'
                    else:
                        fc_txt = ''
                    lines.append(f'- **{gene}**{fc_txt}{suffix}\n')
                lines.append('\n')

            # Regulatory relations. The field is named curatedRelations for every source, so
            # the heading is chosen from the pathway's actual provenance, not the field name.
            relations = mechanism.get('curatedRelations', [])
            if relations:
                # lines.append('**KEGG Curated Regulatory Relations:**\n\n')  # claimed KEGG for every source
                lines.append(f'{_relations_heading(source, confidence)}\n\n')
                for rel in relations:
                    # rel_source, not source: `source` is the pathway's database above.
                    rel_source = rel.get('source', '')
                    target = rel.get('target', '')
                    rel_type = rel.get('type', '')
                    interp = (rel.get('interpretation') or '').strip()
                    suffix = f': {interp}' if interp else ''
                    lines.append(f'- {rel_source} --[{rel_type}]--> {target}{suffix}\n')
                lines.append('\n')

            # Crosstalk mechanisms
            crosstalk = mechanism.get('crosstalk', [])
            if crosstalk:
                lines.append('**Pathway Crosstalk:**\n\n')
                for ct in crosstalk:
                    lines.append(f'- {ct}\n')
                lines.append('\n')

            # Functional consequences
            consequences = mechanism.get('functionalConsequences', '')
            if consequences:
                lines.append(f'**Functional Consequences:** {consequences}\n\n')

            lines.append('---\n\n')

        return ''.join(lines)

    def _synthesize_mechanisms_from_structures(
        self,
        pathway_structures: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Build minimal mechanism entries directly from retrieved structures.

        Used when Step 3's mechanism-interpretation LLM step yields no usable output
        (unparseable JSON, or every batch failed) but pathway structures WERE
        retrieved. Surfaces the curated/inferred gene involvement and relations from
        each structure so the report's Pathway Mechanisms section reflects the real
        retrieved data instead of rendering blank.

        Only genuinely differentially expressed genes (direction up/down) are listed in
        deGeneInvolvement — gene-set placeholder entries (direction 'unknown') are
        excluded, since labelling them as DE would be wrong and would trip the Step 3
        hallucination validator. `isDE` on each relation reflects whether BOTH endpoints
        are real DE genes rather than being assumed true.

        Args:
            pathway_structures: Retrieved structures in the internal (pre-serialization)
                shape, with mapped_de_genes / de_relations as dataclass objects.

        Returns:
            A list of mechanism dicts matching the shape _generate_report_section and the
            downstream consumers expect.
        """
        mechanisms = []
        for ps in pathway_structures:
            display_names = ps.get('display_names') or {}
            de_symbols = {
                _sym(g.gene_symbol) for g in ps.get('mapped_de_genes', [])
                if getattr(g, 'gene_symbol', '') and getattr(g, 'direction', '') in ('up', 'down')
            }
            de_gene_involvement = _dedupe_de_genes([
                {
                    'gene': g.gene_symbol,
                    'foldChange': g.fold_change,
                    'roleInPathway': ''
                }
                for g in ps.get('mapped_de_genes', [])
                if getattr(g, 'gene_symbol', '') and getattr(g, 'direction', '') in ('up', 'down')
            ])
            curated_relations = [
                {
                    'source': _display(r.source, display_names),
                    'target': _display(r.target, display_names),
                    'type': getattr(r, 'subtype', '') or getattr(r, 'type', ''),
                    'isDE': _sym(r.source) in de_symbols and _sym(r.target) in de_symbols,
                    'interpretation': ''
                }
                for r in ps.get('de_relations', [])
                if getattr(r, 'source', '') and getattr(r, 'target', '')
            ]
            mechanisms.append({
                'pathway': ps['pathway'],
                'biologicalFunction': '',
                'deGeneInvolvement': de_gene_involvement,
                'curatedRelations': curated_relations,
                'crosstalk': [],
                'functionalConsequences': ''
            })
        return mechanisms

    def _generate_summary_from_structures(self, mechanisms: List[Dict]) -> str:
        """Factual overview for the structure-derived fallback (no LLM interpretation).

        Deliberately avoids the interpretive claims of _generate_summary_from_mechanisms,
        since these entries come straight from the retrieved structures.

        Args:
            mechanisms: Structure-derived mechanism dicts.

        Returns:
            A plain factual summary string for the report Overview.
        """
        if not mechanisms:
            return 'No pathway mechanisms available.'
        pathway_names = [m.get('pathway', 'Unknown') for m in mechanisms[:5]]
        more = '' if len(mechanisms) <= 5 else f', and {len(mechanisms) - 5} more'
        return (
            f'Mechanism interpretation was unavailable; reporting {len(mechanisms)} '
            f'pathway(s) directly from retrieved structures (curated/inferred gene '
            f'relations), without additional LLM interpretation. '
            f'Pathways: {", ".join(pathway_names)}{more}.'
        )

    def _generate_empty_report(self) -> str:
        """Generate empty report when no pathways found"""
        # Was "...retrieved from KEGG for analysis." — retrieval is not KEGG-only.
        return '''## Pathway Mechanisms and Interactions

No pathway structures could be retrieved for analysis.
'''
