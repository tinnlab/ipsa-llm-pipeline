"""
Deterministic checks on Step 4 hypotheses (items 12 and 13).

Main L252: "After the model responds, the engine replaces any incorrect fold changes and
p-values with the computed values. When a hypothesis describes a gene in the direction
opposite to its fold change, or lists a relation that the tool calls did not return, the
engine asks the LLM to rewrite the hypothesis once and removes the hypothesis if the
rewritten version still has the error."

- strict_direction_errors: a gene described in the direction opposite to its fold change.
  Only short fixed shapes with the direction word next to the gene are read ("up-regulated
  GENE", "up-regulation of GENE", "GENE is up-regulated", "GENE up-regulation", "increased
  GENE expression"); gene symbols match with case; U+2010/U+2011/U+2013 count as hyphens.
- loose_direction_matches: the former wide pattern, kept only as a recorded warning.
- correct_arrow_fc: "GENE (↑1.22)" citations. A wrong magnitude with the right arrow is
  corrected in place; a wrong arrow is a direction error.
- relation_claim_errors: a mechanistic claim (gene -> gene) that is not among the
  relations the tool calls returned.
"""

import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

_DASHES = str.maketrans({'‐': '-', '‑': '-', '–': '-', '‒': '-'})

_UP_ADJ = r'(?:up-?regulated|over-?expressed|increased|elevated|induced|higher)'
_DOWN_ADJ = r'(?:down-?regulated|under-?expressed|decreased|reduced|suppressed|lower|repressed)'
_UP_NOUN = r'(?:up-?regulation|over-?expression|increased expression|elevated expression|induction)'
_DOWN_NOUN = r'(?:down-?regulation|under-?expression|decreased expression|reduced expression|loss of expression|repression)'
# Optional fold-change parenthetical between gene and word: "CDK1 (↑2.41) is up-regulated"
_PAREN = r'(?:\s*\([^()]{0,25}\))?'
_COPULA = r'(?:\s+(?:is|are|was|were|being|becomes|remains|appears))?'
_COPULA_REQ = r'\s+(?:is|are|was|were|being|becomes|remains|appears)'
_ADVERB = r'(?:\s+(?:strongly|markedly|significantly|highly|robustly|modestly|slightly))?'
# Without "is/are", "GENE induced ..." is usually an active verb with the gene as subject
# ("IFNG induced CXCL10 expression", "SOD2 reduced ROS levels"), which claims nothing
# about the gene's own direction. Read it as a direction only when the word ends the
# phrase ("CDK1 up-regulated in tumours", "CDK1 up-regulated,").
_PHRASE_END = (r'(?=\s*(?:$|[,.;:)\]]|\s(?:in|by|at|upon|after|during|across|relative|'
               r'compared|versus|vs\.?|and|or|while|whereas|with)\b))')


def normalize_dashes(text: str) -> str:
    return (text or '').translate(_DASHES)


def _shapes(gene: str):
    g = re.escape(gene)
    gb = rf'(?<![\w-]){g}(?![\w-])'
    for direction, adj, noun in (('up', _UP_ADJ, _UP_NOUN), ('down', _DOWN_ADJ, _DOWN_NOUN)):
        yield direction, re.compile(rf'\b{adj}\s+{gb}')                                  # up-regulated GENE
        yield direction, re.compile(rf'\b{noun}\s+of\s+{gb}')                             # up-regulation of GENE
        # yield direction, re.compile(rf'{gb}{_PAREN}{_COPULA}{_ADVERB}\s+{adj}\b')  # read active verbs
        yield direction, re.compile(rf'{gb}{_PAREN}{_COPULA_REQ}{_ADVERB}\s+{adj}\b')    # GENE is up-regulated
        yield direction, re.compile(rf'{gb}{_PAREN}{_ADVERB}\s+{adj}{_PHRASE_END}')      # GENE up-regulated in ...
        yield direction, re.compile(rf'{gb}{_PAREN}\s+{noun}\b')                          # GENE up-regulation


def strict_direction_errors(text: str, gene_fc: Dict[str, float]) -> List[Dict]:
    """Genes described in the direction opposite to their fold change (strict shapes).

    gene_fc maps gene symbols AS WRITTEN (case kept) to signed fold changes.
    """
    norm = normalize_dashes(text)
    errors, seen = [], set()
    for gene, fc in gene_fc.items():
        if not gene or fc is None or fc == 0 or gene not in norm:
            continue
        true_dir = 'up' if fc > 0 else 'down'
        for claimed, rx in _shapes(gene):
            m = rx.search(norm)
            if m and claimed != true_dir and (gene, claimed) not in seen:
                seen.add((gene, claimed))
                errors.append({'gene': gene, 'claimed': claimed, 'fold_change': fc,
                               'phrase': m.group(0)})
    return errors


_LOOSE_RE = r'(\w+[\s\-]*regulat\w+|overexpress\w+|enhanced|increased|elevated|reduced|decreased|down[\s\-]*regulat\w+|up[\s\-]*regulat\w+|loss|suppression)[\s\w,]*\b{gene}\b'


def loose_direction_matches(text: str, gene_fc: Dict[str, float]) -> List[Dict]:
    """The former wide pattern (farthest word, any case), for a recorded warning only."""
    norm = normalize_dashes(text)
    out = []
    for gene, fc in gene_fc.items():
        if not gene or not fc:
            continue
        true_dir = 'up' if fc > 0 else 'down'
        for ctx in re.findall(_LOOSE_RE.format(gene=re.escape(gene)), norm, re.IGNORECASE):
            c = ctx.lower()
            up = any(k in c for k in ('up-regulat', 'upregulat', 'overexpress', 'enhanced', 'increased', 'elevated'))
            down = any(k in c for k in ('down-regulat', 'downregulat', 'reduced', 'decreased', 'loss', 'suppression'))
            if (true_dir == 'down' and up) or (true_dir == 'up' and down):
                out.append({'gene': gene, 'claimed': 'up' if up else 'down', 'fold_change': fc})
                break
    return out


def correct_arrow_fc(text: str, gene_fc: Dict[str, float]) -> Tuple[str, Set[str]]:
    """Correct arrow fold-change citations: "GENE (↑1.22)", "GENE ↑1.22", "GENE (FC ↓0.85)".

    Returns (text, conflicts): a wrong magnitude with the right arrow is rewritten with the
    computed |FC| at the printed precision; a wrong arrow is returned as a conflict.
    """
    if not text or not gene_fc:
        return text, set()
    conflicts: Set[str] = set()
    for gene, fc in gene_fc.items():
        if not gene or fc is None or gene not in text:
            continue
        # A full stop after the value is a sentence end, not more digits ("CDK1 ↑2.50.").
        # rx = ...(?P<v>\d+(?:\.\d+)?)(?![\d.])')  # rejected "↑2.50." at a sentence end
        rx = re.compile(rf'(?<![\w-])(?P<g>{re.escape(gene)})(?P<pre>\s*\(?\s*(?:FC\s*[:=]?\s*)?)'
                        rf'(?P<arrow>[↑↓])\s*(?P<v>\d+(?:\.\d+)?)(?!\d|\.\d)')

        def repl(m):
            right_arrow = (m.group('arrow') == '↑') == (fc > 0)
            if not right_arrow:
                conflicts.add(gene)
                return m.group(0)
            v = m.group('v')
            decimals = len(v.split('.')[1]) if '.' in v else 0
            true_v = f'{abs(fc):.{decimals}f}'
            if v == true_v:
                return m.group(0)
            return f"{m.group('g')}{m.group('pre')}{m.group('arrow')}{true_v}"

        text = rx.sub(repl, text)
    return text, conflicts


def hypothesis_direction_errors(hyp: Dict, gene_fc: Dict[str, float]) -> List[Dict]:
    """Strict direction errors and wrong arrows in a hypothesis's statement, mechanistic
    model and key players (not directionalPrediction or testability)."""
    texts = [hyp.get('hypothesis', ''), hyp.get('mechanisticModel', '')]
    players = hyp.get('keyPlayers')
    if isinstance(players, list):
        texts += [p for p in players if isinstance(p, str)]
    errors = []
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        errors += strict_direction_errors(t, gene_fc)
        _, wrong_arrows = correct_arrow_fc(t, gene_fc)
        errors += [{'gene': g, 'claimed': 'down' if gene_fc[g] > 0 else 'up',
                    'fold_change': gene_fc[g], 'phrase': 'arrow'} for g in sorted(wrong_arrows)]
    uniq, seen = [], set()
    for e in errors:
        key = (e['gene'], e['claimed'])
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    return uniq


# ----------------------------------------------------------- printed-precision stats (item 25)

_MINUS = '-‑−‐'


def parse_stat(token: str) -> Optional[float]:
    """Parse "2.85e-12", "1.5 × 10^-3", "0.0012", "−2.50" into a float."""
    t = token.strip().replace(' ', '')
    for ch in _MINUS:
        t = t.replace(ch, '-')
    m = re.fullmatch(r'(\d+(?:\.\d+)?)[×xX*]10\^?(-?\d+)', t)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    try:
        return float(t)
    except ValueError:
        return None


def _sig_digits(token: str) -> int:
    t = token.strip().replace(' ', '')
    mant = re.split(r'[eE×xX*]', t)[0].lstrip('+').lstrip(_MINUS)
    digits = mant.replace('.', '').lstrip('0')
    return max(1, len(digits))


def differs_at_printed_precision(token: str, actual: float) -> bool:
    """True if `actual`, rounded to the significant digits the text printed, differs
    from the printed value (e.g. "2.9e-12" vs 2.85e-12 -> False; "3.1e-12" -> True)."""
    claimed = parse_stat(token)
    if claimed is None or actual is None:
        return False
    if claimed == 0 or actual == 0:
        return claimed != actual
    sig = _sig_digits(token)
    return float(f'{actual:.{sig - 1}e}') != float(f'{claimed:.{sig - 1}e}')


_STAT_AFTER_NAME = re.compile(
    r'(?P<label>\b(?:NES|ES|FDR|q|adj(?:usted)?\.?\s*p(?:-value)?)\b\s*(?:of\s*)?[=:≈]?\s*)'
    r'(?P<v>[+\-‑−]?\d+(?:\.\d+)?(?:\s*[eE][+\-‑−]?\d+|\s*[×xX*]\s*10\s*\^?\s*[\-‑−]?\s*\d+)?)',
    re.IGNORECASE)


_ADJACENT_GAP = re.compile(
    r'^[\s,:;=()\[\]\-–—]*(?:(?:with|has|had|have|shows?|showed|at|of|an?|its|and|'
    r'was|is|were|are|pathway|enrichment|signature|gene\s+set|program|programme|module|'
    r'score)\s*[\s,:;=()\[\]\-–—]*){0,3}$', re.IGNORECASE)


def correct_pathway_stats(text: str, pathway_stats: Dict[str, Dict[str, float]],
                          window: int = 60) -> Tuple[str, int]:
    """Correct NES and FDR values cited right after a pathway name (item 25).

    pathway_stats: {pathway name: {'nes': float|None, 'fdr': float|None}}. Only a value
    within `window` characters after the name, before the next sentence end, is read.
    Longer names win over names they contain ("Ribosome biogenesis..." over "Ribosome").
    Returns (text, number of replacements).
    """
    if not text or not pathway_stats:
        return text, 0
    spans = []
    for name in sorted(pathway_stats, key=len, reverse=True):
        # Whole words only, and not followed by more title words of a longer pathway
        # ("Ribosome" inside "Ribosome biogenesis in eukaryotes").
        # for m in re.finditer(re.escape(name), text, re.IGNORECASE):  # matched inside words
        for m in re.finditer(r'(?<![\w-])' + re.escape(name) + r'(?![\w-])', text, re.IGNORECASE):
            if any(s <= m.start() < e for s, e, _ in spans):
                continue
            if re.match(r'\s+(?:biogenesis|signaling|signalling|metabolism|'
                        r'biosynthesis|degradation)\b', text[m.end():], re.IGNORECASE) \
                    and not re.search(r'(?:pathway|metabolism|biosynthesis|degradation|'
                                      r'signaling|signalling|biogenesis)\s*$', name, re.IGNORECASE):
                continue
            spans.append((m.start(), m.end(), name))
    edits = []
    for start, end, name in spans:
        tail = text[end:end + window]
        cut = re.search(r'[.;]\s', tail)
        if cut:
            tail = tail[:cut.start()]
        stats = pathway_stats[name]
        pos = 0
        for m in _STAT_AFTER_NAME.finditer(tail):
            # Only values attached to this name: between the name (or the previous value)
            # and the label there may be brackets, punctuation and a linking word, not
            # other words ("... and the p53 pathway has NES -1.2" belongs to p53).
            gap = tail[pos:m.start()]
            if not _ADJACENT_GAP.match(gap):
                break
            pos = m.end()
            label = m.group('label').strip().lower()
            # key = 'nes' if label.startswith(('nes', 'es')) else 'fdr'  # ES read as NES
            if label.startswith(('nes', 'es')):
                key = 'nes'
                metric = (stats.get('metric') or '').upper() or None
                if metric is None and stats.get('nes') is not None and abs(stats['nes']) > 1.0:
                    metric = 'NES'  # |value| > 1 can only be NES (classic ES is in [-1, 1])
                cited = 'NES' if label.startswith('nes') else 'ES'
                if metric is not None and metric != cited:
                    continue  # the text cites the other score: leave it alone
            else:
                key = 'fdr'
            actual = stats.get(key)
            if actual is None:
                continue
            token = m.group('v')
            if key == 'nes':
                claimed = parse_stat(token)
                decimals = len(token.split('.')[1]) if '.' in token else 0
                if claimed is None or round(claimed, decimals) == round(actual, decimals):
                    continue
                new = f'{actual:+.{decimals}f}' if token.lstrip()[:1] in '+-‑−' else f'{actual:.{decimals}f}'
            else:
                if not differs_at_printed_precision(token, actual):
                    continue
                new = f'{actual:.2e}'
            s = end + m.start('v')
            edits.append((s, s + len(token), new))
    for s, e, new in sorted(edits, reverse=True):
        text = text[:s] + new + text[e:]
    return text, len(edits)


# ----------------------------------------------------------------- relation claims (item 13)

_UNDIRECTED_TYPES = {'binding', 'binding/association', 'association', 'complex',
                     'in-complex-with', 'interacts-with', 'dissociation', 'compound'}


def relation_reference(relations: Iterable[Dict]) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """(directed pairs, undirected pairs) from relations the tool calls returned."""
    directed, undirected = set(), set()
    for r in relations or []:
        s, t = str(r.get('source', '')).strip().upper(), str(r.get('target', '')).strip().upper()
        if not s or not t:
            continue
        rtype = str(r.get('type', '') or '').strip().lower()
        if rtype in _UNDIRECTED_TYPES:
            undirected.add((s, t))
        else:
            directed.add((s, t))
    return directed, undirected


_GENE_SHAPE = re.compile(r'[A-Z][A-Z0-9\-]{1,14}')


def relation_claim_errors(hyp: Dict, reference) -> List[Dict]:
    """Mechanistic claims of a hypothesis that are not among the returned relations.

    Every claim is checked; none is exempt. A claim whose ends are not both gene symbols
    fails (the schema asks for gene-to-gene claims only). A gene-to-gene claim passes if
    its pair was returned: directed relations in their direction, undirected relation
    types in either direction. With an empty reference every claim fails.
    """
    directed, undirected = reference
    errors = []
    for claim in hyp.get('mechanisticClaims') or []:
        if not isinstance(claim, dict):
            continue
        s = str(claim.get('source', '')).strip().upper()
        t = str(claim.get('target', '')).strip().upper()
        if not _GENE_SHAPE.fullmatch(s) or not _GENE_SHAPE.fullmatch(t):
            errors.append({'source': claim.get('source'), 'target': claim.get('target'),
                           'relation': claim.get('relation'), 'reason': 'not gene-to-gene'})
            continue
        if (s, t) in directed or (s, t) in undirected or (t, s) in undirected:
            continue
        errors.append({'source': claim.get('source'), 'target': claim.get('target'),
                       'relation': claim.get('relation'), 'reason': 'not returned by the tool calls'})
    return errors
