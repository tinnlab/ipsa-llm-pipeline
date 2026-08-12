"""Shared fold-change direction helpers.

Historically the pipeline used `'up' if fc > 0 else 'down'` everywhere, which
classifies a fold change of exactly 0 (or missing data) as "down" and renders it
with a ↓ arrow (e.g. the `MIR21 (FC: 0.00 ↓)` artifact). These helpers treat a
zero/absent fold change as neutral instead.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

# Fold changes with |value| below this are treated as no-change / no-data.
FC_EPSILON = 1e-9

# Shared NES (normalized enrichment score) magnitude thresholds, so effect-size
# calibration lives in one place instead of drifting across modules. NES is unbounded
# (typically ±1–4); classic ES is bounded [-1, 1] and uses its own bands. Used by
# step03's `_magnitude_label` and the clustering significance classifier.
NES_STRONG = 2.0      # |NES| >= this = strong signal
NES_MODERATE = 1.5    # |NES| in [this, NES_STRONG) = moderate signal


def to_float(x) -> Optional[float]:
    """Coerce a value (possibly a numeric string from the LLM) to float, else None."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fc_direction(fc: Optional[float], eps: float = FC_EPSILON) -> str:
    """Return 'up', 'down', or 'neutral' for a fold change (neutral if 0/None)."""
    if fc is None:
        return 'neutral'
    try:
        fc = float(fc)
    except (TypeError, ValueError):
        return 'neutral'
    if fc > eps:
        return 'up'
    if fc < -eps:
        return 'down'
    return 'neutral'


def fc_arrow(fc: Optional[float]) -> str:
    """Directional arrow for a fold change ('↑' / '↓' / '' for neutral)."""
    d = fc_direction(fc)
    return '↑' if d == 'up' else '↓' if d == 'down' else ''


def has_fc(fc: Optional[float]) -> bool:
    """True if `fc` carries a real (non-zero, present) fold change."""
    return fc_direction(fc) != 'neutral'


# Self-correction / meta-commentary the model sometimes leaks into otherwise-final
# prose (e.g. "Typo corrected: ALDH2 to AKR1A1", "I previously stated ..."). We match the
# *shape* of a self-correction clause — a known marker up to the next sentence break —
# rather than an ever-growing list of exact sentences, and only remove that clause.
_SELF_CORRECTION_RE = re.compile(
    r'(?:(?<=^)|(?<=[.;:,]))\s*'
    r'(?:'
    r'typos?\s+correct\w*'
    r'|correct(?:ed|ion)s?\s+(?:from\s+)?[A-Z0-9]+\s+to\s+[A-Z0-9]+'
    r'|I\s+(?:previously|earlier|mistakenly|incorrectly)\b'
    r'|let\s+me\s+(?:revise|correct|clarify|rephrase|restate)\b'
    r'|as\s+(?:I|noted|mentioned)\s+(?:previously|earlier|above)\b'
    r'|on\s+second\s+thought\b'
    r'|note\s+to\s+self\b'
    r')'
    # Consume to the end of the clause. A '.' or ';' terminates UNLESS it is a decimal
    # point (followed by a digit), so "1.5" inside a correction clause doesn't split it.
    r'[^;]*?(?:[.;](?!\d)|$)',
    re.IGNORECASE,
)


# Unresolved placeholders the model sometimes emits when it cannot source a value it
# started to cite — e.g. "CDCA8 up-regulated 2.72-fold (p=2.85e-?? not listed but hub
# score high)". These reach the PDF as scratch text. We strip the OFFENDING PARENTHETICAL
# (or a bare placeholder p-value token), keeping the sourced fact that precedes it. Markers
# are placeholder-specific — a VALUE-placeholder '?' (a run of "??", or "=?" for a missing
# assignment, so "e-??"/"=??" match) or an explicit "not listed / not reported / TBD" note.
# A SINGLE rhetorical/tentative '?' does NOT match, so "(HNF4A?)", "(p<0.05?)", "(see Fig
# 2e?)" and legitimate "(+2.72)", "(FC: +1.12)", "(^1.05)" are all left untouched. Phrase
# markers are \b-anchored so word tails like "top unknown" don't match "p unknown".
_PLACEHOLDER_MARKER = (
    r'(?:\?\?+|=\s*\?'                    # a VALUE-placeholder '?' ("??" run, or "=?")
    r'|not\s+listed|unlisted'
    r'|not\s+reported|not\s+provided|not\s+available'
    r'|\bvalue\s+unknown|\bp\s+unknown|\bfdr\s+unknown'
    r'|to\s+be\s+determined|\bTBD\b|placeholder'
    r')'
)
# A parenthetical whose contents include a placeholder marker.
_PLACEHOLDER_PAREN_RE = re.compile(
    r'\s*\([^()]*' + _PLACEHOLDER_MARKER + r'[^()]*\)',
    re.IGNORECASE,
)
# A bare (non-parenthesised) placeholder p-value token whose VALUE is missing, e.g.
# "p=2.85e-??" (exponent replaced) or "p = ?" — so it works even without surrounding
# parens. It matches only when the '?' stands in for the value/exponent: a bare '?+' right
# after '='/':', or a mantissa followed by an 'e' exponent-placeholder. A complete p-value
# with a trailing hedge '?' (e.g. "p=0.05?") is NOT a placeholder and is left untouched.
_PLACEHOLDER_PVALUE_RE = re.compile(
    r'[,;]?\s*\bp\s*[=:]\s*(?:\?+|[0-9.]*[eE][-+]?\?+)[^\s,;.()]*',
    re.IGNORECASE,
)


def sanitize_llm_text(text: Optional[str]) -> str:
    """Strip leaked self-correction / meta-commentary and unresolved placeholders from
    LLM prose, conservatively.

    Removes only clauses that clearly match a self-correction construction (see
    ``_SELF_CORRECTION_RE``) or a placeholder parenthetical / bare placeholder p-value
    (see ``_PLACEHOLDER_*_RE``); everything else is left untouched. Whitespace is
    re-normalized after removal. Non-strings return ''.
    """
    if not text or not isinstance(text, str):
        return text or ''
    cleaned = _SELF_CORRECTION_RE.sub(' ', text)
    cleaned = _PLACEHOLDER_PAREN_RE.sub('', cleaned)
    cleaned = _PLACEHOLDER_PVALUE_RE.sub('', cleaned)
    # A removed p-value token can leave the parenthetical that held it empty
    # (e.g. "(p=2.5e-?)" -> "()"); drop any now-empty parentheses.
    cleaned = re.sub(r'\s*\(\s*\)', '', cleaned)
    # Re-normalize whitespace and tidy punctuation left dangling by a removed clause.
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'\s+([.;,])', r'\1', cleaned)
    cleaned = re.sub(r'([:;,]\s*)+$', '', cleaned).strip()
    return cleaned


def gene_symbol(gene: Dict) -> str:
    """Extract the gene symbol (UPPER) from a DE gene dict, tolerant of field names."""
    return (gene.get('geneSymbol') or gene.get('gene_symbol') or
            gene.get('gene') or gene.get('name') or '').upper()


def gene_fold_change(gene: Dict) -> Optional[float]:
    """Signed fold change from a DE gene dict, or None if absent/unparseable.

    Tries ``foldChange`` / ``log2_fold_change`` / ``fold_change`` in order via
    :func:`to_float` (None-checked, NOT ``or``-chained) so a real fold change of exactly
    0.0 is preserved instead of being treated as falsy and skipped to a later field.
    """
    for key in ('foldChange', 'log2_fold_change', 'fold_change'):
        fc = to_float(gene.get(key))
        if fc is not None:
            return fc
    return None


def gene_fc_lookup(genes: List[Dict]) -> Dict[str, float]:
    """Build an UPPER gene-symbol -> signed fold-change map from a DE gene list."""
    lut: Dict[str, float] = {}
    for g in genes or []:
        sym = gene_symbol(g)
        fc = gene_fold_change(g)
        if sym and fc is not None:
            lut[sym] = fc
    return lut


# A gene symbol immediately followed by an EXPLICITLY SIGNED decimal fold-change, e.g.
# "HEATR1 +1.15", "NOP56 +1.20-fold", "FBL (+1.24)", "NHP2 (FC: +1.12)". We only touch
# explicitly-signed citations: the sign makes the cited direction unambiguous, so a
# magnitude-only error can be fixed safely while a sign error is detected rather than
# silently "corrected". Unsigned / arrow-only forms ("↑3.6") are left to the direction
# validators. Membership in the DE table (checked in the replacer) filters false hits.
_SIGNED_FC_CITATION_RE = re.compile(
    r'(?P<gene>\b[A-Z][A-Z0-9]{1,14}\b)'
    # Require a real separator (whitespace or an opening paren) between the gene and the
    # signed number, so a hyphenated symbol like "IL-6" is NOT parsed as gene "IL" = -6.
    r'(?P<pre>(?:\s+|\s*\(\s*)(?:FC\s*[:=]\s*)?)'
    r'(?P<sign>[+\-−])'
    r'(?P<num>\d+(?:\.\d+)?)'
    # Not part of a longer number / scientific notation ("1.2e-3", "1.24.5").
    r'(?![\d.eE])'
    r'(?P<fold>\s*[\-‑]?\s*fold)?'
    r'(?P<post>\s*\)?)'
)


def correct_fc_citations(
    text: str,
    gene_fc: Dict[str, float],
) -> Tuple[str, Set[str]]:
    """Correct signed fold-change citations in prose against the canonical DE table.

    Args:
        text: prose that may cite fold changes as "GENE +X.XX".
        gene_fc: UPPER gene symbol -> signed fold change (see :func:`gene_fc_lookup`).

    Returns:
        ``(corrected_text, conflict_genes)``. Same-sign magnitude errors are rewritten in
        place. A citation whose SIGN contradicts the DE table (a direction-inverting error,
        not a typo) is left untouched and its gene reported in ``conflict_genes`` — the
        caller resolves those (regenerate, or drop the sentence via
        :func:`drop_sentences_with_fc_conflicts`), since silently renumbering would leave
        the governing direction word wrong.
    """
    if not text or not gene_fc:
        return text, set()

    conflicts: Set[str] = set()

    def _repl(m: 're.Match') -> str:
        sym = m.group('gene').upper()
        if sym not in gene_fc:
            return m.group(0)
        true_fc = gene_fc[sym]
        cited = to_float(m.group('sign').replace('−', '-') + m.group('num'))
        if cited is None:
            return m.group(0)
        same_sign = (cited >= 0) == (true_fc >= 0)
        if same_sign:
            if abs(abs(cited) - abs(true_fc)) < 0.005:
                return m.group(0)  # already correct to 2 decimals
            # Rewrite magnitude in place, preserving surrounding punctuation/"-fold".
            return (f"{m.group('gene')}{m.group('pre')}{true_fc:+.2f}"
                    f"{m.group('fold') or ''}{m.group('post')}")
        # Sign conflict: record it, leave the text for the caller to resolve.
        conflicts.add(sym)
        return m.group(0)

    return _SIGNED_FC_CITATION_RE.sub(_repl, text), conflicts


def drop_sentences_with_fc_conflicts(text: str, gene_fc: Dict[str, float]) -> str:
    """Remove whole sentences that cite a gene with a sign contradicting the DE table.

    Last-resort guard for the rare case where regeneration fails to resolve a
    sign-inverted fold-change citation: dropping just the number would leave the
    governing direction word (e.g. "increased") still wrong, so we drop the entire
    offending sentence. Sentences with no conflict are preserved verbatim.
    """
    if not text or not gene_fc:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    kept = [s for s in sentences if not correct_fc_citations(s, gene_fc)[1]]
    return ' '.join(kept).strip()
