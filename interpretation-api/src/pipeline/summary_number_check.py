"""
Number check for the executive summary (main L263: "after the LLM responds, replaces any
incorrect number in the summary with the computed value").

The executive summary is the only LLM-written prose in the report. This module finds the
numbers it cites and compares them with the values the prompt gave the model:

- enrichment scores written with "NES" or "ES": correct if the sign and the value to two
  decimals are among the scores in the prompt. Otherwise, if exactly one prompt pathway
  matches the words before the number (loose word overlap, same sign), the value is
  replaced in place. A wrong sign or a number that cannot be placed needs a rewrite.
- ranges and approximate values ("NES ≈ 1.7–1.8") pass if they lie within the scores of
  one direction, with 0.05 slack.
- thresholds (after ≥, ≤, >, < or inside "|NES|") are skipped.
- fold changes, p-values and FDR values have no source in the prompt and need a rewrite.
- the hub-gene and hypothesis counts, often written as words, are replaced in place.

The three minus signs the model writes (hyphen, U+2011, U+2212) are all read as minus.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

# Characters the model uses as a minus sign before a number.
MINUS_CHARS = '-‑−‐'
_NUM = rf'[+{MINUS_CHARS}]?\d+(?:\.\d+)?'
_RANGE_SEP = r'\s*(?:[–—]|to|-)\s*'

# "|NES| >= 2", "NES ≈ 1.7–1.8", "NES = -2.50", "(NES +1.81)", "NES of 2.1", "ES: -0.62"
_SCORE_RE = re.compile(
    rf'(?P<pipe1>\|)?\b(?P<metric>NES|ES)\b(?P<pipe2>\|)?'
    rf'(?P<gap>\s*(?:score\s*)?(?:values?\s*)?(?:of\s*)?)'
    rf'(?P<cmp>>=|<=|≥|≤|>|<|=|≈|~|:|≈)?\s*'
    rf'(?P<v1>{_NUM})'
    rf'(?:(?P<sep>{_RANGE_SEP})(?P<v2>{_NUM}))?'
)

# A number after a comparator or with "|...|" is a threshold, not a claim.
_THRESHOLD_CMPS = {'>=', '<=', '≥', '≤', '>', '<'}
_APPROX_CMPS = {'≈', '~', '≈'}

# Fold changes, p-values and FDR values written as claims.
_STAT_RE = re.compile(
    rf'\b(?P<what>log2\s*FC|log2FC|logFC|FC|fold[\s\-‑]change|p[\s\-‑]?value|'
    rf'adjusted\s+p|FDR|q[\s\-‑]?value|p|q)\b'
    rf'\s*(?:of\s*)?(?P<cmp>>=|<=|≥|≤|>|<|=|≈|~|:)?\s*'
    rf'(?P<v>{_NUM}(?:\s*[e×x]\s*10?\^?[+{MINUS_CHARS}]?\d+)?(?:[eE][+{MINUS_CHARS}]?\d+)?)',
    re.IGNORECASE,
)

_NUMBER_WORDS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
    'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20,
}
_WORD_FOR = {v: k for k, v in _NUMBER_WORDS.items()}
# A count is a whole word right before "hub genes" / "hypotheses", optionally with a
# few describing adjectives in between. The earlier free "any three words" gap and the
# missing word boundaries read the 6 of "IL6", the "one" of "None" and the 2 of
# "Hypothesis 2 and the other hypotheses" as counts, and rewrote them.
# _COUNT_TOKEN = r'(?P<n>\d+|' + '|'.join(_NUMBER_WORDS) + r')'
# _HUB_COUNT_RE = re.compile(
#     _COUNT_TOKEN + r'(?P<mid>(?:\s+[\w‑\-]+){0,3}?)\s+hub\s+genes?\b', re.IGNORECASE)
# _HYP_COUNT_RE = re.compile(
#     _COUNT_TOKEN + r'(?P<mid>(?:\s+[\w‑\-]+){0,3}?)\s+hypothes(?:is|es)\b', re.IGNORECASE)
_COUNT_TOKEN = r'(?<![\w\-‑.])(?P<n>\d+|' + '|'.join(_NUMBER_WORDS) + r')\b'
_COUNT_ADJECTIVES = (r'(?:key|top|main|major|central|candidate|distinct|novel|ppi|network|'
                     r'highly|connected|highly[\-‑]connected|well[\-‑]connected|'
                     r'high[\-‑]connectivity|mechanistic|testable|identified|'
                     r'ranked|selected|prioritized|prioritised)')
_COUNT_MID = r'(?P<mid>(?:\s+' + _COUNT_ADJECTIVES + r'){0,3})'
_HUB_COUNT_RE = re.compile(_COUNT_TOKEN + _COUNT_MID + r'\s+hub\s+genes?\b', re.IGNORECASE)
_HYP_COUNT_RE = re.compile(_COUNT_TOKEN + _COUNT_MID + r'\s+hypothes(?:is|es)\b', re.IGNORECASE)

_STOP = {
    'the', 'and', 'of', 'in', 'to', 'a', 'an', 'by', 'with', 'for', 'pathway', 'pathways',
    'signaling', 'signalling', 'regulation', 'regulated', 'process', 'strong', 'moderate',
    'weak', 'enrichment', 'score', 'scores', 'down', 'up', 'nes', 'es', 'its', 'this',
}


def _to_float(token: str) -> float:
    t = token
    for ch in MINUS_CHARS:
        t = t.replace(ch, '-')
    return float(t)


def _words(text: str) -> set:
    text = text.lower()
    for ch in MINUS_CHARS:
        text = text.replace(ch, ' ')
    return {w for w in re.findall(r'[a-z0-9]+', text) if len(w) >= 3 and w not in _STOP}


def _clause_before(text: str, pos: int, width: int = 160) -> str:
    """The text before pos, back to the start of its sentence (or `width` chars)."""
    start = max(0, pos - width)
    window = text[start:pos]
    cut = max(window.rfind('. '), window.rfind('; '), window.rfind('\n'))
    return window[cut + 1:] if cut >= 0 else window


def _nearest_pathway(clause: str, scores: Sequence[Tuple[float, str]]):
    """The (value, name) whose name words end nearest to the end of `clause`.

    Returns None when no name matches or when two pathways tie for nearest.
    """
    lowered = clause.lower()
    for ch in MINUS_CHARS:
        lowered = lowered.replace(ch, ' ')
    tokens = [(t.group(0), t.end()) for t in re.finditer(r'[a-z0-9]+', lowered)]
    best, best_pos, tie = None, -1, False
    for value, name in scores:
        name_words = _words(name)
        if not name_words:
            continue
        positions = [end for tok, end in tokens if tok in name_words]
        if not positions:
            continue
        pos = max(positions)
        if pos > best_pos:
            best, best_pos, tie = (value, name), pos, False
        elif pos == best_pos:
            tie = True
    return None if tie else best


def _format_like(original: str, value: float) -> str:
    """Write value with the original's decimals, sign style and minus character."""
    decimals = len(original.split('.')[1]) if '.' in original else 0
    minus = next((ch for ch in original if ch in MINUS_CHARS), '-')
    body = f'{abs(value):.{decimals}f}'
    if value < 0:
        return minus + body
    return ('+' + body) if original.startswith('+') else body


def _count_word_like(original: str, n: int) -> str:
    if original.isdigit():
        return str(n)
    word = _WORD_FOR.get(n)
    if word is None:
        return str(n)
    if original[:1].isupper():
        word = word.capitalize()
    return word


def find_issues(
    summary: str,
    scores: Sequence[Tuple[float, str]],
    hub_counts: Sequence[int] = (),
    hypothesis_count: Optional[int] = None,
) -> Tuple[str, Dict]:
    """Check the numbers in `summary` against the prompt's values.

    Args:
        summary: the model's executive summary.
        scores: (value, pathway name) for every enrichment score given in the prompt.
        hub_counts: hub-gene counts the summary may state (e.g. shown hubs, all hubs).
        hypothesis_count: the number of hypotheses, or None to skip that check.

    Returns:
        (text with in-place replacements, report) where report has keys
        'checked', 'replaced' (list of {'from','to','context'}) and
        'needs_rewrite' (list of {'kind','text','reason'}).
    """
    report = {'checked': 0, 'replaced': [], 'needs_rewrite': []}
    allowed = {round(v, 2) for v, _ in scores}
    ups = [v for v, _ in scores if v > 0]
    downs = [v for v, _ in scores if v < 0]
    edits: List[Tuple[int, int, str]] = []

    for m in _SCORE_RE.finditer(summary):
        cmp_ = m.group('cmp') or ''
        if m.group('pipe1') or m.group('pipe2') or cmp_ in _THRESHOLD_CMPS:
            continue  # a threshold, e.g. "|NES| >= 2"
        report['checked'] += 1
        v1 = _to_float(m.group('v1'))
        v2 = _to_float(m.group('v2')) if m.group('v2') else None
        if v2 is not None or cmp_ in _APPROX_CMPS:
            lo, hi = (min(v1, v2), max(v1, v2)) if v2 is not None else (v1, v1)
            ok = False
            for group in (ups, downs):
                if not group:
                    continue
                g_lo, g_hi = min(group), max(group)
                # Unsigned approximate values may be written as magnitudes.
                for a, b in ((lo, hi), (-hi, -lo)):
                    if g_lo - 0.05 <= a and b <= g_hi + 0.05:
                        ok = True
            if not ok:
                report['needs_rewrite'].append({
                    'kind': 'score_range', 'text': m.group(0),
                    'reason': 'range or approximate value outside the prompt scores'})
            continue
        if round(v1, 2) in allowed:
            continue
        # Find the pathway this value belongs to: the prompt pathway whose name words occur
        # nearest before the number in the same clause. A sentence often names several
        # pathways ("histidine metabolism (NES -2.50) and primary bile-acid biosynthesis
        # (NES -2.12)"), so "any name in the clause" would pick the wrong one.
        nearest = _nearest_pathway(_clause_before(summary, m.start()), scores)
        if nearest is not None and (nearest[0] < 0) == (v1 < 0):
            new_val = _format_like(m.group('v1'), nearest[0])
            if new_val == m.group('v1'):
                continue  # correct at the printed precision ("+2.1" for 2.13): no change
            edits.append((m.start('v1'), m.end('v1'), new_val))
            report['replaced'].append({'from': m.group('v1'), 'to': new_val,
                                       'context': m.group(0), 'pathway': nearest[1]})
        elif nearest is not None:
            report['needs_rewrite'].append({
                'kind': 'score_sign', 'text': m.group(0),
                'reason': f'sign differs from {nearest[1]} ({nearest[0]:+.2f})'})
        else:
            report['needs_rewrite'].append({
                'kind': 'score_unplaced', 'text': m.group(0),
                'reason': 'value not among the prompt scores and no single pathway matches'})

    for m in _STAT_RE.finditer(summary):
        what = m.group('what')
        if (m.group('cmp') or '') in _THRESHOLD_CMPS:
            continue
        # A bare "p"/"q" only counts when followed by "=" (e.g. "p = 0.01").
        if what.lower() in ('p', 'q') and (m.group('cmp') or '') not in ('=', ':'):
            continue
        report['checked'] += 1
        report['needs_rewrite'].append({
            'kind': 'unsourced_stat', 'text': m.group(0),
            'reason': 'fold changes, p-values and FDR values are not in the summary prompt'})

    for regex, kind, valid in ((_HUB_COUNT_RE, 'hub_count', list(hub_counts)),
                               (_HYP_COUNT_RE, 'hypothesis_count',
                                [] if not hypothesis_count else [hypothesis_count])):
        # (zero hypotheses: nothing to check; "0 hypotheses" was written over the text)
        if not valid:
            continue
        for m in regex.finditer(summary):
            token = m.group('n')
            n = int(token) if token.isdigit() else _NUMBER_WORDS[token.lower()]
            report['checked'] += 1
            if n in valid:
                continue
            if len(set(valid)) > 1:
                # Several counts are valid (e.g. hubs shown vs all hubs): which one the
                # sentence means can't be told, so ask for a rewrite instead of guessing.
                report['needs_rewrite'].append({
                    'kind': kind, 'text': m.group(0),
                    'reason': f'count not among {sorted(set(valid))}'})
                continue
            new_tok = _count_word_like(token, valid[0])
            edits.append((m.start('n'), m.end('n'), new_tok))
            report['replaced'].append({'from': token, 'to': new_tok, 'context': m.group(0)})

    text = summary
    for start, end, new in sorted(edits, key=lambda e: e[0], reverse=True):
        text = text[:start] + new + text[end:]
    return text, report


def rewrite_instructions(issues: List[Dict], scores: Sequence[Tuple[float, str]]) -> str:
    """The user message for the one rewrite."""
    lines = ['The summary you wrote contains numbers that do not match the computed values:']
    for issue in issues:
        lines.append(f"- \"{issue['text']}\": {issue['reason']}")
    if scores:
        lines.append('')
        lines.append('The only enrichment scores you may cite are:')
        for v, nm in scores:
            lines.append(f'- {nm}: {v:+.2f}')
    else:
        lines.append('')
        lines.append('No enrichment scores were provided, so do not cite any.')
    lines.append('')
    lines.append('Do not cite fold changes, p-values or FDR values. Rewrite the summary with '
                 'the same structure and length, correcting only these numbers. '
                 'Return only the summary text.')
    return '\n'.join(lines)
