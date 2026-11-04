"""
Regulon (TF -> target) enrichment service.

Infers candidate upstream transcription-factor regulators of a gene set by testing which
TF's target set (its *regulon*) is over-represented among the query genes, using a
one-sided hypergeometric test with Benjamini-Hochberg FDR correction.

Motivation: master regulators of a down-regulated program (e.g. the hepatocyte metabolic
TFs HNF4A / PPARA / PPARGC1A / CEBPA) are frequently NOT differentially expressed and are
NOT reachable through KEGG gene-expression (``GErel``) edges inside enriched metabolic
maps — those maps encode enzyme<->enzyme (``ECrel``) relations, not transcriptional edges.
A curated TF->target regulon database *is* built to know "HNF4A controls these metabolic
genes", so testing regulon overlap recovers them.

Uses a CollecTRI human regulon at ``data/collectri_human.tsv``. That file is NOT shipped
with the repository — see ``data/README.md`` for how to fetch it. The loader is tolerant
of a missing/malformed file: it degrades to ``available == False`` so callers fall back to
an LLM-proposed candidate list, clearly labelled as hypotheses. Running without the file
is a supported configuration, not an error path.

No scipy dependency: the hypergeometric survival function is evaluated in log-space with
``math.lgamma`` (the deps are deliberately minimal — see requirements.txt).
"""

from math import lgamma, exp
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Iterable
import logging

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / 'data' / 'collectri_human.tsv'


def _log_choose(a: int, b: int) -> float:
    """log(C(a, b)) via lgamma; -inf for out-of-range (so its exp is 0)."""
    if a < 0 or b < 0 or b > a:
        return float('-inf')
    return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)


def hypergeom_sf(k: int, N: int, K: int, n: int) -> float:
    """One-sided hypergeometric survival ``P(X >= k)``.

    Probability of drawing at least ``k`` successes when ``n`` items are drawn without
    replacement from a population of ``N`` items containing ``K`` successes. Evaluated in
    log-space with log-sum-exp for stability; returns a value in ``[0, 1]``. No scipy.
    """
    if k <= 0:
        return 1.0
    lo, hi = k, min(K, n)
    if lo > hi or N <= 0 or n <= 0:
        return 0.0
    denom = _log_choose(N, n)
    if denom == float('-inf'):
        return 0.0
    terms = [_log_choose(K, i) + _log_choose(N - K, n - i) - denom
             for i in range(lo, hi + 1)]
    m = max(terms)
    if m == float('-inf'):
        return 0.0
    return min(1.0, exp(m) * sum(exp(t - m) for t in terms))


def bh_fdr(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR (q-values), returned in the input order.

    Enforces monotonicity from the largest p-value down and clamps to ``[0, 1]``.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])   # ascending p-value
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):                        # largest p (rank m) -> smallest
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        q[i] = min(prev, 1.0)
    return q


def _activity_from_net(net_sign: int, signed_overlap: int) -> str:
    """Map the signed activity tally to an inferred-activity label.

    Each overlapping target contributes ``edge_sign * dir(target)`` with ``dir(down) = -1``:
    an activator (+1) of a down-regulated target => TF activity likely LOST (decreased);
    a repressor (-1) of a down-regulated target => TF likely ACTIVE (increased). Unsigned
    regulons (all weights 0) can't infer a direction.
    """
    if signed_overlap == 0:
        return 'unknown'
    if net_sign > 0:
        return 'increased'
    if net_sign < 0:
        return 'decreased'
    return 'mixed'


class RegulonService:
    """Loads a TF->target regulon and ranks candidate upstream regulators by enrichment."""

    def __init__(self, path: Optional[str] = None):
        self._regulon: Dict[str, Dict[str, int]] = {}   # TF -> {target: weight}
        self._universe: Set[str] = set()
        self._load(_DEFAULT_PATH if path is None else path)

    # -- construction ------------------------------------------------------------------

    @classmethod
    def from_edges(cls, edges: Iterable[Tuple[str, str, Optional[int]]]) -> 'RegulonService':
        """Build a service from in-memory ``(tf, target, weight)`` edges (for tests)."""
        svc = cls.__new__(cls)
        svc._regulon = {}
        svc._universe = set()
        svc._ingest(edges)
        return svc

    def _ingest(self, edges: Iterable[Tuple[str, str, Optional[int]]]) -> None:
        for s, t, w in edges:
            s = (s or '').strip().upper()
            t = (t or '').strip().upper()
            if not s or not t or s == t:
                continue
            self._regulon.setdefault(s, {})[t] = int(w) if w not in (None, '') else 0
            self._universe.add(s)
            self._universe.add(t)

    def _load(self, path) -> None:
        try:
            p = Path(path)
            if not p.exists():
                logger.warning(f'Regulon file not found: {p}; upstream-regulator DB unavailable')
                return
            edges: List[Tuple[str, str, Optional[int]]] = []
            with open(p) as f:
                first = f.readline().rstrip('\n').split('\t')
                # Header detection keys off the literal column name (a real TF symbol is
                # never "source"/"tf"/"regulator"), NOT the weight's numeracy — so a
                # headerless first row with a non-numeric weight like "NA" is still ingested.
                col0 = (first[0] or '').strip().lower() if first else ''
                is_header = col0 in ('source', 'tf', 'regulator')
                if not is_header and len(first) >= 2:
                    edges.append((first[0], first[1],
                                  _parse_weight(first[2] if len(first) > 2 else None)))
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 2:
                        continue
                    edges.append((parts[0], parts[1],
                                  _parse_weight(parts[2] if len(parts) > 2 else None)))
            self._ingest(edges)
            logger.info(f'Loaded regulon: {len(self._regulon)} TFs, {len(self._universe)} genes')
        except Exception as e:  # never let a bad data file crash the pipeline
            logger.warning(f'Failed to load regulon ({e}); upstream-regulator DB unavailable')
            self._regulon = {}
            self._universe = set()

    # -- API ---------------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return len(self._regulon) > 0

    def universe(self) -> Set[str]:
        return set(self._universe)

    def rank_tfs(
        self,
        down_genes: Iterable[str],
        background_n: Optional[int] = None,
        min_overlap: int = 3,
        fdr_threshold: float = 0.10,
        top_n: int = 8,
    ) -> List[Dict]:
        """Rank candidate TFs whose regulon is enriched among ``down_genes``.

        Args:
            down_genes: query gene symbols (the down-regulated DE set).
            background_n: statistical universe size ``N``; defaults to the regulon
                gene-space size (standard ORA "annotated background"). Because callers
                pass a pre-filtered significant-DE set rather than the full measured
                universe, absolute p-values are optimistic — results are ranked
                candidates gated by BH-FDR, not strict frequentist claims.
            min_overlap: minimum overlapping targets for a TF to be tested (also caps how
                many TFs enter the BH correction, so it isn't over-diluted).
            fdr_threshold: keep only TFs with BH q-value <= this.
            top_n: cap on returned candidates.

        Returns: list of result dicts sorted by (fdr, -overlap, tf).
        """
        if not self.available:
            return []
        universe = self._universe
        down = {(g or '').strip().upper() for g in down_genes} & universe
        n = len(down)
        # Background universe size N. The regulon gene space is the SMALLEST sensible
        # background (a real measured universe is larger), so never let an explicit
        # background_n drop below it — a too-small N makes p-values spuriously tiny.
        N = len(universe) if background_n is None else max(int(background_n), len(universe))
        if n == 0 or N <= 0:
            return []

        candidates: List[Dict] = []
        pvals: List[float] = []
        for tf, targets in self._regulon.items():
            reg_targets = set(targets.keys()) & universe
            K = len(reg_targets)
            if K == 0:
                continue
            overlap = reg_targets & down
            k = len(overlap)
            if k < min_overlap:
                continue
            p = hypergeom_sf(k, N, K, n)
            net, signed = 0, 0
            for g in overlap:
                w = targets.get(g, 0)
                if w > 0:
                    net -= 1
                    signed += 1
                elif w < 0:
                    net += 1
                    signed += 1
            candidates.append({
                'tf': tf,
                'targets': sorted(overlap),
                'overlap_count': k,
                'regulon_size': K,
                'p_value': p,
                'net_sign': net,
                'inferred_tf_activity': _activity_from_net(net, signed),
            })
            pvals.append(p)

        if not candidates:
            return []
        for c, q in zip(candidates, bh_fdr(pvals)):
            c['enrichment_fdr'] = q
        kept = [c for c in candidates if c['enrichment_fdr'] <= fdr_threshold]
        kept.sort(key=lambda c: (c['enrichment_fdr'], -c['overlap_count'], c['tf']))
        return kept[:top_n]


def _parse_weight(s) -> Optional[int]:
    if s is None or str(s).strip() in ('', 'None', 'NA', 'nan'):
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None
