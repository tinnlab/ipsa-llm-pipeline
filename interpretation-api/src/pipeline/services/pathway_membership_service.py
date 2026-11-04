"""
Full gene membership for non-KEGG pathways from bundled GMT files (item 19).

Main L236: "For all other pathways (GO, Reactome, and custom pathways), it retrieves the
interactions for all pathway genes from the Pathway Commons database." The website sends
only a pathway's DE members (or nothing, on the meta path), so the engine looks the full
membership up in the GMT files under ``interpretation-api/data/gmt/``.

Matching is deliberately conservative, because GMT set names collide with common pathway
names ("Translation", "Apoptosis", "TCA cycle", "ABC transporters"):

- a pathway's ``source`` (an opaque gene-set database id for custom sets) is associated
  with a GMT file only when most of that source's pathways in the request are named
  sets of that file (``SOURCE_MATCH_FRACTION``, at least ``SOURCE_MIN_PATHWAYS``);
- a pathway of an associated source is then matched by pathway id, else by name
  (case-insensitive), and only if the genes the website sent are all members of the set.

Pathways that match nothing keep the genes the website sent.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional

GMT_DIR = Path(__file__).resolve().parents[3] / 'data' / 'gmt'
SOURCE_MATCH_FRACTION = 0.8
SOURCE_MIN_PATHWAYS = 3


def parse_gmt(text: str) -> Dict[str, List[str]]:
    """{set name -> genes}. Column 1 is the name, column 2 a description, then genes.

    The description is also indexed when it differs (MitoPathways puts the full
    hierarchy path there, e.g. "OXPHOS > Complex I").
    """
    sets: Dict[str, List[str]] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.rstrip('\n').split('\t')]
        if len(parts) < 3 or not parts[0]:
            continue
        genes = []
        for g in parts[2:]:
            if g and g not in genes:
                genes.append(g)
        sets.setdefault(parts[0], genes)
        if parts[1] and parts[1] != parts[0]:
            sets.setdefault(parts[1], genes)
    return sets


def _key(name: str) -> str:
    return ' '.join(str(name or '').split()).lower()


# A GMT file declares its organism by its name prefix ("Mouse.MitoPathways3.0.gmt").
_FILE_ORGANISMS = {
    'mouse': 'mus musculus',
    'human': 'homo sapiens',
    'rat': 'rattus norvegicus',
}


def file_organism(fname: str) -> Optional[str]:
    """Organism a GMT file is for, from its name prefix; None when it does not say."""
    prefix = str(fname or '').split('.', 1)[0].strip().lower()
    return _FILE_ORGANISMS.get(prefix)


class PathwayMembershipService:
    """Looks up full pathway membership in bundled GMT files."""

    def __init__(self, gmt_dir: Optional[Path] = None):
        self.files: Dict[str, Dict[str, List[str]]] = {}
        directory = Path(gmt_dir) if gmt_dir else GMT_DIR
        if directory.is_dir():
            for path in sorted(directory.glob('*.gmt')):
                try:
                    self.files[path.name] = {
                        _key(k): v for k, v in parse_gmt(path.read_text()).items()}
                except OSError:
                    continue
        self._source_file: Dict[str, str] = {}

    def associate_sources(self, pathways: Iterable[Dict],
                          organism: Optional[str] = None) -> Dict[str, str]:
        """Associate each pathway `source` with the GMT file most of its pathways name.

        A file whose name declares another organism than `organism` is never used (a
        human MitoPathways run must not get the mouse set's members and casing).
        Returns {source: gmt file name} and remembers it for `members`.
        """
        wanted = str(organism or '').strip().lower() or None
        files = {f: s for f, s in self.files.items()
                 if not (wanted and file_organism(f) and file_organism(f) != wanted)}
        by_source: Dict[str, List[Dict]] = {}
        for p in pathways or []:
            by_source.setdefault(str(p.get('source') or ''), []).append(p)
        self._source_file = {}
        for source, plist in by_source.items():
            if len(plist) < SOURCE_MIN_PATHWAYS:
                continue
            best, best_frac = None, 0.0
            for fname, sets in files.items():
                hits = sum(1 for p in plist if self._lookup(sets, p) is not None)
                frac = hits / len(plist)
                if frac > best_frac:
                    best, best_frac = fname, frac
            if best and best_frac >= SOURCE_MATCH_FRACTION:
                self._source_file[source] = best
        return dict(self._source_file)

    @staticmethod
    def _lookup(sets: Dict[str, List[str]], pathway: Dict) -> Optional[List[str]]:
        for candidate in (pathway.get('pathwayId'), pathway.get('id'),
                          pathway.get('name'), pathway.get('pathwayName')):
            if candidate and _key(candidate) in sets:
                return sets[_key(candidate)]
        return None

    def members(self, pathway: Dict) -> Optional[List[str]]:
        """Full membership of `pathway`, or None when no GMT set applies."""
        fname = self._source_file.get(str(pathway.get('source') or ''))
        if not fname:
            return None
        found = self._lookup(self.files[fname], pathway)
        if not found:
            return None
        sent = {str(g).upper() for g in pathway.get('genes', []) or []}
        if sent and not sent <= {g.upper() for g in found}:
            return None  # the website's genes are not all in this set: not the same pathway
        return list(found)
