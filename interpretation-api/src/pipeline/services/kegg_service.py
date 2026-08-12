"""
KEGG REST API Service

Retrieves curated pathway structures and mechanisms from KEGG database.
Reference: https://www.kegg.jp/kegg/rest/keggapi.html

This service provides:
- Pathway structure retrieval (KGML format)
- Gene and relation mapping
- Differential expression gene mapping to pathways
"""

import requests
import xmltodict
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


# KEGG compound identifiers look like C00338 / G12345 / D00123 (letter + 5 digits)
_KEGG_COMPOUND_RE = re.compile(r'^[CGD]\d{4,}$')


def _clean_symbol(sym: Optional[str]) -> Optional[str]:
    """Normalise a KEGG-derived gene symbol.

    KEGG graphics names for multi-gene boxes are truncated with a trailing "..."
    (e.g. "UGT2B11...", "CNPY3-GNMT..."). Strip that and surrounding
    punctuation/whitespace so we never emit a truncated pseudo-symbol. Legitimate
    readthrough names (e.g. NT5C1B-RDH14, CYP3A7-CYP3A51P) contain no "..." and
    are preserved.
    """
    if not sym:
        return None
    sym = sym.strip().strip(',').strip()
    sym = re.sub(r'\.{2,}$', '', sym).strip()   # drop trailing ellipsis
    return sym or None


def _is_gene_symbol(sym: Optional[str]) -> bool:
    """True if `sym` is a plausible gene symbol (not undefined / compound / map name)."""
    sym = _clean_symbol(sym)
    if not sym:
        return False
    if sym.lower() == 'undefined':
        return False
    if ' ' in sym:                       # pathway/map names contain spaces
        return False
    if _KEGG_COMPOUND_RE.match(sym):     # compound/glycan/drug IDs
        return False
    if sym.isdigit():                    # unmapped Entrez id
        return False
    return True


@dataclass
class PathwayEntry:
    """Represents a single entry in a KEGG pathway (gene, compound, etc.)"""
    id: str
    type: str  # gene, compound, group, map
    name: str
    names: List[str]
    gene_symbol: Optional[str] = None
    link: Optional[str] = None
    reaction: Optional[str] = None
    components: List[str] = field(default_factory=list)  # member entry ids for 'group' entries


@dataclass
class PathwayRelation:
    """Represents a regulatory relationship between two pathway entries"""
    entry1: str
    entry2: str
    type: str  # ECrel, PPrel, GErel, PCrel
    subtype: str  # activation, inhibition, phosphorylation, etc.
    value: Optional[str] = None
    source: Optional[str] = None
    target: Optional[str] = None
    source_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)


@dataclass
class PathwayReaction:
    """Represents a biochemical reaction in a pathway"""
    id: str
    name: str
    type: str  # reversible or irreversible
    substrates: List[str] = field(default_factory=list)
    products: List[str] = field(default_factory=list)


@dataclass
class PathwayStructure:
    """Complete pathway structure with genes, relations, and reactions"""
    id: str
    name: str
    organism: str
    source: str = 'kegg'  # 'kegg' or 'reactome'
    genes: List[PathwayEntry] = field(default_factory=list)
    compounds: List[PathwayEntry] = field(default_factory=list)
    groups: List[PathwayEntry] = field(default_factory=list)
    relations: List[PathwayRelation] = field(default_factory=list)
    reactions: List[PathwayReaction] = field(default_factory=list)
    entry_map: Dict[str, PathwayEntry] = field(default_factory=dict)


@dataclass
class MappedDEGene:
    """Differentially expressed gene mapped to a pathway"""
    pathway_entry_id: str
    gene_symbol: str
    kegg_names: List[str]
    fold_change: float
    p_value: float
    direction: str  # 'up' or 'down'


class KEGGService:
    """Service for interacting with KEGG REST API"""

    def __init__(self):
        self.base_url = 'https://rest.kegg.jp'
        self.cache: Dict[str, Any] = {}
        self.organism_map = {
            'Homo sapiens': 'hsa',
            'Mus musculus': 'mmu',
            'Rattus norvegicus': 'rno',
            'Danio rerio': 'dre',
            'Drosophila melanogaster': 'dme',
            'Caenorhabditis elegans': 'cel',
            'Saccharomyces cerevisiae': 'sce'
        }

    def get_pathway_structure(
        self,
        pathway_name: str,
        organism: str = 'Homo sapiens'
    ) -> Optional[PathwayStructure]:
        """
        Get pathway structure (KGML) from KEGG

        Args:
            pathway_name: Pathway name (e.g., "MAPK signaling pathway")
            organism: Organism name (e.g., "Homo sapiens")

        Returns:
            PathwayStructure object or None if not found
        """
        print(f'  Querying KEGG for pathway: "{pathway_name}"')

        # 1. Find pathway ID from name
        pathway_id = self._find_pathway_id(pathway_name, organism)

        if not pathway_id:
            print(f'  ⚠️  Pathway not found in KEGG: "{pathway_name}"')
            return None

        print(f'  Found KEGG pathway ID: {pathway_id}')

        # 2. Get KGML (pathway structure XML)
        kgml = self._get_kgml(pathway_id)

        if not kgml:
            print(f'  ⚠️  Could not retrieve KGML for {pathway_id}')
            return None

        # 3. Parse KGML to extract structure
        structure = self._parse_kgml(kgml, pathway_id)

        if structure:
            print(f'  Parsed pathway structure: {len(structure.genes)} genes, '
                  f'{len(structure.relations)} relations')

        return structure

    def _find_pathway_id(self, pathway_name: str, organism: str) -> Optional[str]:
        """Find KEGG pathway ID from pathway name"""
        org_code = self.organism_map.get(organism)
        if org_code is None:
            # Unknown organism: do NOT silently default to human (hsa) — that would
            # convert a generic 'map' id to an hsa id and surface wrong-species
            # curated data. Skip KEGG so the caller uses the organism-aware
            # Pathway Commons/LLM fallback instead. Warn only once per organism.
            warn_key = f'unsupported_org_{organism}'
            if warn_key not in self.cache:
                print(f'  ⚠️  Organism not supported for KEGG lookup: "{organism}" — '
                      f'skipping KEGG (Pathway Commons/LLM fallback will handle it)')
                self.cache[warn_key] = True
            return None

        cache_key = f'search_{org_code}_{pathway_name}'

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Search KEGG pathways
            search_term = self._normalize_pathway_name(pathway_name)
            url = f'{self.base_url}/find/pathway/{search_term}'

            response = requests.get(url, timeout=30)
            if not response.ok:
                return None

            text = response.text.strip()
            if not text:
                return None

            lines = [ln for ln in text.split('\n') if ln.strip()]

            # KEGG's find/pathway returns reference IDs whose prefix convention has
            # changed over time: it used to return "path:map00650" but now returns a
            # bare "map00650". Prefer an organism-specific hit if present, otherwise
            # take the first reference hit, then normalize to the organism-specific
            # form (e.g. hsa00650) — KGML only exists for that form, not the generic
            # reference map (get/map00650/kgml -> HTTP 404).
            chosen = None
            for line in lines:
                parts = line.split('\t')
                if len(parts) < 2:
                    continue

                pid = parts[0].split(':', 1)[-1]  # strip any db prefix (e.g. 'path:')

                # Prefer organism-specific pathway
                if pid.startswith(org_code):
                    chosen = pid
                    break
                if chosen is None:
                    chosen = pid

            if chosen is None:
                return None

            pathway_id = self._normalize_pathway_id(chosen, org_code)
            self.cache[cache_key] = pathway_id
            return pathway_id

        except Exception as e:
            print(f'  Error searching KEGG: {e}')
            return None

    def _normalize_pathway_id(self, raw_id: str, org_code: str) -> str:
        """Normalize a KEGG pathway ID to the organism-specific form.

        Handles all observed KEGG find/pathway shapes:
          'path:map00650' / 'map00650' / '00650' -> '<org_code>00650'
          'path:hsa00650' / 'hsa00650'           -> 'hsa00650' (already org-specific)

        KGML is only available for the organism-specific pathway (e.g. hsa00650);
        the generic reference map has no KGML (get/map00650/kgml -> HTTP 404).

        Args:
            raw_id: A pathway ID from KEGG find/pathway, with or without a db prefix.
            org_code: KEGG organism code (e.g. 'hsa', 'mmu').

        Returns:
            The organism-specific pathway ID, or the input (prefix-stripped) unchanged
            if it does not match a KEGG reference-pathway shape.
        """
        pid = raw_id.split(':', 1)[-1]  # 'path:map00650' -> 'map00650'
        if pid.startswith(org_code):
            return pid
        m = re.match(r'^(?:map)?(\d{5})$', pid)  # 'map00650'/'00650' -> '00650'
        if m:
            return f'{org_code}{m.group(1)}'
        return pid

    def _get_kgml(self, pathway_id: str) -> Optional[str]:
        """Retrieve KGML (XML) for pathway"""
        cache_key = f'kgml_{pathway_id}'

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            url = f'{self.base_url}/get/{pathway_id}/kgml'

            response = requests.get(url, timeout=30)
            if not response.ok:
                return None

            xml = response.text
            self.cache[cache_key] = xml
            return xml

        except Exception as e:
            print(f'  Error fetching KGML: {e}')
            return None

    def _parse_kgml(self, kgml: str, pathway_id: str) -> Optional[PathwayStructure]:
        """Parse KGML XML to extract pathway structure"""
        try:
            parsed = xmltodict.parse(kgml)
            pathway = parsed['pathway']

            # Extract entries (genes, compounds, groups)
            entry_list = pathway.get('entry', [])
            if not isinstance(entry_list, list):
                entry_list = [entry_list]

            entries = self._extract_entries(entry_list)

            # Extract relations (interactions between entries)
            relation_list = pathway.get('relation', [])
            if not isinstance(relation_list, list):
                relation_list = [relation_list] if relation_list else []

            relations = self._extract_relations(relation_list, entries)

            # Extract reactions (biochemical reactions)
            reaction_list = pathway.get('reaction', [])
            if not isinstance(reaction_list, list):
                reaction_list = [reaction_list] if reaction_list else []

            reactions = self._extract_reactions(reaction_list, entries)

            # Build entry map
            entry_map = {e.id: e for e in entries}

            structure = PathwayStructure(
                id=pathway_id,
                name=pathway['@title'] if '@title' in pathway else pathway.get('@name', ''),
                organism=pathway.get('@org', ''),
                genes=[e for e in entries if e.type == 'gene'],
                compounds=[e for e in entries if e.type == 'compound'],
                groups=[e for e in entries if e.type == 'group'],
                relations=relations,
                reactions=reactions,
                entry_map=entry_map
            )

            return structure

        except Exception as e:
            print(f'  Error parsing KGML: {e}')
            return None

    def _extract_entries(self, entry_array: List[Dict]) -> List[PathwayEntry]:
        """Extract entries (genes, compounds, groups) from KGML"""
        entries = []

        for entry in entry_array:
            graphics = entry.get('graphics', {})
            if not isinstance(graphics, dict):
                graphics = graphics[0] if isinstance(graphics, list) else {}

            # Extract gene names from entry
            name_attr = entry['@name']
            gene_names = [
                name.split(':')[-1]  # Remove organism prefix (e.g., "hsa:7157" -> "7157")
                for name in name_attr.split()
            ]

            graphics_name = graphics.get('@name', entry['@name'])

            # For 'group' entries (protein complexes), collect member entry ids so
            # relations to a complex can be expanded to its constituent genes.
            components = []
            if entry.get('@type') == 'group':
                comp = entry.get('component', [])
                if not isinstance(comp, list):
                    comp = [comp] if comp else []
                components = [c['@id'] for c in comp if isinstance(c, dict) and '@id' in c]

            entries.append(PathwayEntry(
                id=entry['@id'],
                type=entry['@type'],
                name=graphics_name,
                names=gene_names,
                gene_symbol=self._extract_gene_symbol(graphics_name),
                link=entry.get('@link'),
                reaction=entry.get('@reaction'),
                components=components
            ))

        return entries

    def _extract_relations(
        self,
        relation_array: List[Dict],
        entries: List[PathwayEntry]
    ) -> List[PathwayRelation]:
        """Extract relations (regulatory relationships) from KGML.

        Keeps only gene<->gene edges. Endpoints that are compounds or maps produce
        garbage ("AGXT --> Sulfur metabolism", "C00338 --> CD14", "undefined --> CDK1"),
        so they are dropped. A 'group' (protein complex) endpoint is EXPANDED to its
        member genes so real gene<->complex interactions are preserved rather than lost.
        """
        entry_map = {e.id: e for e in entries}
        relations = []
        seen = set()

        def resolve_genes(entry: PathwayEntry) -> List[str]:
            """Resolve an entry to a list of valid gene symbols (expanding groups)."""
            if entry.type == 'gene':
                sym = _clean_symbol(entry.gene_symbol)
                return [sym] if _is_gene_symbol(sym) else []
            if entry.type == 'group':
                genes = []
                for comp_id in entry.components:
                    member = entry_map.get(comp_id)
                    if member and member.type == 'gene':
                        sym = _clean_symbol(member.gene_symbol)
                        if _is_gene_symbol(sym):
                            genes.append(sym)
                return genes
            return []  # compound / map endpoints are not genes

        for rel in relation_array:
            subtype = rel.get('subtype', {})
            if not isinstance(subtype, dict):
                subtype = subtype[0] if isinstance(subtype, list) else {}

            entry1 = entry_map.get(rel['@entry1'])
            entry2 = entry_map.get(rel['@entry2'])

            if not (entry1 and entry2):
                continue

            sources = resolve_genes(entry1)
            targets = resolve_genes(entry2)
            if not sources or not targets:
                continue

            rel_type = rel['@type']
            rel_subtype = subtype.get('@name', '')

            # Emit one edge per gene pair (group expansion), dropping self-loops and
            # de-duplicating identical edges (KEGG lists many near-identical ECrel rows).
            for source in sources:
                for target in targets:
                    if source == target:
                        continue
                    key = (source, target, rel_type, rel_subtype)
                    if key in seen:
                        continue
                    seen.add(key)
                    relations.append(PathwayRelation(
                        entry1=rel['@entry1'],
                        entry2=rel['@entry2'],
                        type=rel_type,
                        subtype=rel_subtype,
                        value=subtype.get('@value'),
                        source=source,
                        target=target,
                        source_names=entry1.names,
                        target_names=entry2.names
                    ))

        return relations

    def _extract_reactions(
        self,
        reaction_array: List[Dict],
        entries: List[PathwayEntry]
    ) -> List[PathwayReaction]:
        """Extract reactions (biochemical reactions) from KGML"""
        reactions = []

        for reaction in reaction_array:
            substrate_list = reaction.get('substrate', [])
            if not isinstance(substrate_list, list):
                substrate_list = [substrate_list] if substrate_list else []

            product_list = reaction.get('product', [])
            if not isinstance(product_list, list):
                product_list = [product_list] if product_list else []

            substrates = [s['@name'] for s in substrate_list if '@name' in s]
            products = [p['@name'] for p in product_list if '@name' in p]

            reactions.append(PathwayReaction(
                id=reaction['@id'],
                name=reaction['@name'],
                type=reaction['@type'],
                substrates=substrates,
                products=products
            ))

        return reactions

    def _extract_gene_symbol(self, name: Optional[str]) -> Optional[str]:
        """
        Extract gene symbol from KEGG graphics name.

        KEGG graphics names are a comma-separated alias list whose FIRST token is the
        primary HGNC symbol, e.g. "TP53, p53" -> "TP53", "ACAT1, ACAT, MAT" -> "ACAT1".

        Do NOT strip trailing digits: real gene symbols routinely end in digits
        (TP53, ACAT1, BDH1, COX4I2, NDUFA4L2). Stripping them silently corrupts symbols
        and breaks DE-gene matching against the pathway.
        """
        if not name:
            return None

        # KEGG format: "SYMBOL, alias, alias..." or just "SYMBOL".
        # _clean_symbol strips KEGG's trailing "..." truncation so a multi-gene box
        # like "UGT2B11..." yields "UGT2B11" rather than a truncated pseudo-symbol.
        return _clean_symbol(name.split(',')[0])

    def _normalize_pathway_name(self, name: str) -> str:
        """Normalize pathway name for search"""
        # Remove "pathway" suffix
        name = re.sub(r'\s+pathway$', '', name, flags=re.IGNORECASE)
        # Remove database suffix (KEGG)
        name = re.sub(r'\s*\([^)]*\)', '', name)
        # Replace special chars with space
        name = re.sub(r'[^\w\s]', ' ', name)
        return name.strip()

    def map_de_genes_to_pathway(
        self,
        pathway_structure: PathwayStructure,
        de_genes: List[Dict]
    ) -> List[MappedDEGene]:
        """
        Map user's DE genes to pathway entries

        Args:
            pathway_structure: Parsed pathway structure
            de_genes: List of differentially expressed genes

        Returns:
            List of genes that are both in the pathway and differentially expressed
        """
        if not pathway_structure:
            return []

        # Create map of gene symbols to DE data
        de_gene_map = {}
        for gene in de_genes:
            symbol = (
                gene.get('name') or
                gene.get('gene') or
                gene.get('geneName') or
                gene.get('geneSymbol', '')
            ).upper()
            de_gene_map[symbol] = gene

        # Map pathway genes to DE genes.
        # KEGG places the same gene in multiple graphics nodes, so a symbol can
        # appear many times in pathway_structure.genes. De-duplicate by symbol so a
        # DE gene is listed once (avoids "CYP1A1 x6" in the report and in the LLM prompt).
        mapped_genes = []
        seen_symbols = set()

        for pathway_gene in pathway_structure.genes:
            if not pathway_gene.gene_symbol:
                continue

            symbol = pathway_gene.gene_symbol.upper()

            if symbol in de_gene_map and symbol not in seen_symbols:
                seen_symbols.add(symbol)
                de_gene = de_gene_map[symbol]
                fold_change = de_gene.get('foldChange', 0)

                mapped_genes.append(MappedDEGene(
                    pathway_entry_id=pathway_gene.id,
                    gene_symbol=pathway_gene.gene_symbol,
                    kegg_names=pathway_gene.names,
                    fold_change=fold_change,
                    p_value=de_gene.get('pValue', 1.0),
                    # 0 fold change is neither up nor down (avoids false ↓ on FC 0)
                    direction='up' if fold_change > 0 else 'down' if fold_change < 0 else 'neutral'
                ))

        return mapped_genes

    def find_de_gene_relations(
        self,
        pathway_structure: PathwayStructure,
        mapped_de_genes: List[MappedDEGene]
    ) -> List[PathwayRelation]:
        """
        Find relations involving DE genes

        Args:
            pathway_structure: Parsed pathway structure
            mapped_de_genes: List of mapped DE genes

        Returns:
            List of relations involving at least one DE gene
        """
        if not pathway_structure or not mapped_de_genes:
            return []

        de_gene_ids = {g.pathway_entry_id for g in mapped_de_genes}

        return [
            rel for rel in pathway_structure.relations
            if rel.entry1 in de_gene_ids or rel.entry2 in de_gene_ids
        ]
