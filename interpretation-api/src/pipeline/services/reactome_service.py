"""
Reactome Content Service API Client

Retrieves curated pathway structures and mechanisms from Reactome database.
Reference: https://reactome.org/dev/content-service

This service provides:
- Pathway structure retrieval (JSON format)
- Reaction and participant mapping
- Differential expression gene mapping to pathways
"""

import requests
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Import shared data classes from kegg_service
from .kegg_service import (
    PathwayEntry,
    PathwayRelation,
    PathwayReaction,
    PathwayStructure,
    MappedDEGene
)


class ReactomeService:
    """Service for interacting with Reactome Content Service API"""

    def __init__(self):
        self.base_url = 'https://reactome.org/ContentService'
        self.cache: Dict[str, Any] = {}
        self.organism_map = {
            'Homo sapiens': '9606',
            'Mus musculus': '10090',
            'Rattus norvegicus': '10116',
            'Danio rerio': '7955',
            'Drosophila melanogaster': '7227',
            'Caenorhabditis elegans': '6239',
            'Saccharomyces cerevisiae': '4932'
        }

    def get_pathway_structure(
        self,
        pathway_name_or_id: str,
        organism: str = 'Homo sapiens'
    ) -> Optional[PathwayStructure]:
        """
        Get pathway structure from Reactome

        Args:
            pathway_name_or_id: Pathway name or Reactome stable ID (R-HSA-xxxxx)
            organism: Organism name (e.g., "Homo sapiens")

        Returns:
            PathwayStructure object or None if not found
        """
        print(f'  Querying Reactome for pathway: "{pathway_name_or_id}"')

        # 1. Find pathway ID if name was provided
        pathway_id = pathway_name_or_id
        if not self._is_reactome_id(pathway_name_or_id):
            pathway_id = self._find_pathway_id(pathway_name_or_id, organism)
            if not pathway_id:
                print(f'  ⚠️  Pathway not found in Reactome: "{pathway_name_or_id}"')
                return None

        print(f'  Found Reactome pathway ID: {pathway_id}')

        # 2. Get pathway data
        pathway_data = self._get_pathway_data(pathway_id)
        if not pathway_data:
            print(f'  ⚠️  Could not retrieve pathway data for {pathway_id}')
            return None

        # 3. Get participating physical entities
        participants = self._get_participants(pathway_id)
        if not participants:
            print(f'  ⚠️  Could not retrieve participants for {pathway_id}')
            return None

        # 4. Parse pathway structure
        structure = self._parse_pathway_structure(pathway_id, pathway_data, participants)

        if structure:
            print(f'  Parsed pathway structure: {len(structure.genes)} genes, '
                  f'{len(structure.relations)} relations')

        return structure

    def _is_reactome_id(self, identifier: str) -> bool:
        """Check if string matches Reactome stable ID pattern"""
        return bool(re.match(r'^R-[A-Z]{3}-\d+$', identifier))

    def _find_pathway_id(self, pathway_name: str, organism: str) -> Optional[str]:
        """
        Find Reactome pathway ID from pathway name

        Note: Reactome API doesn't have a simple text search endpoint.
        This implementation assumes pathway_name_or_id is already a Reactome ID.
        For production, you could:
        1. Use Reactome search API (if available)
        2. Download and cache pathway list
        3. Use fuzzy matching on pathway names
        """
        # For now, if it's not a Reactome ID, we can't find it
        # Future enhancement: implement pathway name search
        print(f'  ⚠️  Pathway name search not yet implemented. Please provide Reactome stable ID (R-XXX-xxxxx)')
        return None

    def _get_pathway_data(self, pathway_id: str) -> Optional[Dict]:
        """Retrieve pathway data from Reactome"""
        cache_key = f'pathway_{pathway_id}'

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            url = f'{self.base_url}/data/query/{pathway_id}'
            response = requests.get(url, timeout=30)

            if not response.ok:
                return None

            data = response.json()
            self.cache[cache_key] = data
            return data

        except Exception as e:
            print(f'  Error fetching pathway data: {e}')
            return None

    def _get_participants(self, pathway_id: str) -> Optional[List[Dict]]:
        """Retrieve participating physical entities"""
        cache_key = f'participants_{pathway_id}'

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Correct endpoint: /data/participants/{id}
            url = f'{self.base_url}/data/participants/{pathway_id}'
            response = requests.get(url, timeout=30)

            if not response.ok:
                return None

            participants = response.json()
            self.cache[cache_key] = participants
            return participants

        except Exception as e:
            print(f'  Error fetching participants: {e}')
            return None

    def _parse_pathway_structure(
        self,
        pathway_id: str,
        pathway_data: Dict,
        participants: List[Dict]
    ) -> Optional[PathwayStructure]:
        """Parse Reactome JSON to extract pathway structure"""
        try:
            # Extract basic info
            pathway_name = pathway_data.get('displayName', pathway_data.get('name', ''))
            species = pathway_data.get('species', [{}])[0] if 'species' in pathway_data else {}
            organism = species.get('displayName', '')

            # Extract genes/proteins from participants
            entries = self._extract_entries(participants)

            # Extract relations from pathway reactions
            # Parse reactions to create relations (input→output, catalysis, regulation)
            print(f'  Parsing Reactome reactions to extract relations...')
            relations = self._extract_relations_from_pathway(pathway_data, entries)
            print(f'  Extracted {len(relations)} relations from reactions')

            # Build entry map
            entry_map = {e.id: e for e in entries}

            structure = PathwayStructure(
                id=pathway_id,
                name=pathway_name,
                organism=organism,
                source='reactome',  # Explicitly set source
                genes=[e for e in entries if e.type in ['gene', 'protein', 'complex']],
                compounds=[e for e in entries if e.type == 'compound'],
                groups=[],  # Reactome doesn't have explicit groups
                relations=relations,
                reactions=[],  # Could parse reactions if needed
                entry_map=entry_map
            )

            return structure

        except Exception as e:
            print(f'  Error parsing pathway structure: {e}')
            return None

    def _extract_entries(self, participants: List[Dict]) -> List[PathwayEntry]:
        """Extract pathway entries (genes, proteins, complexes) from participants"""
        entries = []
        seen_gene_symbols = set()

        for participant in participants:
            entry_id = participant.get('stId', participant.get('dbId', ''))
            entry_type = participant.get('schemaClass', 'unknown')
            display_name = participant.get('displayName', '')

            # Map Reactome types to KEGG-like types
            if entry_type in ['EntityWithAccessionedSequence', 'Protein', 'ReferenceGeneProduct']:
                entry_type_mapped = 'protein'
            elif entry_type == 'Complex':
                entry_type_mapped = 'complex'
            elif entry_type in ['SimpleEntity', 'ChemicalDrug']:
                entry_type_mapped = 'compound'
            elif entry_type in ['EntitySet', 'CandidateSet']:
                entry_type_mapped = 'gene'  # Gene sets
            else:
                entry_type_mapped = 'gene'  # Default

            # For complexes with multiple genes (e.g., "TRADD:TRAF2:RIP1"),
            # create entries for each gene
            if ':' in display_name and entry_type == 'Complex':
                # Extract all gene symbols from complex
                name_without_compartment = display_name.split('[')[0].strip()
                potential_genes = name_without_compartment.split(':')

                for gene_name in potential_genes:
                    gene_name = gene_name.strip()
                    # Remove parentheses (e.g., "CASP8(1-479)" -> "CASP8")
                    match = re.match(r'^([A-Z][A-Z0-9]*)', gene_name)
                    if match:
                        gene_symbol = match.group(1)

                        # Skip duplicates and non-gene names
                        if gene_symbol in seen_gene_symbols or len(gene_symbol) < 2:
                            continue

                        seen_gene_symbols.add(gene_symbol)

                        entries.append(PathwayEntry(
                            id=f'{entry_id}_{gene_symbol}',
                            type='protein',
                            name=f'{gene_symbol} (from complex)',
                            names=[gene_symbol, display_name],
                            gene_symbol=gene_symbol,
                            link=f'https://reactome.org/content/detail/{entry_id}'
                        ))
            else:
                # Single gene/protein
                gene_symbol = self._extract_gene_symbol(participant, display_name)

                if gene_symbol and gene_symbol not in seen_gene_symbols:
                    seen_gene_symbols.add(gene_symbol)

                    entries.append(PathwayEntry(
                        id=str(entry_id),
                        type=entry_type_mapped,
                        name=display_name,
                        names=[display_name],
                        gene_symbol=gene_symbol,
                        link=f'https://reactome.org/content/detail/{entry_id}'
                    ))

        return entries

    def _extract_gene_symbol(self, participant: Dict, display_name: str) -> Optional[str]:
        """
        Extract gene symbol from Reactome participant

        Reactome format examples:
        - "TP53 [nucleoplasm]" -> "TP53"
        - "CASP8(1-479) [cytosol]" -> "CASP8"
        - "PIK3CA:p85 [cytosol]" -> "PIK3CA"
        - "TRADD:TRAF2:RIP1 [cytosol]" -> "TRADD" (first in complex)
        """
        # Try to get from geneName field if available
        if 'geneName' in participant:
            return participant['geneName']

        # Parse from displayName
        # Remove compartment info first: "GENE [compartment]" -> "GENE"
        name_without_compartment = display_name.split('[')[0].strip()

        # Handle complexes: "GENE1:GENE2:GENE3" -> take first gene
        if ':' in name_without_compartment:
            name_without_compartment = name_without_compartment.split(':')[0].strip()

        # Handle isoforms/fragments: "CASP8(1-479)" -> "CASP8"
        # Match gene symbol before parentheses
        match = re.match(r'^([A-Z][A-Z0-9]*)', name_without_compartment)
        if match:
            symbol = match.group(1)
            # Don't remove trailing numbers - they're part of the gene name
            # (e.g., CASP8, TP53, PIK3CA)
            return symbol

        # Fallback: take first word before space
        parts = name_without_compartment.split()
        if parts:
            symbol = parts[0]
            # Only return if it starts with uppercase (likely a gene)
            if symbol and symbol[0].isupper():
                return symbol

        return None

    def _extract_relations_from_pathway(
        self,
        pathway_data: Dict,
        entries: List[PathwayEntry]
    ) -> List[PathwayRelation]:
        """
        Extract relations from Reactome pathway data by parsing reactions

        Converts Reactome reactions (input → output + catalysts) into PathwayRelation format.
        Adds support for:
        - reaction: input → output
        - catalysis: catalyst → product
        - positive_regulation: regulator → target
        - negative_regulation: regulator → target
        """
        relations = []
        pathway_id = pathway_data.get('stId', pathway_data.get('dbId'))

        if not pathway_id:
            return relations

        # Get all contained events (reactions) recursively
        try:
            contained_events = self._get_contained_events(pathway_id)
            if not contained_events:
                return relations

            # Build gene symbol map from entries
            gene_symbol_to_entry_id = {}
            for entry in entries:
                if entry.gene_symbol:
                    gene_symbol_to_entry_id[entry.gene_symbol.upper()] = entry.id

            # Process reactions
            for event in contained_events:
                # Skip if not a dict (sometimes can be just an ID)
                if not isinstance(event, dict):
                    continue

                event_type = event.get('schemaClass', '')

                # Only process ReactionLikeEvents
                if 'Reaction' not in event_type and event_type != 'BlackBoxEvent':
                    continue

                event_id = event.get('stId', event.get('dbId'))
                if not event_id:
                    continue

                # Get reaction details
                reaction_details = self._get_reaction_details(event_id)
                if not reaction_details:
                    continue

                # Extract relations from this reaction
                reaction_relations = self._parse_reaction_to_relations(
                    reaction_details,
                    gene_symbol_to_entry_id
                )

                relations.extend(reaction_relations)

        except Exception as e:
            print(f'  Warning: Error extracting reactions: {e}')

        return relations

    def _get_contained_events(self, pathway_id: str) -> Optional[List[Dict]]:
        """Get all contained events (reactions) recursively from a pathway"""
        cache_key = f'contained_events_{pathway_id}'

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            url = f'{self.base_url}/data/pathway/{pathway_id}/containedEvents'
            response = requests.get(url, timeout=30)

            if not response.ok:
                return None

            events = response.json()
            self.cache[cache_key] = events
            return events

        except Exception as e:
            print(f'  Error fetching contained events: {e}')
            return None

    def _get_reaction_details(self, reaction_id: str) -> Optional[Dict]:
        """Get detailed reaction information"""
        cache_key = f'reaction_details_{reaction_id}'

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            url = f'{self.base_url}/data/query/{reaction_id}'
            response = requests.get(url, timeout=30)

            if not response.ok:
                return None

            reaction = response.json()
            self.cache[cache_key] = reaction
            return reaction

        except Exception as e:
            return None

    def _parse_reaction_to_relations(
        self,
        reaction: Dict,
        gene_symbol_to_entry_id: Dict[str, str]
    ) -> List[PathwayRelation]:
        """
        Parse a Reactome reaction into PathwayRelation objects

        Creates relations for:
        1. Input → Output (type: "reaction")
        2. Catalyst → Output (type: "catalysis")
        3. Regulator → Target (type: "positive_regulation" or "negative_regulation")
        """
        relations = []

        inputs = reaction.get('input', [])
        outputs = reaction.get('output', [])
        catalysts = reaction.get('catalystActivity', [])
        regulators = reaction.get('regulatedBy', [])

        # Helper to extract gene symbols from physical entity
        def extract_genes_from_entity(entity):
            """Extract gene symbols from a physical entity"""
            if not isinstance(entity, dict):
                return []

            display_name = entity.get('displayName', '')
            return self._extract_gene_symbols_from_name(display_name)

        # 1. Create input → output relations (type: "reaction")
        for inp in inputs:
            input_genes = extract_genes_from_entity(inp)

            for outp in outputs:
                output_genes = extract_genes_from_entity(outp)

                for inp_gene in input_genes:
                    inp_gene_upper = inp_gene.upper()
                    if inp_gene_upper not in gene_symbol_to_entry_id:
                        continue

                    for outp_gene in output_genes:
                        outp_gene_upper = outp_gene.upper()
                        if outp_gene_upper not in gene_symbol_to_entry_id:
                            continue

                        # Skip self-relations
                        if inp_gene == outp_gene:
                            continue

                        relations.append(PathwayRelation(
                            entry1=gene_symbol_to_entry_id[inp_gene_upper],
                            entry2=gene_symbol_to_entry_id[outp_gene_upper],
                            type='reaction',
                            subtype='reaction',
                            source=inp_gene,
                            target=outp_gene,
                            source_names=[inp_gene],
                            target_names=[outp_gene]
                        ))

        # 2. Create catalyst → output relations (type: "catalysis")
        for catalyst_activity in catalysts:
            if not isinstance(catalyst_activity, dict):
                continue

            # Try to get catalyst genes from multiple possible fields
            catalyst_genes = []

            # Method 1: physicalEntity field (full query)
            physical_entity = catalyst_activity.get('physicalEntity')
            if physical_entity:
                catalyst_genes.extend(extract_genes_from_entity(physical_entity))

            # Method 2: activeUnit field (contains actual protein components)
            active_units = catalyst_activity.get('activeUnit', [])
            if isinstance(active_units, list):
                for unit in active_units:
                    if isinstance(unit, dict):
                        catalyst_genes.extend(extract_genes_from_entity(unit))

            # Method 3: Extract from displayName as fallback
            if not catalyst_genes:
                display_name = catalyst_activity.get('displayName', '')
                catalyst_genes = self._extract_gene_symbols_from_name(display_name)

            # Create relations for each catalyst gene → output
            for outp in outputs:
                output_genes = extract_genes_from_entity(outp)

                for cat_gene in catalyst_genes:
                    cat_gene_upper = cat_gene.upper()
                    if cat_gene_upper not in gene_symbol_to_entry_id:
                        continue

                    for outp_gene in output_genes:
                        outp_gene_upper = outp_gene.upper()
                        if outp_gene_upper not in gene_symbol_to_entry_id:
                            continue

                        relations.append(PathwayRelation(
                            entry1=gene_symbol_to_entry_id[cat_gene_upper],
                            entry2=gene_symbol_to_entry_id[outp_gene_upper],
                            type='catalysis',
                            subtype='catalysis',
                            source=cat_gene,
                            target=outp_gene,
                            source_names=[cat_gene],
                            target_names=[outp_gene]
                        ))

        # 3. Create regulator → target relations
        for regulation in regulators:
            if not isinstance(regulation, dict):
                continue

            regulation_type = regulation.get('schemaClass', '')
            regulator_entity = regulation.get('regulator')

            if not regulator_entity:
                continue

            regulator_genes = extract_genes_from_entity(regulator_entity)

            # Determine regulation subtype
            if 'Positive' in regulation_type:
                subtype = 'positive_regulation'
            elif 'Negative' in regulation_type:
                subtype = 'negative_regulation'
            else:
                subtype = 'regulation'

            # Regulators affect all outputs
            for outp in outputs:
                output_genes = extract_genes_from_entity(outp)

                for reg_gene in regulator_genes:
                    reg_gene_upper = reg_gene.upper()
                    if reg_gene_upper not in gene_symbol_to_entry_id:
                        continue

                    for outp_gene in output_genes:
                        outp_gene_upper = outp_gene.upper()
                        if outp_gene_upper not in gene_symbol_to_entry_id:
                            continue

                        relations.append(PathwayRelation(
                            entry1=gene_symbol_to_entry_id[reg_gene_upper],
                            entry2=gene_symbol_to_entry_id[outp_gene_upper],
                            type='regulation',
                            subtype=subtype,
                            source=reg_gene,
                            target=outp_gene,
                            source_names=[reg_gene],
                            target_names=[outp_gene]
                        ))

        return relations

    def _extract_gene_symbols_from_name(self, display_name: str) -> List[str]:
        """
        Extract all gene symbols from a display name

        Examples:
        - "CASP8(1-479) [cytosol]" → ["CASP8"]
        - "TRADD:TRAF2:RIP1 [cytosol]" → ["TRADD", "TRAF2", "RIP1"]
        """
        # Remove compartment
        name_without_compartment = display_name.split('[')[0].strip()

        # Split by colon for complexes
        parts = name_without_compartment.split(':')

        gene_symbols = []
        for part in parts:
            part = part.strip()

            # Remove parentheses (isoforms/fragments)
            match = re.match(r'^([A-Z][A-Z0-9]*)', part)
            if match:
                symbol = match.group(1)
                if len(symbol) >= 2:  # Skip single letters
                    gene_symbols.append(symbol)

        return gene_symbols

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

        # Map pathway genes to DE genes
        mapped_genes = []

        for pathway_gene in pathway_structure.genes:
            if not pathway_gene.gene_symbol:
                continue

            symbol = pathway_gene.gene_symbol.upper()

            if symbol in de_gene_map:
                de_gene = de_gene_map[symbol]
                fold_change = de_gene.get('foldChange', 0)

                mapped_genes.append(MappedDEGene(
                    pathway_entry_id=pathway_gene.id,
                    gene_symbol=pathway_gene.gene_symbol,
                    kegg_names=pathway_gene.names,  # Actually Reactome names
                    fold_change=fold_change,
                    p_value=de_gene.get('pValue', 1.0),
                    # 0 fold change is neither up nor down (consistent with kegg_service)
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

    def get_reaction_details(self, reaction_id: str) -> Optional[Dict]:
        """
        Get detailed reaction information including inputs, outputs, catalysts, regulators

        This method can be used to extract detailed mechanistic information
        for reactions involving DE genes.
        """
        cache_key = f'reaction_{reaction_id}'

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            url = f'{self.base_url}/data/query/{reaction_id}'
            response = requests.get(url, timeout=30)

            if not response.ok:
                return None

            data = response.json()
            self.cache[cache_key] = data
            return data

        except Exception as e:
            print(f'  Error fetching reaction details: {e}')
            return None
