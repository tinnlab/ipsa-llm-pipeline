"""
Meta-Step 07: Reproducibility Analysis

Compares meta-analysis results with individual dataset results to assess:
1. Reproducibility across studies
2. Power gain from meta-analysis (meta-unique findings)
3. Direction concordance and effect size consistency
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


@dataclass
class PathwayReproducibility:
    """Reproducibility metrics for a single pathway"""
    pathway_id: str  # Pathway ID (e.g., hsa04110, R-HSA-123456) or name if ID unavailable
    pathway_name: str  # Human-readable pathway name
    found_in_meta: bool
    found_in_datasets: List[str]  # Dataset names where found
    study_count: int  # Number of datasets where found
    direction_concordance: float  # 0-1, agreement in up/down regulation
    meta_pvalue: Optional[float]
    meta_fdr: Optional[float]
    meta_es: Optional[float]  # Enrichment score from meta
    dataset_pvalues: Dict[str, float]  # dataset_name -> pvalue
    dataset_es: Dict[str, float]  # dataset_name -> enrichment score
    is_meta_unique: bool  # Found only in meta (power gain)
    reproducibility_category: str  # 'high', 'moderate', 'low', 'meta_unique'


@dataclass
class GeneReproducibility:
    """Reproducibility metrics for a single gene"""
    gene_symbol: str
    found_in_meta: bool
    found_in_datasets: List[str]
    study_count: int
    direction_concordance: float
    meta_fc: Optional[float]
    meta_pvalue: Optional[float]
    dataset_fcs: Dict[str, float]  # dataset_name -> fold change
    dataset_pvalues: Dict[str, float]
    is_meta_unique: bool
    reproducibility_category: str


@dataclass
class ReproducibilityResult:
    """Results from reproducibility analysis"""
    pathway_reproducibility: List[Dict[str, Any]]
    gene_reproducibility: List[Dict[str, Any]]
    summary_stats: Dict[str, Any]
    meta_unique_pathways: List[str]
    meta_unique_genes: List[str]
    highly_reproducible_pathways: List[str]
    highly_reproducible_genes: List[str]


class MetaStep07Reproducibility:
    """
    Meta-Step 07: Reproducibility Analysis

    Compares meta-analysis findings with individual dataset results to calculate
    reproducibility metrics and identify meta-unique discoveries (power gain).
    """

    def __init__(self):
        self.step_number = 7
        self.step_name = 'Reproducibility Analysis'

    def execute(
        self,
        meta_results: Dict[str, Any],
        dataset_results: Dict[str, Dict[str, Any]],
        dataset_names: List[str]
    ) -> ReproducibilityResult:
        """
        Execute reproducibility analysis

        Args:
            meta_results: Pipeline results from meta-analysis data
            dataset_results: Dict mapping dataset_name -> pipeline results
            dataset_names: List of dataset names

        Returns:
            ReproducibilityResult with reproducibility metrics
        """
        print(f'\n[Meta-Step {self.step_number}] {self.step_name}')
        print(f'  Comparing meta-analysis with {len(dataset_names)} individual datasets')

        # Extract pathways and genes from all results
        meta_pathways = self._extract_pathways(meta_results)
        meta_genes = self._extract_genes(meta_results)

        dataset_pathways = {
            name: self._extract_pathways(results)
            for name, results in dataset_results.items()
        }

        dataset_genes = {
            name: self._extract_genes(results)
            for name, results in dataset_results.items()
        }

        # Calculate pathway reproducibility
        print('\n  Analyzing pathway reproducibility...')
        pathway_repro = self._calculate_pathway_reproducibility(
            meta_pathways,
            dataset_pathways,
            dataset_names
        )

        # Calculate gene reproducibility (skip if no genes in any dataset)
        has_any_genes = bool(meta_genes) or any(dataset_genes.values())
        if has_any_genes:
            print('  Analyzing gene reproducibility...')
            gene_repro = self._calculate_gene_reproducibility(
                meta_genes,
                dataset_genes,
                dataset_names
            )
        else:
            print('  Skipping gene reproducibility (no genes provided in any dataset)...')
            gene_repro = []

        # Generate summary statistics
        print('  Computing summary statistics...')
        summary_stats = self._generate_summary_stats(
            pathway_repro,
            gene_repro,
            len(dataset_names)
        )

        # Identify key findings
        meta_unique_pathways = [
            p.pathway_name for p in pathway_repro if p.is_meta_unique
        ]
        meta_unique_genes = [
            g.gene_symbol for g in gene_repro if g.is_meta_unique
        ]

        highly_reproducible_pathways = [
            p.pathway_name for p in pathway_repro
            if p.reproducibility_category == 'high'
        ]
        highly_reproducible_genes = [
            g.gene_symbol for g in gene_repro
            if g.reproducibility_category == 'high'
        ]

        print(f'\n  Summary:')
        print(f'    Meta-unique pathways: {len(meta_unique_pathways)}')
        print(f'    Meta-unique genes: {len(meta_unique_genes)}')
        print(f'    Highly reproducible pathways: {len(highly_reproducible_pathways)}')
        print(f'    Highly reproducible genes: {len(highly_reproducible_genes)}')

        return ReproducibilityResult(
            pathway_reproducibility=[asdict(p) for p in pathway_repro],
            gene_reproducibility=[asdict(g) for g in gene_repro],
            summary_stats=summary_stats,
            meta_unique_pathways=meta_unique_pathways,
            meta_unique_genes=meta_unique_genes,
            highly_reproducible_pathways=highly_reproducible_pathways,
            highly_reproducible_genes=highly_reproducible_genes
        )

    def _extract_pathways(self, results: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Extract pathway data from pipeline results

        Returns:
            Dict mapping pathway_id -> pathway data (including name)
            Uses pathway ID as primary key for stable cross-dataset matching
            (KEGG/Reactome IDs are consistent across datasets, names may vary)
        """
        pathways = {}

        # From Step 1: Pathway themes
        # Support both old key format and new nested format
        step1_results = results.get('step1_pathway_themes', {})
        if not step1_results:
            # Try new nested format: results['steps']['step1']
            step1_results = results.get('steps', {}).get('step1', {})
        if step1_results:
            # Get pathways from themes
            for theme in step1_results.get('themes', []):
                for pathway in theme.get('pathways', []):
                    name = pathway.get('name', pathway.get('pathway', ''))
                    pathway_id = pathway.get('pathwayId', pathway.get('pathway_id', ''))

                    # Use pathway_id as key (stable KEGG/Reactome IDs),
                    # fall back to normalized name. Always lowercase for consistent matching.
                    key = (pathway_id.lower().strip() if pathway_id else
                           (name.lower().strip() if name else ''))

                    if key:
                        # Support both camelCase and snake_case field names
                        pvalue = pathway.get('pValue') or pathway.get('p_value')
                        pvalue_fdr = pathway.get('pValueFDR') or pathway.get('p_value_fdr')
                        es = pathway.get('ES') or pathway.get('es')

                        pathways[key] = {
                            'pathway_id': pathway_id,
                            'name': name,
                            'pValue': pvalue,
                            'pValueFDR': pvalue_fdr,
                            'ES': es,
                            'source': 'themed'
                        }

            # Get ungrouped pathways (support both key names)
            ungrouped_list = step1_results.get('ungrouped_pathways', []) or step1_results.get('ungrouped', [])
            for pathway in ungrouped_list:
                name = pathway.get('name', pathway.get('pathway', ''))
                pathway_id = pathway.get('pathwayId', pathway.get('pathway_id', ''))

                # Use pathway_id as key, fall back to normalized name. Always lowercase.
                key = (pathway_id.lower().strip() if pathway_id else
                       (name.lower().strip() if name else ''))

                if key and key not in pathways:
                    # Support both camelCase and snake_case field names
                    pvalue = pathway.get('pValue') or pathway.get('p_value')
                    pvalue_fdr = pathway.get('pValueFDR') or pathway.get('p_value_fdr')
                    es = pathway.get('ES') or pathway.get('es')

                    pathways[key] = {
                        'pathway_id': pathway_id,
                        'name': name,
                        'pValue': pvalue,
                        'pValueFDR': pvalue_fdr,
                        'ES': es,
                        'source': 'ungrouped'
                    }

        return pathways

    def _extract_genes(self, results: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Extract gene data from pipeline results

        Returns:
            Dict mapping gene_symbol -> gene data (empty if no genes provided)
        """
        genes = {}

        # Check if Step 2 was skipped (no genes provided)
        # Support both old key format and new nested format
        step2_results = results.get('step2_hub_genes', {})
        if not step2_results:
            # Try new nested format: results['steps']['step2']
            step2_results = results.get('steps', {}).get('step2', {})

        if step2_results and step2_results.get('metadata', {}).get('skipped'):
            # Step 2 was skipped due to missing genes
            return genes

        # From Step 2: Hub genes (support both 'hub_genes' and 'network_hubs' keys)
        if step2_results:
            hub_list = step2_results.get('hub_genes', []) or step2_results.get('network_hubs', [])
            for hub_gene in hub_list:
                symbol = hub_gene.get('gene', hub_gene.get('geneSymbol', hub_gene.get('gene_symbol', '')))
                if symbol:
                    # Support both camelCase and snake_case field names
                    fc = hub_gene.get('foldChange') or hub_gene.get('fold_change')
                    pval = hub_gene.get('pValue') or hub_gene.get('p_value')

                    genes[symbol] = {
                        'foldChange': fc,
                        'pValue': pval,
                        'degree': hub_gene.get('degree'),
                        'hub_score': hub_gene.get('hub_score'),
                        'source': 'hub'
                    }

        # Also get all DE genes from input if available
        # (This ensures we capture all significant genes, not just hubs)
        input_genes = results.get('input', {}).get('genes', [])
        for gene in input_genes:
            symbol = gene.get('geneSymbol', '')
            if symbol and symbol not in genes:
                genes[symbol] = {
                    'foldChange': gene.get('foldChange'),
                    'pValue': gene.get('pValue'),
                    'source': 'de_gene'
                }

        return genes

    def _calculate_pathway_reproducibility(
        self,
        meta_pathways: Dict[str, Dict],
        dataset_pathways: Dict[str, Dict[str, Dict]],
        dataset_names: List[str]
    ) -> List[PathwayReproducibility]:
        """
        Calculate reproducibility metrics for each pathway

        Note: Dictionary keys are pathway IDs (or names if ID unavailable)
        """
        # Get union of all pathway IDs/keys
        all_pathway_keys = set(meta_pathways.keys())
        for ds_pathways in dataset_pathways.values():
            all_pathway_keys.update(ds_pathways.keys())

        reproducibility_list = []

        for pathway_key in all_pathway_keys:
            # Check where this pathway appears (by ID/key)
            found_in_meta = pathway_key in meta_pathways
            found_in_datasets = [
                ds_name for ds_name, ds_pathways in dataset_pathways.items()
                if pathway_key in ds_pathways
            ]
            study_count = len(found_in_datasets)

            # Get meta values
            meta_data = meta_pathways.get(pathway_key, {})
            meta_pvalue = meta_data.get('pValue')
            meta_fdr = meta_data.get('pValueFDR')
            meta_es = meta_data.get('ES')
            pathway_id = meta_data.get('pathway_id', pathway_key)
            pathway_name = meta_data.get('name', pathway_key)

            # If not in meta, get name from first dataset
            if not found_in_meta and found_in_datasets:
                first_ds = found_in_datasets[0]
                ds_data = dataset_pathways[first_ds][pathway_key]
                pathway_id = ds_data.get('pathway_id', pathway_key)
                pathway_name = ds_data.get('name', pathway_key)

            # Get dataset values
            dataset_pvalues = {
                ds_name: dataset_pathways[ds_name][pathway_key].get('pValue')
                for ds_name in found_in_datasets
                if pathway_key in dataset_pathways[ds_name]
            }

            dataset_es = {
                ds_name: dataset_pathways[ds_name][pathway_key].get('ES')
                for ds_name in found_in_datasets
                if pathway_key in dataset_pathways[ds_name]
            }

            # Calculate direction concordance (based on ES sign)
            es_values = [es for es in dataset_es.values() if es is not None]
            if meta_es is not None:
                es_values.append(meta_es)

            direction_concordance = self._calculate_direction_concordance(es_values)

            # Determine if meta-unique (power gain)
            # Meta-unique means: found in meta-analysis but NOT in any individual dataset
            is_meta_unique = found_in_meta and study_count == 0

            # Categorize reproducibility
            if is_meta_unique:
                category = 'meta_unique'
            elif study_count >= len(dataset_names) * 0.75:  # Found in ≥75% of datasets
                category = 'high'
            elif study_count >= len(dataset_names) * 0.5:  # Found in ≥50% of datasets
                category = 'moderate'
            else:
                category = 'low'

            reproducibility_list.append(PathwayReproducibility(
                pathway_id=pathway_id,
                pathway_name=pathway_name,
                found_in_meta=found_in_meta,
                found_in_datasets=found_in_datasets,
                study_count=study_count,
                direction_concordance=direction_concordance,
                meta_pvalue=meta_pvalue,
                meta_fdr=meta_fdr,
                meta_es=meta_es,
                dataset_pvalues=dataset_pvalues,
                dataset_es=dataset_es,
                is_meta_unique=is_meta_unique,
                reproducibility_category=category
            ))

        # Sort by study count (most reproducible first), then by meta p-value
        reproducibility_list.sort(
            key=lambda p: (
                -p.study_count,
                p.meta_pvalue if p.meta_pvalue is not None else 1.0
            )
        )

        return reproducibility_list

    def _calculate_gene_reproducibility(
        self,
        meta_genes: Dict[str, Dict],
        dataset_genes: Dict[str, Dict[str, Dict]],
        dataset_names: List[str]
    ) -> List[GeneReproducibility]:
        """Calculate reproducibility metrics for each gene"""
        # Get union of all gene symbols
        all_gene_symbols = set(meta_genes.keys())
        for ds_genes in dataset_genes.values():
            all_gene_symbols.update(ds_genes.keys())

        reproducibility_list = []

        for gene_symbol in all_gene_symbols:
            # Check where this gene appears
            found_in_meta = gene_symbol in meta_genes
            found_in_datasets = [
                ds_name for ds_name, ds_genes in dataset_genes.items()
                if gene_symbol in ds_genes
            ]
            study_count = len(found_in_datasets)

            # Get meta values
            meta_data = meta_genes.get(gene_symbol, {})
            meta_fc = meta_data.get('foldChange')
            meta_pvalue = meta_data.get('pValue')

            # Get dataset values
            dataset_fcs = {
                ds_name: dataset_genes[ds_name][gene_symbol].get('foldChange')
                for ds_name in found_in_datasets
                if gene_symbol in dataset_genes[ds_name]
            }

            dataset_pvalues = {
                ds_name: dataset_genes[ds_name][gene_symbol].get('pValue')
                for ds_name in found_in_datasets
                if gene_symbol in dataset_genes[ds_name]
            }

            # Calculate direction concordance (based on FC sign)
            fc_values = [fc for fc in dataset_fcs.values() if fc is not None]
            if meta_fc is not None:
                fc_values.append(meta_fc)

            direction_concordance = self._calculate_direction_concordance(fc_values)

            # Determine if meta-unique
            is_meta_unique = found_in_meta and study_count == 0

            # Categorize reproducibility
            if is_meta_unique:
                category = 'meta_unique'
            elif study_count >= len(dataset_names) * 0.75:
                category = 'high'
            elif study_count >= len(dataset_names) * 0.5:
                category = 'moderate'
            else:
                category = 'low'

            reproducibility_list.append(GeneReproducibility(
                gene_symbol=gene_symbol,
                found_in_meta=found_in_meta,
                found_in_datasets=found_in_datasets,
                study_count=study_count,
                direction_concordance=direction_concordance,
                meta_fc=meta_fc,
                meta_pvalue=meta_pvalue,
                dataset_fcs=dataset_fcs,
                dataset_pvalues=dataset_pvalues,
                is_meta_unique=is_meta_unique,
                reproducibility_category=category
            ))

        # Sort by study count, then by meta p-value
        reproducibility_list.sort(
            key=lambda g: (
                -g.study_count,
                g.meta_pvalue if g.meta_pvalue is not None else 1.0
            )
        )

        return reproducibility_list

    def _calculate_direction_concordance(self, values: List[float]) -> float:
        """
        Calculate direction concordance (agreement in sign)

        Returns:
            Float 0-1 representing proportion of values with same sign
        """
        if not values or len(values) < 2:
            return 1.0  # Perfect concordance if only one value

        # Count positive and negative
        positive_count = sum(1 for v in values if v > 0)
        negative_count = sum(1 for v in values if v < 0)

        # Concordance is the proportion of the majority direction
        total = len(values)
        concordance = max(positive_count, negative_count) / total

        return concordance

    def _generate_summary_stats(
        self,
        pathway_repro: List[PathwayReproducibility],
        gene_repro: List[GeneReproducibility],
        n_datasets: int
    ) -> Dict[str, Any]:
        """Generate summary statistics"""
        # Pathway stats
        pathway_categories = {
            'high': len([p for p in pathway_repro if p.reproducibility_category == 'high']),
            'moderate': len([p for p in pathway_repro if p.reproducibility_category == 'moderate']),
            'low': len([p for p in pathway_repro if p.reproducibility_category == 'low']),
            'meta_unique': len([p for p in pathway_repro if p.reproducibility_category == 'meta_unique'])
        }

        # Gene stats
        gene_categories = {
            'high': len([g for g in gene_repro if g.reproducibility_category == 'high']),
            'moderate': len([g for g in gene_repro if g.reproducibility_category == 'moderate']),
            'low': len([g for g in gene_repro if g.reproducibility_category == 'low']),
            'meta_unique': len([g for g in gene_repro if g.reproducibility_category == 'meta_unique'])
        }

        # Average metrics
        avg_pathway_concordance = (
            sum(p.direction_concordance for p in pathway_repro) / len(pathway_repro)
            if pathway_repro else 0.0
        )

        avg_gene_concordance = (
            sum(g.direction_concordance for g in gene_repro) / len(gene_repro)
            if gene_repro else 0.0
        )

        return {
            'n_datasets': n_datasets,
            'total_pathways_analyzed': len(pathway_repro),
            'total_genes_analyzed': len(gene_repro),
            'pathway_categories': pathway_categories,
            'gene_categories': gene_categories,
            'avg_pathway_direction_concordance': avg_pathway_concordance,
            'avg_gene_direction_concordance': avg_gene_concordance,
            'power_gain': {
                'meta_unique_pathways': pathway_categories['meta_unique'],
                'meta_unique_genes': gene_categories['meta_unique']
            }
        }
