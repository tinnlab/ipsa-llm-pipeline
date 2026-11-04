"""
Items 23 and 24: the Step 4 prompt shows every pathway theme (main L248) and every DE
target of each upstream regulator (main L246). No LLM involved.
"""

from src.pipeline.steps.step04_hypothesis_generation import Step04HypothesisGeneration

GENES = [{'geneSymbol': 'ACADS', 'foldChange': -1.4}, {'geneSymbol': 'CDK1', 'foldChange': 2.4}]
PATHWAYS = [{'name': 'Fatty acid degradation', 'NES': -3.4, 'genes': ['ACADS']}]


def _prompt(themes=None, mechanisms=None):
    step = Step04HypothesisGeneration.__new__(Step04HypothesisGeneration)
    return step._build_user_prompt(
        genes=GENES, pathways=PATHWAYS, themes=themes, hub_genes_result=None,
        mechanisms_result=mechanisms, experiment_context='HCC')


def test_all_nine_themes_reach_the_prompt():
    themes = [
        {'name': f'Theme {i}', 'description': 'd', 'key_genes': ['ACADS'], 'pathway_count': 2,
         'pathways': [{'name': f'P{i}', 'NES': 2.0 - i * 0.1, 'p_value_fdr': 0.01}],
         'avg_p_value_fdr': 0.01, 'significance': 'high'}
        for i in range(9)
    ]
    prompt = _prompt(themes=themes)
    for i in range(9):
        assert f'Theme {i}' in prompt, f'Theme {i} missing (Case Study 2 has nine themes)'


def test_every_regulator_target_is_listed():
    targets = [f'G{i}' for i in range(12)]
    prompt = _prompt(mechanisms={'upstream_regulators': [
        {'tf': 'HNF4A', 'overlap_count': 12, 'is_de': False, 'targets': targets}]})
    line = next(l for l in prompt.splitlines() if '**HNF4A**' in l)
    for g in targets:
        assert g in line
