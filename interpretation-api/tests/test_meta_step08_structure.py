"""Meta report restructure: the meta report mirrors the single-route report (step 6) plus
one meta-only section, with contiguous section numbers. Step-6 markdown fixtures come from
the real Step06ReportGeneration, so the tests track its actual (numbered) headings."""
import re
from unittest.mock import MagicMock

import pytest

from src.pipeline.steps.step06_report_generation import Step06ReportGeneration
from src.pipeline.steps_meta.meta_step08_comparative_report import MetaStep08ComparativeReport

SKIPPED = {'metadata': {'skipped': True}}


def _step6_markdown(with_genes: bool) -> str:
    step = Step06ReportGeneration.__new__(Step06ReportGeneration)
    step.llm = MagicMock()
    step.llm.chat.return_value = 'Narrative text.'
    input_data = {
        'metadata': {'organism': 'Mus musculus', 'disease': 'Spaceflight'},
        'genes': [{'gene': 'CDK1', 'foldChange': 2.4}] if with_genes else [],
        'pathways': [{'name': 'OXPHOS', 'NES': -1.77}, {'name': 'Translation', 'NES': 1.4}],
    }
    step1 = {'themes': [{'name': 'Mitochondrial energy', 'significance': 'high',
                         'pathways': [{'name': 'OXPHOS', 'p_value_fdr': 1e-4, 'gene_count': 40}],
                         'pathway_count': 1, 'shared_genes': [], 'shared_gene_count': 0,
                         'avg_jaccard_overlap': 0.3, 'key_genes': [],
                         'biological_context': 'ctx', 'avg_p_value_fdr': 1e-4}],
             'ungrouped': []}
    if with_genes:
        step2 = {'network_hubs': [{'gene': 'CDK1', 'degree': 40, 'hub_score': 0.8, 'fold_change': 2.4}],
                 'master_regulators': [{'gene': 'CDK1', 'hub_score': 0.8, 'degree': 40,
                                        'fold_change': 2.4, 'role': 'kinase', 'pathways': []}]}
        step3 = {'pathwayMechanisms': [{'pathway': 'OXPHOS', 'biologicalFunction': 'ATP',
                                        'deGeneInvolvement': [{'gene': 'CDK1', 'foldChange': 2.4,
                                                               'roleInPathway': 'driver'}]}],
                 'report_section': ''}
    else:
        step2, step3 = SKIPPED, SKIPPED
    step4 = {'hypotheses': [{'statement': 'H1', 'confidence': 'HIGH'}]}
    report = step.execute(input_data=input_data, step1_output=step1, step2_output=step2,
                          step3_output=step3, step4_output=step4)
    return report.markdown_content if hasattr(report, 'markdown_content') else report['markdown_content']


def _discovery():
    return {
        'individual_findings': {
            'total_datasets': 14,
            'total_unique_pathways': 30,
            'pathways_in_multiple': 22,
            'pathways_in_single': 8,
            'top_pathways': [{'name': f'Pathway {i}', 'count': 20 - i, 'best_pvalue': 1e-4}
                             for i in range(20)],
            'pathway_counts': {f'Pathway {i}': {} for i in range(30)},
            'llm_summary': 'One paragraph about the datasets.',
        },
        'summary': {'all_synthetic': True},
        'enhanced_significance_pathways': [{'pathway_name': 'X'}],
        'novel_cooccurrence_pairs': [{'pathway_a': 'A', 'pathway_b': 'B'}],
        'cross_dataset_convergence': [{'theme_name': 'T'}],
        'biological_interpretation': 'old interpretation',
    }


def _meta_results(markdown, n_selected=40, hypotheses=1):
    return {
        'input_summary': {'num_pathways': n_selected},
        'steps': {'step4': {'hypotheses': [{}] * hypotheses},
                  'step5': {'markdown_content': markdown}},
    }


def _repro():
    return {'summary_stats': {'pathway_categories': {}, 'gene_categories': {},
                              'total_pathways_analyzed': 54, 'total_genes_analyzed': 0}}


def _run(markdown, discovery=None, meta_results=None):
    step = MetaStep08ComparativeReport.__new__(MetaStep08ComparativeReport)
    step.step_number, step.step_name = 8, 'Comparative Report Generation'
    step.llm = MagicMock()  # step 8 makes no LLM calls
    report = step.execute(
        reproducibility_results=_repro(),
        discovery_results=discovery if discovery is not None else _discovery(),
        meta_results=meta_results if meta_results is not None else _meta_results(markdown),
        dataset_results={},
        dataset_names=[f'd{i}' for i in range(14)],
        metadata={'organism': 'Mus musculus'},
    )
    step.llm.chat.assert_not_called()
    return report


def _h2(report):
    return [line[3:] for line in report.splitlines() if line.startswith('## ')]


@pytest.mark.parametrize('with_genes', [True, False])
def test_section_order_and_contiguous_numbers(with_genes):
    report = _run(_step6_markdown(with_genes))
    headings = _h2(report)
    numbers = [int(m.group(1)) for h in headings if (m := re.match(r'(\d+)\. ', h))]
    assert numbers == list(range(1, len(numbers) + 1))

    texts = [re.sub(r'^\d+\.\s*', '', h) for h in headings]
    expected = ['Executive Summary', 'Findings Across Individual Datasets', 'Pathway Theme Analysis']
    if with_genes:
        expected.append('Network Hub Genes')
    assert texts[:len(expected)] == expected
    assert texts[-1] == 'Appendices'
    assert texts[-2] == 'Summary and Conclusions'
    assert any(t.startswith('Pathway Mechanisms') for t in texts)
    assert any(t.startswith('Mechanistic Hypotheses') for t in texts)
    assert headings[0] == 'Executive Summary' and headings[-1] == 'Appendices'  # unnumbered
    assert not any('Therapeutic' in h for h in headings)


def test_no_genes_run_reads_one_to_five():
    headings = _h2(_run(_step6_markdown(with_genes=False)))
    assert [h for h in headings if re.match(r'\d+\. ', h)] == [
        '1. Findings Across Individual Datasets',
        '2. Pathway Theme Analysis',
        '3. Pathway Mechanisms and Molecular Interactions',
        '4. Mechanistic Hypotheses and Predictions',
        '5. Summary and Conclusions',
    ]


def test_step6_body_text_is_verbatim():
    md = _step6_markdown(with_genes=True)
    report = _run(md)
    for kind, _, body in MetaStep08ComparativeReport._split_markdown_sections(md):
        if kind in ('executive_summary', 'conclusions'):
            continue  # these get one appended line / subsection
        assert body.strip() in report


def test_removed_sections_absent():
    report = _run(_step6_markdown(with_genes=False))
    for gone in ('Key Meta-Analysis', 'Enhanced Significance', 'Cross-Dataset Pathway Convergence',
                 'Novel Pathway Co-occurrence', 'Reproducibility Overview',
                 'Direction Concordance', 'Biological Interpretation of Meta-Analysis',
                 'Cross-Study Validated', 'Emergent Theme', 'Weak Signal', 'Meta-Analysis Insights',
                 'Meta-Unique', 'Power Gain', 'old interpretation', 'Recommendations'):
        assert gone not in report, gone


def test_meta_section_renamed_and_no_beyond_or_studies():
    report = _run(_step6_markdown(with_genes=False))
    assert '## 1. Findings Across Individual Datasets' in report
    assert '### Dataset Statistics' in report
    assert '### Summary of Dataset Findings' in report
    assert 'What Individual Studies Found' not in report
    assert 'BEYOND' not in report
    section = report.split('## 1. Findings Across Individual Datasets')[1].split('\n## ')[0]
    assert 'stud' not in section.lower()


def test_top_pathways_table_labels_cap():
    report = _run(_step6_markdown(with_genes=False))
    assert 'Showing top 15 of 30 pathways' in report


def test_meta_facts_line_in_executive_summary():
    report = _run(_step6_markdown(with_genes=False))
    exec_summary = report.split('## Executive Summary')[1].split('\n## ')[0]
    assert ('**Meta-analysis scope:** 14 datasets pooled; 22 of 40 selected pathways are '
            'significant (FDR < 0.05) in two or more datasets.') in exec_summary


def test_meta_facts_line_missing_values_do_not_raise():
    md = _step6_markdown(with_genes=False)
    meta = _meta_results(md)
    del meta['input_summary']
    report = _run(md, meta_results=meta)
    assert 'the number of selected pathways is unavailable' in report

    line = MetaStep08ComparativeReport._meta_facts_line({}, None)
    assert 'number of pooled datasets unavailable' in line
    assert 'per-dataset pathway frequencies unavailable' in line


def test_meta_facts_line_flags_unrestricted_lists():
    line = MetaStep08ComparativeReport._meta_facts_line(
        {'total_datasets': 14, 'pathways_in_multiple': 48, 'total_unique_pathways': 54}, 40)
    assert 'not restricted to the selection' in line
    assert 'of 40 selected' not in line


def test_conclusions_gets_dataset_subsection():
    report = _run(_step6_markdown(with_genes=False))
    conclusions = report.split('Summary and Conclusions')[1].split('\n## ')[0]
    assert '### Across Individual Datasets' in conclusions
    assert 'significant in two or more datasets: 22' in conclusions
    assert 'significant in no individual dataset: 10 (of 40 selected)' in conclusions
    assert 'generated 1 mechanistic hypothesis ' in conclusions


def test_conclusions_says_when_no_hypotheses():
    md = _step6_markdown(with_genes=False)
    report = _run(md, meta_results=_meta_results(md, hypotheses=0))
    assert 'No mechanistic hypotheses were generated from the combined data.' in report


def test_missing_step6_report_gives_explicit_notice():
    meta = {'input_summary': {'num_pathways': 40}, 'steps': {}, 'error': 'boom'}
    report = _run('', meta_results=meta)
    assert 'The combined-data report was not produced' in report
    assert 'boom' in report
    assert '## 1. Findings Across Individual Datasets' in report


def test_heading_variants_are_classified():
    md = '\n'.join([
        '# Title', 'header',
        '## Executive Summary', 'e',
        '## 1. Pathway Theme Analysis', 't',
        '## Pathway Mechanisms (report_section)', 'm',
        '## Mechanistic Hypotheses', 'h',
        '## 6. Summary and Conclusions', 'c',
        '## Appendices', 'a',
    ])
    kinds = [k for k, _, _ in MetaStep08ComparativeReport._split_markdown_sections(md)]
    assert kinds == ['executive_summary', 'themes', 'mechanisms', 'hypotheses',
                     'conclusions', 'appendices']


def test_unrecognised_section_is_kept_in_position():
    md = '\n'.join([
        '## Executive Summary', 'e',
        '## 1. Pathway Theme Analysis', 't',
        '## Something New', 'keep me',
        '## 4. Mechanistic Hypotheses and Predictions', 'h',
        '## 6. Summary and Conclusions', 'c',
        '## Appendices', 'a',
    ])
    headings = _h2(_run(md, meta_results=_meta_results(md)))
    assert headings == ['Executive Summary', '1. Findings Across Individual Datasets',
                        '2. Pathway Theme Analysis', '3. Something New',
                        '4. Mechanistic Hypotheses and Predictions',
                        '5. Summary and Conclusions', 'Appendices']
