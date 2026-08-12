"""
Tests for report structural consistency (Group F):
- Step 5 removed from the pipeline and the report
- master-regulator count consistent between executive summary and hub section
"""

from unittest.mock import MagicMock

from src.pipeline.steps.step06_report_generation import Step06ReportGeneration


def _bare_step06():
    step = Step06ReportGeneration.__new__(Step06ReportGeneration)
    step.llm = MagicMock()
    return step




def test_methodology_appendix_has_no_therapeutic_step():
    """The old Therapeutic Step 5 must be gone; Step 5 is now report generation."""
    step = _bare_step06()
    section = step._generate_appendices(
        input_data={'metadata': {'organism': 'Homo sapiens'}},
        step1={'themes': []}, step2={'network_hubs': []}, step3={}, step4={},
    )
    text = section.content
    assert 'Therapeutic' not in text
    assert 'no gene expression data' not in text
    assert 'Step 4' in text
    assert 'Step 5' in text and 'report generation' in text.lower()  # report is Step 5


def test_executive_summary_master_regulator_count_consistent():
    step = _bare_step06()
    step.llm.chat.return_value = 'A concise executive summary.'
    hubs = [{'gene': f'G{i}'} for i in range(20)]
    masters = [{'gene': g} for g in ['CDCA8', 'CDK1', 'UBE2T', 'MKI67', 'CCNB1']]
    section = step._generate_executive_summary(
        study_context='disease: HCC',
        input_data={},
        step1={'themes': [{'name': 'Cell cycle control'}]},
        step2={'network_hubs': hubs, 'master_regulators': masters},
        step3={}, step4={'hypotheses': [1, 2, 3, 4]},
    )
    content = section.content
    # All 5 master regulators listed, count says "5 of 20" (not the old "3 of 20")
    assert '5 of 20 hub genes' in content
    for g in ['CDCA8', 'CDK1', 'UBE2T', 'MKI67', 'CCNB1']:
        assert g in content
    # No therapeutics/drug line now that Step 5 is removed
    assert 'Druggable targets' not in content


def test_execute_takes_no_step5_output():
    """The removed Therapeutics step must leave no vestigial parameter on execute."""
    import inspect
    sig = inspect.signature(Step06ReportGeneration.execute)
    assert 'step5_output' not in sig.parameters


def test_pipeline_steps_contiguous_and_report_is_step5():
    """Report generation is the final step (Step 5), numbering is contiguous, and no
    Therapeutic step remains."""
    from src.api import job_manager as jm
    assert [s['step'] for s in jm.PIPELINE_STEPS] == [1, 2, 3, 4, 5]
    assert 'Therapeutic' not in ' '.join(s['name'] for s in jm.PIPELINE_STEPS)
    last = jm.PIPELINE_STEPS[-1]
    assert last['step'] == 5 and 'Report' in last['name']
    assert [s['step'] for s in jm.META_PIPELINE_STEPS] == [1, 2, 3, 4, 5, 6, 7]


def test_set_current_step_message_by_number_not_position():
    """R2-2: with non-contiguous steps [1,2,3,4,6], step 6's message must resolve by
    the 'step' field, not list position."""
    from src.api.job_manager import JobManager
    jmgr = JobManager()
    steps = [{"step": 1, "name": "A", "message": "m1"},
             {"step": 2, "name": "B", "message": "m2"},
             {"step": 4, "name": "D", "message": "m4"},
             {"step": 6, "name": "F", "message": "report-msg"}]
    job_id = jmgr.create_job({"pathways": []}, total_steps=len(steps), steps_info=steps)
    jmgr.set_current_step(job_id, 6)
    assert jmgr.jobs[job_id].current_step_message == "report-msg"


def test_execute_end_to_end_omits_step5():
    """Integration: full report assembles without step5_output, omits therapeutics,
    and does not raise. (NES labelling is Step-3's job — covered in test_report_hygiene.)"""
    step = _bare_step06()
    step.llm.chat.return_value = 'Narrative text.'
    input_data = {
        'metadata': {'organism': 'Homo sapiens', 'disease': 'HCC'},
        'genes': [{'gene': 'CDK1', 'foldChange': 2.4}],
        'pathways': [{'name': 'Cell cycle', 'NES': 1.77},
                     {'name': 'Fatty acid degradation', 'NES': -3.4}],
    }
    step1 = {'themes': [{'name': 'Cell cycle control', 'significance': 'high',
                         'pathways': [{'name': 'Cell cycle', 'p_value_fdr': 1e-4, 'gene_count': 40}],
                         'pathway_count': 1, 'shared_genes': ['CDK1'], 'shared_gene_count': 1,
                         'avg_jaccard_overlap': 0.3, 'key_genes': ['CDK1'],
                         'biological_context': 'ctx', 'avg_p_value_fdr': 1e-4}],
             'ungrouped': []}
    step2 = {'network_hubs': [{'gene': 'CDK1', 'degree': 40, 'hub_score': 0.8, 'fold_change': 2.4}],
             'master_regulators': [{'gene': 'CDK1', 'hub_score': 0.8, 'degree': 40,
                                    'fold_change': 2.4, 'role': 'kinase', 'pathways': []}]}
    step3 = {'pathwayMechanisms': [{'pathway': 'Cell cycle', 'biologicalFunction': 'division',
                                    'deGeneInvolvement': [{'gene': 'CDK1', 'foldChange': 2.4,
                                                           'roleInPathway': 'driver'}]}],
             'report_section': ''}
    step4 = {'hypotheses': [{'statement': 'H1', 'confidence': 'HIGH'}]}

    report = step.execute(input_data=input_data, step1_output=step1, step2_output=step2,
                          step3_output=step3, step4_output=step4)
    md = report.markdown_content if hasattr(report, 'markdown_content') else report.get('markdown_content', '')
    assert 'Therapeutic Implications' not in md
    assert 'Therapeutic' not in md            # no leftover therapeutics content
    assert 'Druggable targets' not in md
    assert 'Network Hub Genes' in md or 'hub genes' in md.lower()
