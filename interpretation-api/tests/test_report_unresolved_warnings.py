"""
Item 26 (main L214): unresolved validation warnings are printed in a report appendix.
"""

from unittest.mock import MagicMock

from src.pipeline.steps.step06_report_generation import Step06ReportGeneration


def _step():
    step = Step06ReportGeneration.__new__(Step06ReportGeneration)
    step.llm = MagicMock()
    return step


def test_appendix_lists_unresolved_warnings():
    step4 = {'metadata': {
        'direction_validation_errors': ['Hypothesis 3: SLC27A5 is DOWN-regulated (FC=-1.31) but described as up-regulated'],
        'kegg_validation_warnings': ['Hypothesis 1: 1 unsupported claim(s): E2F1 -> CCNA2'],
        'fold_change_sign_conflicts': ['CDK1'],
        'numerical_validation_warnings': [],
    }}
    section = _step()._generate_appendices({'metadata': {}}, {}, {}, {}, step4, {'metadata': {'skipped': True}})
    text = section.content
    assert '### C. Unresolved Validation Warnings' in text
    assert 'SLC27A5 is DOWN-regulated' in text
    assert 'Relations not returned by the tool calls' in text
    assert 'CDK1: cited with a direction opposite to its fold change' in text
    assert 'Numerical claims' not in text  # empty groups are omitted


def test_no_appendix_without_warnings():
    section = _step()._generate_appendices({'metadata': {}}, {}, {}, {}, {'metadata': {}},
                                           {'metadata': {'skipped': True}})
    assert 'Unresolved Validation Warnings' not in section.content


def test_summary_unresolved_numbers_are_listed():
    step = _step()
    step._summary_unresolved = [{'text': 'NES = +2.50', 'reason': 'sign differs'}]
    section = step._generate_appendices({'metadata': {}}, {}, {}, {}, {'metadata': {}},
                                        {'metadata': {'skipped': True}})
    assert '"NES = +2.50": sign differs' in section.content
