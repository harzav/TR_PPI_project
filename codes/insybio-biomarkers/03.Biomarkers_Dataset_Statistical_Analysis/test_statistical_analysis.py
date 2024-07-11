# content of test test_statistical_analysis.py
import unittest
import pytest
from pytest import approx
from pytest_steps import test_steps, optional_step
from os import listdir
from os.path import isfile, join
from pathlib import Path
import pkg_resources
import warnings

import biomarkers_dataset_statistical_analysis as bdsa
import biomarkers_dataset_statistical_analysis_schedule as bdsas

_REQUIREMENTS_PATH = Path(__file__).with_name("requirements.txt")


class TestRequirements(unittest.TestCase):
    """Test availability of required packages."""

    def test_requirements(self):
        """Test that each required package is available."""
        # Ref: https://stackoverflow.com/a/45474387/
        requirements = pkg_resources.parse_requirements(_REQUIREMENTS_PATH.open())
        for requirement in requirements:
            requirement = str(requirement)
            with self.subTest(requirement=requirement):
                pkg_resources.require(requirement)


class TestMain:
    #@test_steps('step_main', 'step_checkfiles')
    @pytest.mark.parametrize("jobparams, expected_result",
                             [({"biomarkers_dataset": "/opt/backend-application/insybio-biomarkers"
                                                      "/03.Biomarkers_Dataset_Statistical_Analysis/testfiles/input/"
                                                      "preprocessed_data1621349122_5915.txt",
                                "labels_filename": "/opt/backend-application/insybio-biomarkers"
                                                   "/03.Biomarkers_Dataset_Statistical_Analysis/testfiles/input/"
                                                   "dsfile1621348633_5346.txt",
                                "selected_comorbidities_string": "", "paired_flag": "0",
                                "outputpath": "/opt/backend-application/insybio-biomarkers/"
                                              "03.Biomarkers_Dataset_Statistical_Analysis/testfiles/output/SA_0/",
                                "parametric_flag": "1", "logged_flag": "0", "filetype": "18",
                                "has_features_header": "1", "has_samples_header": "1", "pvalue_threshold": "0.05"},
                               1),
                              ({"biomarkers_dataset": "/opt/backend-application/insybio-biomarkers"
                                                      "/03.Biomarkers_Dataset_Statistical_Analysis/testfiles/input/"
                                                      "preprocessed_data1621349122_5915.txt",
                                "labels_filename": "/opt/backend-application/insybio-biomarkers"
                                                   "/03.Biomarkers_Dataset_Statistical_Analysis/testfiles/input/"
                                                   "dsfile1621348633_5346.txt",
                                "selected_comorbidities_string": "", "paired_flag": "0",
                                "outputpath": "/opt/backend-application/insybio-biomarkers/"
                                              "03.Biomarkers_Dataset_Statistical_Analysis/testfiles/output/SA_1/",
                                "parametric_flag": "2", "logged_flag": "0", "filetype": "18",
                                "has_features_header": "1", "has_samples_header": "1", "pvalue_threshold": "0.5"},
                               1),
                              ({"biomarkers_dataset": "/opt/backend-application/insybio-biomarkers"
                                                      "/03.Biomarkers_Dataset_Statistical_Analysis/testfiles/input/"
                                                      "preprocessed_data1621349122_5915.txt",
                                "labels_filename": "/opt/backend-application/insybio-biomarkers"
                                                   "/03.Biomarkers_Dataset_Statistical_Analysis/testfiles/input/"
                                                   "dsfile1621348633_5346.txt",
                                "selected_comorbidities_string": "", "paired_flag": "0",
                                "outputpath": "/opt/backend-application/insybio-biomarkers/"
                                              "03.Biomarkers_Dataset_Statistical_Analysis/testfiles/output/SA_2/",
                                "parametric_flag": "3", "logged_flag": "0", "filetype": "18",
                                "has_features_header": "1", "has_samples_header": "1", "pvalue_threshold": "0.5"},
                               1)
                              ])
    def test_statistical_analysis(self, jobparams, expected_result, tmp_path):
        # Step Main Run
        d = tmp_path / "stat/"
        # d = jobparams['outputpath']
        try:
            d.mkdir()
        except FileExistsError:
            print('File exists')
        directory = str(d) + "/"
        jobparams = bdsas.check_image_parameters(jobparams, 0, 0, 'test')
        result = bdsa.meta_statistical_analysis(
            jobparams['biomarkers_dataset'], jobparams['labels_filename'], jobparams['selected_comorbidities_string'],
            directory, int(jobparams['filetype']), int(jobparams['has_features_header']),
            int(jobparams['has_samples_header']), int(jobparams['paired_flag']), int(jobparams['logged_flag']),
            float(jobparams['pvalue_threshold']), jobparams['parametric_flag'], int(jobparams['volcano_width']),
            int(jobparams["volcano_height"]), int(jobparams["volcano_titles"]), int(jobparams["volcano_axis_labels"]),
            int(jobparams["volcano_labels"]), float(jobparams["volcano_axis_relevance"]),
            int(jobparams["volcano_criteria"]), float(jobparams["abs_log_fold_changes_threshold"]),
            int(jobparams["volcano_labeled"]), int(jobparams["heatmap_width"]), int(jobparams["heatmap_height"]),
            jobparams["features_hier"], jobparams["features_metric"], jobparams["features_linkage"],
            jobparams["samples_hier"], jobparams["samples_metric"], jobparams["samples_linkage"],
            int(jobparams["heatmap_zscore_bar"]), int(jobparams["beanplot_width"]), int(jobparams["beanplot_height"]),
            float(jobparams["beanplot_axis"]), float(jobparams["beanplot_xaxis"]), float(jobparams["beanplot_yaxis"]),
            float(jobparams["beanplot_titles"]), float(jobparams["beanplot_axis_titles"]), 'test', 0, 0)

        assert result[0] == expected_result, result[1]
        # yield

        # with optional_step('step_checkfiles') as step_checkfiles:
        # Step Check Files
        """try:
            onlyfiles = [f for f in listdir(jobparams['outputpath']) if isfile(join(jobparams['outputpath'], f))]
        except FileNotFoundError:
            warnings.warn('No folder')
            print('No folder')
        else:
            for file in onlyfiles:
                print('Asserting file: {}'.format(file))
                with open(join(directory, file)) as test, open(join(jobparams['outputpath'], file)) as expected:
                    assert test.read() == expected.read(), 'failed'"""

        onlyfiles = [f for f in listdir(jobparams['outputpath']) if isfile(join(jobparams['outputpath'], f))]

        for file in onlyfiles:
            print('Asserting file: {}'.format(file))
            if '.png' not in file and '.zip' not in file:
                with open(join(directory, file)) as test, open(join(jobparams['outputpath'], file)) as expected:

                    for line1 in test:
                        for line2 in expected:
                            line1 = line1.split('\t')
                            line2 = line2.split('\t')
                            assert line1[0] == line2[0]
                            for i in range(len(line1)):
                                if self.is_number(line1[i]) and self.is_number(line2[i]):
                                    assert float(line1[i]) == approx(float(line2[i])), \
                                        "{} Not same line {}\n{}".format(file, line1, line2)
                                else:
                                    assert line1[i] == line2[i], "Not same line {}\n{}".format(line1, line2)
                            break

        # yield step_checkfiles

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
