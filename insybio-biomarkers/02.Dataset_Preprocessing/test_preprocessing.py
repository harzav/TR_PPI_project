# content of test test_preprocessing.py
import unittest
import pytest
from pytest import approx
from pytest_steps import test_steps, optional_step
from os import listdir
from os.path import isfile, join
from pathlib import Path
import pkg_resources

import dataset_preprocessing as dp

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
    @test_steps('step_main', 'step_checkfiles')
    @pytest.mark.parametrize("jobparams, expected_result",
                             [({"input_dataset": "/opt/backend-application/insybio-biomarkers/02.Dataset_Preprocessing"
                                                 "/testfiles/input/training_regr_totaldrugs_dataset.txt",
                                "selected_features_string": "",
                                "outputpath": "/opt/backend-application/insybio-biomarkers/02.Dataset_Preprocessing"
                                              "/testfiles/output/Pre_0",
                                "missing_threshold": "0.05", "missing_imputation_method": "1",
                                "normalization_method": "1", "filetype": "18", "has_features_header": "1",
                                "has_samples_header": "1"},
                               1),
                              ({"input_dataset": "/opt/backend-application/insybio-biomarkers/02.Dataset_Preprocessing"
                                                 "/testfiles/input/training_dataset_multiclass.txt",
                                "selected_features_string": "",
                                "outputpath": "/opt/backend-application/insybio-biomarkers/02.Dataset_Preprocessing"
                                              "/testfiles/output/Pre_1",
                                "missing_threshold": "0.05", "missing_imputation_method": "2",
                                "normalization_method": "2", "filetype": "18", "has_features_header": "1",
                                "has_samples_header": "1"},
                               1),
                              ({"input_dataset": "/opt/backend-application/insybio-biomarkers/02.Dataset_Preprocessing"
                                                 "/testfiles/input/training_dataset_multiclass.txt",
                                "selected_features_string": "",
                                "outputpath": "/opt/backend-application/insybio-biomarkers/02.Dataset_Preprocessing"
                                              "/testfiles/output/Pre_2",
                                "missing_threshold": "0.05", "missing_imputation_method": "2",
                                "normalization_method": "2", "filetype": "18", "has_features_header": "0",
                                "has_samples_header": "0"},
                               0)
                              ])
    def test_differential_expression(self, jobparams, expected_result, tmp_path):
        # Step Main Run
        d = tmp_path / "preprocess/"
        # d = jobparams['outputpath']
        try:
            d.mkdir()
        except FileExistsError:
            print('File exists')
        directory = str(d) + "/"
        result = dp.preprocess_data(jobparams['input_dataset'], jobparams['selected_features_string'],
                                    directory, "preprocessed_data.txt",
                                    float(jobparams['missing_threshold']), int(jobparams['missing_imputation_method']),
                                    int(jobparams['normalization_method']), int(jobparams['filetype']),
                                    int(jobparams['has_features_header']), int(jobparams['has_samples_header']), 'test',
                                    0, 0)

        assert result[0] == expected_result, result[1]
        yield

        with optional_step('step_checkfiles') as step_checkfiles:
            # Step Check Files
            try:
                onlyfiles = [f for f in listdir(jobparams['outputpath']) if isfile(join(jobparams['outputpath'], f))]
            except FileNotFoundError:
                print('No folder')
                pass
            else:
                for file in onlyfiles:
                    print('Asserting file: {}'.format(file))
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
        yield step_checkfiles

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False