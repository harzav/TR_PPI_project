# content of test test_testing.py
import unittest
import pytest
from pytest import approx
from pytest_steps import test_steps, optional_step
from os import listdir
from os.path import isfile, join
from pathlib import Path
import pkg_resources
import warnings

import testing_multibiomarker_predictive_analytics_model_backend as model_testing

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
                             [({"model_filename": "/opt/backend-application/insybio-biomarkers/"
                                                  "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                  "input/TE_0/models_321.zip",
                                "testset_filename": "/opt/backend-application/insybio-biomarkers/"
                                                    "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                    "input/TE_0/test_dataset_with_headers.txt",
                                "testset_labels_filename": "/opt/backend-application/insybio-biomarkers/"
                                                           "05.Testing_Multibiomarker_Predictive_Analytics_Model/"
                                                           "testfiles/input/TE_0/test_labels.txt",
                                "normalization_method": "1", "missing_imputation_method": "1",
                                "data_been_preprocessed_flag": "1", "has_samples_header": "1",
                                "has_features_header": "1", "filetype": 21, "variables_for_normalization_string": "",
                                "training_labels_filename": "/opt/backend-application/insybio-biomarkers/"
                                                            "05.Testing_Multibiomarker_Predictive_Analytics_Model/"
                                                            "testfiles/input/TE_0/training_labels.txt",
                                "features_filename": "/opt/backend-application/insybio-biomarkers/"
                                                     "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                     "input/TE_0/features_list.txt", "selection_flag": "2",
                                "selected_comorbidities_string": "",
                                "length_of_features_from_training_filename":
                                    "/opt/backend-application/insybio-biomarkers/"
                                    "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                    "input/TE_0/length_of_features_from_training.txt",
                                "minimums_filename": "", "maximums_filename": "", "length_of_features_filename": "",
                                "averages_filename": "",
                                "outputpath": "/opt/backend-application/insybio-biomarkers/"
                                              "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/output/"
                                              "TE_0/"},
                               1),
                              ({"model_filename": "/opt/backend-application/insybio-biomarkers/"
                                                  "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                  "input/TE_1/models_335.zip",
                                "testset_filename": "/opt/backend-application/insybio-biomarkers/"
                                                    "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                    "input/TE_1/dsfile1618562734_9114.csv",
                                "testset_labels_filename": "/opt/backend-application/insybio-biomarkers/"
                                                           "05.Testing_Multibiomarker_Predictive_Analytics_Model/"
                                                           "testfiles/input/TE_1/dsfile1618562788_7762.txt",
                                "normalization_method": "1", "missing_imputation_method": "1",
                                "data_been_preprocessed_flag": "0", "has_samples_header": "1",
                                "has_features_header": "1", "filetype": 21, "variables_for_normalization_string": "",
                                "training_labels_filename": "/opt/backend-application/insybio-biomarkers/"
                                                            "05.Testing_Multibiomarker_Predictive_Analytics_Model/"
                                                            "testfiles/input/TE_1/training_labels.txt",
                                "features_filename": "/opt/backend-application/insybio-biomarkers/"
                                                     "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                     "input/TE_1/features_list.txt", "selection_flag": "0",
                                "selected_comorbidities_string": "",
                                "length_of_features_from_training_filename":
                                    "/opt/backend-application/insybio-biomarkers/"
                                    "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                    "input/TE_1/length_of_features_from_training.txt",
                                "minimums_filename": "/opt/backend-application/insybio-biomarkers/"
                                                     "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                     "input/TE_1/minimums.txt",
                                "maximums_filename": "/opt/backend-application/insybio-biomarkers/"
                                                     "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                     "input/TE_1/maximums.txt",
                                "length_of_features_filename": "/opt/backend-application/insybio-biomarkers/"
                                                     "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                     "input/TE_1/length_of_features.txt",
                                "averages_filename": "",
                                "outputpath": "/opt/backend-application/insybio-biomarkers/"
                                              "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/output/"
                                              "TE_1/"},
                               1),
                              ({"model_filename": "/opt/backend-application/insybio-biomarkers/"
                                                  "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                  "input/TE_2/models_336.zip",
                                "testset_filename": "/opt/backend-application/insybio-biomarkers/"
                                                    "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                    "input/TE_2/dsfile1623851846_5662.txt",
                                "testset_labels_filename": "/opt/backend-application/insybio-biomarkers/"
                                                           "05.Testing_Multibiomarker_Predictive_Analytics_Model/"
                                                           "testfiles/input/TE_2/dsfile1623851866_1311.txt",
                                "normalization_method": "1", "missing_imputation_method": "1",
                                "data_been_preprocessed_flag": "1", "has_samples_header": "1",
                                "has_features_header": "1", "filetype": 21, "variables_for_normalization_string": "",
                                "training_labels_filename": "/opt/backend-application/insybio-biomarkers/"
                                                            "05.Testing_Multibiomarker_Predictive_Analytics_Model/"
                                                            "testfiles/input/TE_2/training_labels.txt",
                                "features_filename": "/opt/backend-application/insybio-biomarkers/"
                                                     "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                     "input/TE_2/features_list.txt", "selection_flag": "1",
                                "selected_comorbidities_string": "",
                                "length_of_features_from_training_filename":
                                    "/opt/backend-application/insybio-biomarkers/"
                                    "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                    "input/TE_2/length_of_features_from_training.txt",
                                "minimums_filename": "/opt/backend-application/insybio-biomarkers/"
                                                     "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                     "input/TE_2/minimums.txt",
                                "maximums_filename": "/opt/backend-application/insybio-biomarkers/"
                                                     "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                     "input/TE_2/maximums.txt",
                                "length_of_features_filename": "/opt/backend-application/insybio-biomarkers/"
                                                     "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                     "input/TE_2/length_of_features.txt",
                                "averages_filename": "",
                                "outputpath": "/opt/backend-application/insybio-biomarkers/"
                                              "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/output/"
                                              "TE_2/"},
                               1),
                              ({"model_filename": "/opt/backend-application/insybio-biomarkers/"
                                                  "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                  "input/TE_2/models_336.zip",
                                "testset_filename": "/opt/backend-application/insybio-biomarkers/"
                                                    "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                    "input/TE_1/dsfile1618562734_9114.csv",
                                "testset_labels_filename": "/opt/backend-application/insybio-biomarkers/"
                                                           "05.Testing_Multibiomarker_Predictive_Analytics_Model/"
                                                           "testfiles/input/TE_1/dsfile1618562788_7762.txt",
                                "normalization_method": "1", "missing_imputation_method": "1",
                                "data_been_preprocessed_flag": "0", "has_samples_header": "1",
                                "has_features_header": "1", "filetype": 21, "variables_for_normalization_string": "",
                                "training_labels_filename": "/opt/backend-application/insybio-biomarkers/"
                                                            "05.Testing_Multibiomarker_Predictive_Analytics_Model/"
                                                            "testfiles/input/TE_1/training_labels.txt",
                                "features_filename": "/opt/backend-application/insybio-biomarkers/"
                                                     "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                     "input/TE_1/features_list.txt", "selection_flag": "1",
                                "selected_comorbidities_string": "",
                                "length_of_features_from_training_filename":
                                    "/opt/backend-application/insybio-biomarkers/"
                                    "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                    "input/TE_1/length_of_features_from_training.txt",
                                "minimums_filename": "/opt/backend-application/insybio-biomarkers/"
                                                     "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                     "input/TE_1/minimums.txt",
                                "maximums_filename": "/opt/backend-application/insybio-biomarkers/"
                                                     "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                     "input/TE_1/maximums.txt",
                                "length_of_features_filename": "/opt/backend-application/insybio-biomarkers/"
                                                     "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                     "input/TE_1/length_of_features.txt",
                                "averages_filename": "",
                                "outputpath": "/opt/backend-application/insybio-biomarkers/"
                                              "05.Testing_Multibiomarker_Predictive_Analytics_Model/testfiles/output/"
                                              "TE_3/"},
                               0)
                              ])
    def test_training(self, jobparams, expected_result, tmp_path):
        # Step Main Run
        d = tmp_path / "test/"
        # d = jobparams['outputpath']
        try:
            d.mkdir()
        except FileExistsError:
            print('File exists')
        directory = str(d) + "/"

        result = model_testing.run_all(jobparams['testset_filename'], jobparams['testset_labels_filename'],
                                       jobparams['maximums_filename'], jobparams['minimums_filename'],
                                       jobparams['averages_filename'], jobparams['features_filename'],
                                       int(jobparams['missing_imputation_method']),
                                       int(jobparams['normalization_method']),
                                       jobparams['model_filename'], int(jobparams['selection_flag']),
                                       int(jobparams['data_been_preprocessed_flag']),
                                       jobparams['variables_for_normalization_string'], int(jobparams['filetype']),
                                       int(jobparams['has_features_header']), int(jobparams['has_samples_header']),
                                       jobparams['training_labels_filename'], jobparams['length_of_features_filename'],
                                       jobparams['length_of_features_from_training_filename'],
                                       directory, jobparams['selected_comorbidities_string'], 4, 'test', 0, 0)

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
                onlyfiles = [f for f in listdir(jobparams['outputpath'] + 'models/')
                             if isfile(join(jobparams['outputpath'] + 'models/', f))]
                for file in onlyfiles:
                    if isfile(join(directory, 'models/', file)):
                        pass
                    else:
                        pytest.fail("Model found: {}".format(file))

        yield step_checkfiles

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
