# content of test test_training.py
import unittest
import pytest
from pytest import approx
from pytest_steps import test_steps, optional_step
from os import listdir
from os.path import isfile, join
from pathlib import Path
import pkg_resources
import warnings

import biomarker_discovery_script_selection_backend as model_training

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
                             [({"dataset_filename": "/opt/backend-application/insybio-biomarkers/"
                                                    "04.Training_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                    "input/dsfile1618317841_5210.txt",
                                "labels_filename": "/opt/backend-application/insybio-biomarkers/"
                                                    "04.Training_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                    "input/dsfile1618239588_5682.txt",
                                "split_dataset_flag": "1", "filtering_percentage": "0.3", "selection_flag": "2",
                                "goal_significances_string": "1,10,10,1,20,1,3,1", "selected_comorbidities_string": "",
                                "has_features_header": "1", "has_samples_header": "1", "logged_flag": "0",
                                "population": "20", "generations": "20", "mutation_probability": "0.01",
                                "arithmetic_crossover_probability": "0", "two_points_crossover_probability": "0.9",
                                "num_of_folds": "5", "outputpath": "/opt/backend-application/insybio-biomarkers/"
                                                    "04.Training_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                    "output/TR_0/"},
                               1),
                              ({"dataset_filename": "/opt/backend-application/insybio-biomarkers/"
                                                    "04.Training_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                    "input/preprocessed_data_334.txt",
                                "labels_filename": "/opt/backend-application/insybio-biomarkers/"
                                                    "04.Training_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                    "input/dsfile1621348633_5346.txt",
                                "split_dataset_flag": "0", "filtering_percentage": "0.3", "selection_flag": "0",
                                "goal_significances_string": "1,10,10,1,20,1,3,1", "selected_comorbidities_string": "",
                                "has_features_header": "1", "has_samples_header": "1", "logged_flag": "0",
                                "population": "20", "generations": "20", "mutation_probability": "0.01",
                                "arithmetic_crossover_probability": "0", "two_points_crossover_probability": "0.9",
                                "num_of_folds": "5", "outputpath": "/opt/backend-application/insybio-biomarkers/"
                                                    "04.Training_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                    "output/TR_1/"},
                               1),
                              ({"dataset_filename": "/opt/backend-application/insybio-biomarkers/"
                                                    "02.Dataset_Preprocessing/testfiles/output/Pre_0/"
                                                    "preprocessed_data.txt",
                                "labels_filename": "/opt/backend-application/insybio-biomarkers/"
                                                    "04.Training_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                    "input/training_regression_labels.txt",
                                "split_dataset_flag": "0", "filtering_percentage": "0.3", "selection_flag": "1",
                                "goal_significances_string": "1,10,10,1,20,1,3,1", "selected_comorbidities_string": "",
                                "has_features_header": "1", "has_samples_header": "1", "logged_flag": "0",
                                "population": "20", "generations": "20", "mutation_probability": "0.01",
                                "arithmetic_crossover_probability": "0", "two_points_crossover_probability": "0.9",
                                "num_of_folds": "5", "outputpath": "/opt/backend-application/insybio-biomarkers/"
                                                    "04.Training_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                    "output/TR_2/"},
                               1),
                              ({"dataset_filename": "/opt/backend-application/insybio-biomarkers/"
                                                    "04.Training_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                    "input/preprocessed_data_327.txt",
                                "labels_filename": "/opt/backend-application/insybio-biomarkers/"
                                                   "04.Training_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                                   "input/dsfile1618239588_5682.txt",
                                "split_dataset_flag": "1", "filtering_percentage": "0.3", "selection_flag": "1",
                                "goal_significances_string": "1,10,10,1,20,1,3,1", "selected_comorbidities_string": "",
                                "has_features_header": "1", "has_samples_header": "1", "logged_flag": "0",
                                "population": "20", "generations": "20", "mutation_probability": "0.01",
                                "arithmetic_crossover_probability": "0", "two_points_crossover_probability": "0.9",
                                "num_of_folds": "5",
                                "outputpath": "/opt/backend-application/insybio-biomarkers/"
                                              "04.Training_Multibiomarker_Predictive_Analytics_Model/testfiles/"
                                              "output/TR_3/"},
                               0)
                              ])
    def test_training(self, jobparams, expected_result, tmp_path):
        # Step Main Run
        d = tmp_path / "train/"
        # d = jobparams['outputpath']
        try:
            d.mkdir()
        except FileExistsError:
            print('File exists')
        directory = str(d) + "/"

        result = model_training.run_model_and_splitter_selectors(
            jobparams['dataset_filename'], jobparams['labels_filename'], float(jobparams['filtering_percentage']),
            int(jobparams['selection_flag']), int(jobparams['split_dataset_flag']),
            jobparams['goal_significances_string'], jobparams['selected_comorbidities_string'],
            18, int(jobparams['has_features_header']), int(jobparams['has_samples_header']),
            directory, int(jobparams['logged_flag']), int(jobparams['population']),
            int(jobparams['generations']), float(jobparams['mutation_probability']),
            float(jobparams['arithmetic_crossover_probability']), float(jobparams['two_points_crossover_probability']),
            int(jobparams['num_of_folds']), 'test', 0, 0, 4)

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
                        assert test.read() == expected.read(), file
        yield step_checkfiles

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
