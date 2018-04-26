"""Test case for ap.py. Run from root directory of this repository."""

import argparse
import logging
import unittest
import yaml

import numpy as np

from ap import compute_average_precision, compute_multiple_aps


class ApTest(unittest.TestCase):
    pass


class MultipleApTest(unittest.TestCase):
    pass


def test_generator(groundtruth, predictions, expected_ap):
    def test(self):
        ap = compute_average_precision(np.asarray(groundtruth),
                                       np.asarray(predictions))
        self.assertAlmostEqual(ap, expected_ap)
    return test


def multiple_ap_test_generator(groundtruth, predictions, expected_aps):
    def test(self):
        aps = compute_multiple_aps(np.asarray(groundtruth),
                                   np.asarray(predictions))
        for i, (ap, expected_ap) in enumerate(zip(aps, expected_aps)):
            self.assertAlmostEqual(
                ap,
                expected_ap,
                msg='Incorrect AP for label %s. Expected: %s, Saw: %s' %
                (i, expected_ap, ap))

    return test


if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring
    # exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('yaml_config', nargs='?', default='test_cases.yaml')

    args = parser.parse_args()

    with open(args.yaml_config, 'r') as f:
        tests = yaml.load(f.read())
    for test in tests:
        test_fn = test_generator(test['groundtruth'], test['predictions'],
                                 test['expected_ap'])
        setattr(ApTest, 'test_' + test['name'], test_fn)

    num_labels = 3
    for test in tests:
        num_samples = len(test['groundtruth'])
        predictions = np.zeros((num_samples, num_labels))
        groundtruth = np.zeros((num_samples, num_labels))
        for i in range(num_labels):
            predictions[:, i] = test['predictions']
            groundtruth[:, i] = test['groundtruth']
        expected_aps = [test['expected_ap']] * num_samples
        setattr(MultipleApTest, 'test_multi_' + test['name'],
                multiple_ap_test_generator(groundtruth, predictions,
                                           expected_aps))
    unittest.main()
