import argparse
import logging
import unittest
import yaml

import numpy as np

# from ap import compute_average_precision
from ap_voc import calc_pr_ovr_noref as compute_average_precision


class ApTest(unittest.TestCase):
    pass


def test_generator(groundtruth, predictions, expected_ap):
    def test(self):
        ap = compute_average_precision(np.asarray(groundtruth),
                                       np.asarray(predictions))[3][0]
        self.assertAlmostEqual(ap, expected_ap)
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

    unittest.main()
