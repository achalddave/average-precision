from __future__ import division

import numpy as np

def compute_average_precision(groundtruth, predictions):
    """
    Computes average precision for a binary problem. This is based off of the
    PASCAL VOC implementation.

    Args:
        groundtruth (array-like): Binary vector indicating whether each sample
            is positive or negative.
        predictions (array-like): Contains scores for each sample.

    Returns:
        Average precision.

    """
    predictions = np.asarray(predictions)
    groundtruth = np.asarray(groundtruth, dtype=float)

    sorted_indices = np.argsort(predictions)[::-1]
    predictions = predictions[sorted_indices]
    groundtruth = groundtruth[sorted_indices]
    # The false positives are all the negative groundtruth instances, since we
    # assume all instances were 'retrieved'. Ideally, these will be low scoring
    # and therefore in the end of the vector.
    false_positives = 1 - groundtruth

    tp = np.cumsum(groundtruth)      # tp[i] = # of positive examples up to i
    fp = np.cumsum(false_positives)  # fp[i] = # of false positives up to i

    num_positives = tp[-1]

    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    recalls = tp / num_positives

    # Set precisions[i] = max(precisions[j] for j >= i)
    # This is because (for j > i), recall[j] >= recall[i], so we can always use
    # a lower threshold to get the higher recall and higher precision at j.
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    # Append end points of the precision recall curve.
    precisions = np.concatenate(([0.], precisions, [0.]))
    recalls = np.concatenate(([0.], recalls, [1.]))

    # Find points where recall value changes.
    recall_changes = set(np.where(recalls[1:] != recalls[:-1])[0] + 1)

    # Find points where prediction score changes.
    prediction_changes = set(
        np.where(predictions[1:] != predictions[:-1])[0] + 1)

    num_examples = predictions.shape[0]

    # Recall and scores always "change" at the first and last prediction.
    c = recall_changes & prediction_changes | set([0, num_examples])
    c = np.array(sorted(list(c)), dtype=np.int)

    ap = np.sum((recalls[c[1:]] - recalls[c[:-1]]) * precisions[c[1:]])

    return ap
