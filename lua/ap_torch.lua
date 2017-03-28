local function compute_average_precision(predictions, groundtruth)
    --[[
    Compute mean average prediction.

    Args:
        predictions ((num_samples) Tensor)
        groundtruth ((num_samples) Tensor): Contains 0/1 values.

    Returns:
        mean_average_precision (num)
    ]]--
    predictions = predictions:float()
    groundtruth = groundtruth:byte()
    local ap
    --[[
    Let P(k) be the precision at cut-off for item k. Then, we compute average
    precision for each label as

    \frac{ \sum_{k=1}^n (P(k) * is_positive(k)) }{ # of relevant documents }

    where is_positive(k) is 1 if the groundtruth labeled item k as positive.
    ]]--
    if not torch.any(groundtruth) then
        return 0
    end
    local label_predictions = predictions[{{}, label}]
    local _, sorted_indices = torch.sort(
        label_predictions, 1, true --[[descending]])
    local true_positives = 0
    local average_precision = 0

    local sorted_groundtruth = groundtruth:index(
        1, sorted_indices):float()

    local true_positives = torch.cumsum(sorted_groundtruth)
    local false_positives = torch.cumsum(1 - sorted_groundtruth)
    local num_positives = true_positives[-1]

    local precisions = torch.cdiv(true_positives,
                                    true_positives + false_positives)
    local recalls = true_positives / num_positives


    -- Set precisions[i] = max(precisions[j] for j >= i)
    -- This is because (for j > i), recall[j] >= recall[i], so we can
    -- always use a lower threshold to get the higher recall and higher
    -- precision at j.
    for i = precisions:nElement()-1, 1, -1 do
        precisions[i] = precisions[{{i, i+1}}]:max()
    end

    -- Append end points of the precision recall curve.
    local zero = torch.zeros(1):float()
    local one = torch.ones(1):float()
    precisions = torch.cat({zero, precisions, zero}, 1)
    recalls = torch.cat({zero, recalls, one})

    -- Find points where recall changes.
    local changes = torch.ne(recalls[{{2, -1}}], recalls[{{1, -2}}])
    changes_plus_1 = torch.cat({torch.zeros(1):byte(), changes})
    changes = torch.cat({changes, torch.zeros(1):byte()})

    return torch.cmul((recalls[changes_plus_1] - recalls[changes]),
                      precisions[changes_plus_1]):sum()
end

return { compute_average_precision = compute_average_precision }
