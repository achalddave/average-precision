local torch = require 'torch'

local function compute_average_precision(predictions, groundtruth)
    --[[
    Compute mean average prediction.

    Args:
        predictions ((num_samples) Tensor)
        groundtruth ((num_samples) Tensor): Contains 0/1 values.

    Returns:
        average_precision (num)
    ]]--
    predictions = predictions:float()
    groundtruth = groundtruth:byte()

    --[[
    Let P(k) be the precision at cut-off for item k. Then, we compute average
    precision for each label as

    \frac{ \sum_{k=1}^n (P(k) * is_positive(k)) }{ # of relevant documents }

    where is_positive(k) is 1 if the groundtruth labeled item k as positive.
    ]]--
    if not torch.any(groundtruth) then
        return 0
    end
    local predictions_, sorted_indices = torch.sort(
        predictions, 1, true --[[descending]])
    predictions = predictions_

    local sorted_groundtruth = groundtruth:index(1, sorted_indices):float()

    local true_positives = torch.cumsum(sorted_groundtruth)
    local false_positives = torch.cumsum(1 - sorted_groundtruth)
    local num_positives = true_positives[-1]

    local precisions = torch.cdiv(
        true_positives,
        torch.cmax(true_positives + false_positives, 1e-16))
    local recalls = true_positives / num_positives

    -- Append end points of the precision recall curve.
    local zero = torch.zeros(1):float()
    local one = torch.ones(1):float()
    precisions = torch.cat({zero, precisions, zero}, 1)
    recalls = torch.cat({zero, recalls, one})

    -- Find points where prediction score changes.
    local changes = torch.ne(predictions[{{2, -1}}], predictions[{{1, -2}}])

    -- First and last element should always count as change.
    changes = torch.cat({one:byte(), changes, one:byte()})
    -- Make changes same size as recall/precision. TODO(achald): Explain why
    -- this line is necessary.
    changes = torch.cat({changes, zero:byte()})

    local recall_at_changes = recalls[changes]
    local recall_at_changes_offset = recall_at_changes[{{2, -1}}]
    recall_at_changes = recall_at_changes[{{1, -2}}]
    local precision_at_changes_offset = precisions[changes]
    precision_at_changes_offset = precision_at_changes_offset[{{2, -1}}]

    -- Set precisions[i] = max(precisions[j] for j >= i)
    -- This is because (for j > i), recall[j] >= recall[i], so we can
    -- always use a lower threshold to get the higher recall and higher
    -- precision at j.
    for i = precision_at_changes_offset:nElement()-1, 1, -1 do
        precision_at_changes_offset[i] = math.max(
            precision_at_changes_offset[i],
            precision_at_changes_offset[i+1])
    end

    return torch.cmul(
        (recall_at_changes_offset - recall_at_changes),
        precision_at_changes_offset):sum()
end

return { compute_average_precision = compute_average_precision }
