local torch = require 'torch'
local compute_ap = require('lua/ap_torch').compute_average_precision
local lyaml = require 'lyaml'

local ap_test = torch.TestSuite()
local ap_tester = torch.Tester()

function test_map1()
    local scores = torch.range(10, 1, -1):resize(10, 1)
    local labels = torch.ByteTensor(
        {1, 0, 1, 1, 1, 1, 0, 0, 0, 1}):resizeAs(scores:byte())
    ap = compute_ap(scores, labels)
    ap_true = (1 + 2/3 + 3/4 + 4/5 + 5/6 + 6/10) / 6
    assert(ap == ap_true,
           string.format('Expected %.5f, received %.5f', ap_true, ap))
end

function ap_test.testAll()
    local config = lyaml.load(io.open('test_cases.yaml', 'r'):read('*a'))

    for i, test_case in ipairs(config) do
        print('Testing ', test_case.name)
        ap_tester:assertalmosteq(
            compute_ap(torch.FloatTensor(test_case.predictions),
                       torch.FloatTensor(test_case.groundtruth),
                       test_case.false_negatives),
            test_case.expected_ap,
            1e-6 --[[tolerance]])
    end
end

ap_tester:add(ap_test)
ap_tester:run()
