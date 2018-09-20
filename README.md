# (Mean) Average Precision in Various Languages

Computing mean average precision is a common task, so I'm collecting a single
(hopefully solid) implementation in various languages here.

This method follows the description of AP in the PASCAL VOC 2012 challenge:
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#sec:ap
Note: This code is not yet battle-tested!  Your best resource for a battle
tested/well accepted AP calculation is probably to look at the PASCAL VOC MATLAB
code for mean average precision:
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#sec:ap

However, note that the VOC code fails on test cases where the predictions have
many ties. For example, the 'all_ties_1' test case in
[test_cases.yaml](./test_cases.yaml) results in an AP of 1.0 in many AP
implementations. The code in this repository fixes that issue.
