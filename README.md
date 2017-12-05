# (Mean) Average Precision in Various Languages

Computing mean average precision is a common task, so I'm collecting a single
(hopefully solid) implementation in various languages here.

NOTE: This is not yet battle-tested! Concretely, this code still does not behave
correctly in the presence of ties in prediction scores. See test_cases.yaml for
details. However, to my knowledge, that test case is broken in almost all
popular AP calculations.

Your best resource for a battle tested/well accepted AP calculation is probably
to look at the PASCAL VOC MATLAB code for mean average precision:
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#sec:ap
