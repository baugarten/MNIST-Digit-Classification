import numpy
from dataset import MNistDigitDataset, OERunner
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

dataset = MNistDigitDataset()
dataset.visualize(row=10)

runner = OERunner(dataset)

rng = numpy.random.RandomState(1)
classifier = RandomForestClassifier(n_estimators=500, max_features='auto')

results = runner.run(classifier)
print results

# Training Results: 0.965298 (0.002656 std)
# Test Results: 0.970357
# (array([[772,   0,   0,   1,   0,   0,   2,   0,   5,   0],
#        [  0, 950,   7,   2,   3,   0,   1,   3,   3,   1],
#        [  2,   1, 805,   1,   5,   0,   2,   6,   5,   0],
#        [  0,   2,  11, 846,   1,   7,   0,   3,   6,   5],
#        [  2,   0,   0,   0, 736,   0,   6,   0,   2,   9],
#        [  4,   1,   0,  12,   0, 754,  10,   0,   2,   5],
#        [  5,   0,   0,   0,   2,   0, 813,   0,   2,   0],
#        [  1,   2,   7,   1,   1,   0,   0, 890,   2,  16],
#        [  0,   5,   3,   5,   3,   5,   3,   1, 773,   4],
#        [  6,   0,   0,  13,   9,   2,   1,   8,   4, 812]]),)
#              precision    recall  f1-score   support
# 
#           0       0.97      0.99      0.98       780
#           1       0.99      0.98      0.98       970
#           2       0.97      0.97      0.97       827
#           3       0.96      0.96      0.96       881
#           4       0.97      0.97      0.97       755
#           5       0.98      0.96      0.97       788
#           6       0.97      0.99      0.98       822
#           7       0.98      0.97      0.97       920
#           8       0.96      0.96      0.96       802
#           9       0.95      0.95      0.95       855
# 
# avg / total       0.97      0.97      0.97      8400
