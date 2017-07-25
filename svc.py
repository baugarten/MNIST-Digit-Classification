from dataset import MNistDigitDataset, OERunner
from sklearn.linear_model import SGDClassifier

dataset = MNistDigitDataset()

runner = OERunner(dataset)

from sklearn.svm import SVC
classifier = SVC(C=5, gamma=.05)
results = runner.run(classifier, cross_validate=False)
print results

# Test Results: 0.968571
# (array([[838,   0,   3,   2,   1,   4,   3,   0,   5,   0],
#        [  0, 961,   2,   0,   0,   1,   2,   1,   3,   1],
#        [  0,   1, 812,   3,   2,   1,   0,   0,   8,   1],
#        [  0,   0,  12, 810,   0,   6,   0,   5,   6,   4],
#        [  1,   1,  10,   0, 801,   0,   2,   1,   0,   7],
#        [  0,   0,   5,  14,   0, 742,   3,   0,   4,   1],
#        [  2,   1,   9,   0,   1,   6, 760,   0,   3,   0],
#        [  0,   1,  19,   2,   4,   0,   0, 857,   4,   9],
#        [  1,   1,  12,  12,   4,   4,   1,   0, 768,   2],
#        [  1,   1,   8,   9,   9,   2,   0,   6,   4, 787]]),)
#              precision    recall  f1-score   support
# 
#           0       0.99      0.98      0.99       856
#           1       0.99      0.99      0.99       971
#           2       0.91      0.98      0.94       828
#           3       0.95      0.96      0.96       843
#           4       0.97      0.97      0.97       823
#           5       0.97      0.96      0.97       769
#           6       0.99      0.97      0.98       782
#           7       0.99      0.96      0.97       896
#           8       0.95      0.95      0.95       805
#           9       0.97      0.95      0.96       827
# 
# avg / total       0.97      0.97      0.97      8400
