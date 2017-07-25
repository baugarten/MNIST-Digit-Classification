from dataset import MNistDigitDataset, OERunner
from sklearn.linear_model import SGDClassifier

dataset = MNistDigitDataset()
dataset.visualize(row=10)

runner = OERunner(dataset)

results = runner.run(SGDClassifier(shuffle=True))
print results

