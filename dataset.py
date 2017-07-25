import pandas
import matplotlib.pyplot as plt
from numpy import nan
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class OEDataset(object):
    def __init__(self, dataset):
        dataset = dataset.values
        x = dataset[:,1:]
        y = dataset[:,0]
        self.__init__(x, y)

    def __init__(self, x_values, y_values):
        self.x = x_values
        self.y = y_values

    def subset(rows): 
        return (self.x[1:rows,:], self.y[1:rows,:])

class OERunner:
    def __init__(self, oe_dataset):
        self.oe_dataset = oe_dataset

    def run(self, classifier, validation_size=0.2, cross_validate=True):
        x_train, x_validation, y_train, y_validation = model_selection.train_test_split(self.oe_dataset.x, self.oe_dataset.y, test_size=validation_size)

        if cross_validate:
            print "Cross val score"
            training_score = model_selection.cross_val_score(classifier, x_train, y_train, cv=model_selection.KFold(n_splits=10), scoring='accuracy')
            print "Cross val score computed"
        else:
            training_score = NoTrainingScore()

        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_validation)

        return OEResults(training_score, 
                         accuracy_score(y_validation, predictions), 
                         confusion_matrix(y_validation, predictions), 
                         classification_report(y_validation, predictions))

class OEResults:
    def __init__(self, training_scores, test_score, confusion_matrix, classification_report):
        self.training_scores = training_scores
        self.test_score = test_score
        self.confusion_matrix = confusion_matrix,
        self.classification_report = classification_report
    
    def __str__(self):
        return ("Training Results: %f (%f stf)\n"
                "Test Results: %f\n"
                "%s\n"
                "%s\n"
               ) % (self.training_scores.mean(), self.training_scores.std(), self.test_score, self.confusion_matrix, self.classification_report)

class NoTrainingScore():
    def mean(self):
        return nan
    def std(self):
        return nan

class MNistDigitDataset(OEDataset):
    def __init__(self):
        dataset = pandas.read_csv('./train.csv')
        dataset = dataset.values
        x = dataset[:,1:]
        y = dataset[:,0]
        x[x>0] = 1
        super(MNistDigitDataset, self).__init__(x, y)

    def visualize(self, row=None):
        if row is None:
            row = random.randrange(0, self.x.shape[0] - 1)
        img = self.x[row].reshape(28,28)
        plt.imshow(img, cmap='binary')
        plt.show()
        
