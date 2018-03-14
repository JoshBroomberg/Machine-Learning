import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets, metrics
from sklearn.cross_validation import train_test_split

# Load data
digits = datasets.load_digits()

condition = np.logical_or(digits.target == 7, digits.target == 1)
selected_digits = np.where(condition)[0]

x_img = digits.data[selected_digits]
y_digit = digits.target[selected_digits]

# Split to train and test
x_train, x_test, y_train, y_test = train_test_split(x_img, y_digit,
    test_size=0.1)

# Set k neighbours
n_neighbors = 15

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(y_test, y_pred)))

    print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))

# Perfect recognition.