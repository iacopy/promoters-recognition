import sys

from sklearn.neural_network import MLPClassifier

import loader

print('Loading data set...')
(X_train, y_train),(X_test, y_test) = loader.load_data(sys.argv[1])

HSIZE = 10

clf = MLPClassifier(
    hidden_layer_sizes=(HSIZE, ),
    activation='logistic',
    solver='sgd',
    early_stopping=True,
    validation_fraction=0.15,
    verbose=True,
)

print('Training...')
clf.fit(X_train, y_train)
res = clf.score(X_test, y_test)
print(res)
