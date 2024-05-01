import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-s', '--split', type=float, default=0.2)
    group.add_argument('-t', '--test')
    parser.add_argument('-o', '--output')
    args = parser.parse_args(sys.argv[1:])

    # Load csv data
    df = pd.read_csv(args.data)

    if args.test == None:
        # If test data file not specified, split data randomly
        id = df['id']
        X = df.drop(['id', 'Class'], axis=1)
        y = df['Class']
        id_train, id_test, X_train, X_test, y_train, y_test = train_test_split(id, X, y,
                                                                               test_size=args.split,
                                                                               random_state=1337)
    else:
        # Otherwise use first file for training data
        id_train = df['id']
        X_train = df.drop(['id', 'Class'], axis=1)
        y_train = df['Class']

        # And use second file for testing data
        df = pd.read_csv(args.test)
        id_test = df['id']
        X_test = df.drop(['id', 'Class'], axis=1)
        y_test = df['Class']

    # Fit model and make predictions
    bayes = GaussianNB()
    bayes.fit(X_train, y_train)
    predictions = bayes.predict(X_test)

    # Display confusion matrix
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()

    # Display metrics
    print('Accuracy score: {}'.format(accuracy_score(y_test, predictions)))
    print('Precision score: {}'.format(precision_score(y_test, predictions)))
    print('Recall score: {}'.format(recall_score(y_test, predictions)))
    print('F1 score: {}'.format(f1_score(y_test, predictions)))

    # Output test predictions to file
    if args.output != None:
        data = np.vstack([id_test.values, y_test, predictions]).T
        df = pd.DataFrame(data, columns=['id', 'y_test', 'y_pred'])
        df.to_csv(args.output, index=False)