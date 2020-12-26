import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import utils
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn import tree
import graphviz
import io

def dataLoading(fpath):
    # read selected csv
    dataset = pd.read_csv(fpath)


    # prepare independent and dependent variables
    dataset = dataset.drop('Date', axis=1)
    x = dataset.drop('GINIALLRF', axis=1)
    y = dataset['GINIALLRF']

    # data preprocessing
    X_norm = MinMaxScaler().fit_transform(x)
    Y = y.to_numpy()

    # using label encoder to encode dependent values
    lab_enc = preprocessing.LabelEncoder()
    Y_encod = lab_enc.fit_transform(Y)

    # Select attributes using logistic regression
    model = LogisticRegression()
    rfe = RFE(model, 12)
    fit = rfe.fit(X_norm, Y_encod)

    # Dropping attributes as suggested by logistic regression
    data = dataset.drop('W019RCQ027SBEA', axis=1)
    data = dataset.drop('ASCOEPQ027S', axis=1)
    data = dataset.drop('GDP', axis=1)
    data = dataset.drop('GNP', axis=1)
    data = dataset.drop('Government', axis=1)
    data = dataset.drop('HNICTIQ027S', axis=1)
    X_norm = MinMaxScaler().fit_transform(data)

    # Prepare test and training data
    X_train, X_test, y_train, y_test = train_test_split(X_norm, Y_encod, test_size=0.30)

    # Training the model
    classifier = DecisionTreeClassifier(criterion="gini",max_depth=6)
    classifier.fit(X_train, y_train)

    # Testing the model
    y_pred = classifier.predict(X_test)

    # Checking Accuracy
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    acc = metrics.accuracy_score(y_test, y_pred)
    accP = round(metrics.accuracy_score(y_test, y_pred), 3) * 100
    bal_acc = round(metrics.balanced_accuracy_score(y_test, y_pred), 3) * 100

    # Getting output data
    buffer = io.StringIO()
    dataset.info(buf=buffer)
    csvInfo = buffer.getvalue()

    return csvInfo, conf_matrix, class_report, acc, accP, bal_acc, y_test, y_pred, classifier
