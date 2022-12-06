# Logistic reg imports
from sklearn.linear_model import LogisticRegression

# SVM imports
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# K-NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

# NN imports
from sklearn.neural_network import MLPClassifier


def train_test_model(model_to_use,x_train,y_train,x_test):
    
    if model_to_use == "LogisticRegression":
        logisticRegr = LogisticRegression(max_iter=4000)
        logisticRegr.fit(x_train, y_train) 
        return logisticRegr.predict(x_test)
    
    if model_to_use == "SVM":
        # Use linear SVM for optimization (text classification is often linear)
        lin_clf = svm.LinearSVC()
        lin_clf.fit(x_train,y_train)
        return lin_clf.predict(x_test)
    
    if model_to_use == "NeuralNetwork":
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1,max_iter=4000)
        clf.fit(x_train, y_train)
        return clf.predict(x_test)
        