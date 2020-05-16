import matplotlib.pyplot as plt
from LoadData import load_from_file
from sklearn.model_selection import train_test_split
import numpy as np

def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df['commenttext'], df['category_id'], random_state = 10, train_size = 0.25)
    return (X_train, X_test, y_train, y_test)
    
def test_model(verbose, text_clf, X_train, X_test, y_train, y_test):

    ##Lets predict to see the generality of our model 
    general_predicted = text_clf.predict(X_test)    

    train_score = float(text_clf.score(X_train, y_train))
    test_score = float(text_clf.score(X_test, y_test))
    
    if verbose:
        print("Train accuracy = " + str(train_score))
        print("Test accuracy = " + str(test_score))
        #Show percentage of general prediction types 
        unique, counts = np.unique(general_predicted, return_counts=True)
        print(dict(zip(unique, counts*100/(len(general_predicted)))))

    
    return (train_score, test_score)
    
    
def test_naive_bayes():
    from NaiveBayes import naive_bayes_model
    ## Init variables
    number_of_examples = 20000
    verbose = False
    
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    X_train, X_test, y_train, y_test = split_data(df)
    
    list = []
    iMax = 25
    for i in range(5,iMax):
        print("running for i = " + str(i))
        m = naive_bayes_model(X_train, y_train, verbose, i)
        s = test_model(verbose, m, X_train, X_test, y_train, y_test)
        list.append(s)
    test_acc = [b for  (a,b) in list]
    
    
    fig = plt.figure(figsize=(8,6))
    plt.scatter(range(5,iMax), test_acc)
    plt.show()
    
    
    
def test_knn():
    from KNN import knn_model

     ## Init variables
    number_of_examples = 20000
    verbose = False 
        
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    list = []
    iMax = 20
    for i in range(1, iMax):
        print("running for i = " + str(i))
        m = knn_model(X_train, y_train, verbose, amount_neighbors = i, min_words = 10)
        s = test_model(verbose, m, X_train, X_test, y_train, y_test)
        list.append(s)
    test_acc = [b for (a,b) in list]
    fig = plt.figure(figsize=(8,6))
    plt.scatter(range(1,iMax), test_acc)
    plt.show()
    
def test_svm():
    from SVM import svm_model
    
     ## Init variables
    number_of_examples = 20000
    verbose = False 
        
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    list = []
    iMax = 25
    for i in range(10, iMax):
        print("running for i = " + str(i))
        m = svm_model(X_train, y_train, verbose, min_words = i)
        s = test_model(verbose, m, X_train, X_test, y_train, y_test)
        list.append(s)
    test_acc = [b for (a,b) in list]
    fig = plt.figure(figsize=(8,6))
    plt.scatter(range(10,iMax), test_acc)
    plt.show()
 

def test_decision_tree():
    from DecisionTree import decision_tree_model
       ## Init variables
    number_of_examples = 20000
    verbose = False 
        
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    list = []
    iMax = 25
    for i in range(10, iMax):
        print("running for i = " + str(i))
        m = decision_tree_model(X_train, y_train, verbose, min_words = i)
        s = test_model(verbose, m, X_train, X_test, y_train, y_test)
        list.append(s)
    test_acc = [b for (a,b) in list]
    fig = plt.figure(figsize=(8,6))
    plt.scatter(range(10,iMax), test_acc)
    plt.show()

def test_logistic_regression():
    from LogisticRegression import logistic_regression_model
       ## Init variables
    number_of_examples = 20000
    verbose = False 
        
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    list = []
    iMax = 25
    for i in range(10, iMax):
        print("running for i = " + str(i))
        m = logistic_regression_model(X_train, y_train, verbose, min_words = i)
        s = test_model(verbose, m, X_train, X_test, y_train, y_test)
        list.append(s)
    test_acc = [b for (a,b) in list]
    fig = plt.figure(figsize=(8,6))
    plt.scatter(range(10,iMax), test_acc)
    plt.show()

    
def compare_all():
    from LogisticRegression import logistic_regression_model
    from DecisionTree import decision_tree_model
    from SVM import svm_model
    from KNN import knn_model
    from NaiveBayes import naive_bayes_model
    
    ## Init variables
    number_of_examples = 20000
    verbose = False
    
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    X_train, X_test, y_train, y_test = split_data(df)
    
    name_list = ["naive_bayes", "logistic_regression", "decision_tree", "svm", "knn"]
    list = []
    i = 10
    
    print("NaiveBayes STATS ----------")
    m = naive_bayes_model(X_train, y_train, verbose, i)
    s = test_model(True, m, X_train, X_test, y_train, y_test)
    list.append(s[1])
    print("--------------------\n")
    
    print("LG STATS ----------")
    x = logistic_regression_model(X_train, y_train, verbose, min_words = i)
    s = test_model(True, m, X_train, X_test, y_train, y_test)
    list.append(s[1])
    print("--------------------\n")
    
    print("DT STATS ----------")
    m = decision_tree_model(X_train, y_train, verbose, min_words = i)
    s = test_model(True, m, X_train, X_test, y_train, y_test)
    list.append(s[1])
    print("--------------------\n")
    
    print("SVM STATS ----------")
    m = svm_model(X_train, y_train, verbose, min_words = i)
    s = test_model(True, m, X_train, X_test, y_train, y_test)
    list.append(s[1])
    print("--------------------\n")
    
    print("KNN STATS ----------")
    m = knn_model(X_train, y_train, verbose, amount_neighbors = 3, min_words = i)
    s = test_model(True, m, X_train, X_test, y_train, y_test)
    list.append(s[1])
    print("--------------------\n")   
        
    fig = plt.figure(figsize=(8,6))
    plt.scatter(name_list, list)
    plt.show()
    
    
#test_logistic_regression()
#test_decision_tree()    
#test_svm()    
#test_knn()    
#test_naive_bayes()
compare_all()
