import matplotlib.pyplot as plt
from LoadData import load_from_file
from sklearn.model_selection import train_test_split
import numpy as np

from LogisticRegression import train_logistic_regression
from DecisionTree import train_decision_tree
from SVM import train_svm
from KNN import train_knn
from NaiveBayes import train_naive_bayes
    


def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df['commenttext'], df['category_id'], random_state = 10, train_size = 0.25)
    return (X_train, X_test, y_train, y_test)
    
def test_model(verbose, text_clf, X_train, X_test, y_train, y_test):
    train_score = float(text_clf.score(X_train, y_train))
    test_score = float(text_clf.score(X_test, y_test))
    
    if verbose:
        general_predicted = text_clf.predict(X_test)  
        print("Train accuracy = " + str(train_score))
        print("Test accuracy = " + str(test_score))
        #Show percentage of general prediction types 
        unique, counts = np.unique(general_predicted, return_counts=True)
        print(dict(zip(unique, counts*100/(len(general_predicted)))))

    
    return (train_score, test_score)
    
    
def test_naive_bayes():
    ## Init variables
    number_of_examples = 20000
    verbose = False
    
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    X_train, X_test, y_train, y_test = split_data(df)
    
    list = []
    iMax = 25
    for i in range(5,iMax):
        print("running for i = " + str(i))
        m = train_naive_bayes_model(X_train, y_train, i)
        s = test_model(verbose, m, X_train, X_test, y_train, y_test)
        list.append(s)
    test_acc = [b for  (a,b) in list]
    
    
    fig = plt.figure(figsize=(8,6))
    plt.scatter(range(5,iMax), test_acc)
    plt.show()
    
    
    
def test_knn():
     ## Init variables
    number_of_examples = 20000
    verbose = False 
        
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    list = []
    iMax = 20
    for i in range(1, iMax):
        print("running for i = " + str(i))
        m = train_knn(X_train, y_train, amount_neighbors = i, min_df = 10)
        s = test_model(verbose, m, X_train, X_test, y_train, y_test)
        list.append(s)
    test_acc = [b for (a,b) in list]
    fig = plt.figure(figsize=(8,6))
    plt.scatter(range(1,iMax), test_acc)
    plt.show()
    
def test_svm():
     ## Init variables
    number_of_examples = 20000
    verbose = False 
        
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    list = []
    iMax = 25
    for i in range(10, iMax):
        print("running for i = " + str(i))
        m = train_svm(X_train, y_train, min_df = i)
        s = test_model(verbose, m, X_train, X_test, y_train, y_test)
        list.append(s)
    test_acc = [b for (a,b) in list]
    fig = plt.figure(figsize=(8,6))
    plt.scatter(range(10,iMax), test_acc)
    plt.show()
 

def test_decision_tree():
    ## Init variables
    number_of_examples = 20000
    verbose = False 
        
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    list = []
    iMax = 25
    for i in range(10, iMax):
        print("running for i = " + str(i))
        m = train_decision_tree(X_train, y_train, min_df = i)
        s = test_model(verbose, m, X_train, X_test, y_train, y_test)
        list.append(s)
    test_acc = [b for (a,b) in list]
    fig = plt.figure(figsize=(8,6))
    plt.scatter(range(10,iMax), test_acc)
    plt.show()

def test_logistic_regression():
    ## Init variables
    number_of_examples = 20000
    verbose = False 
        
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    list = []
    iMax = 25
    for i in range(10, iMax):
        print("running for i = " + str(i))
        m = train_logistic_regression(X_train, y_train, min_df = i)
        s = test_model(verbose, m, X_train, X_test, y_train, y_test)
        list.append(s)
    test_acc = [b for (a,b) in list]
    fig = plt.figure(figsize=(8,6))
    plt.scatter(range(10,iMax), test_acc)
    plt.show()

    
def compare_all():
    ## Init variables
    number_of_examples = 100000
    verbose = False
    
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    X_train, X_test, y_train, y_test = split_data(df)
    
    

    print("Distribution -")
    print(df.groupby('classification').commenttext.count()/(df.shape[0]/100))
    print("-------------\n")
    
    name_list = ["naive_bayes", "logistic_regression", "decision_tree", "svm", "knn"]
    listTest = []
    listTrain = []
    i = 10
    
    print("NaiveBayes STATS ----------")
    m = train_naive_bayes(X_train, y_train, i)
    s = test_model(True, m, X_train, X_test, y_train, y_test)
    listTest.append(s[1])
    listTrain.append(s[0])
    print("--------------------\n")
    
    print("LG STATS ----------")
    x = train_logistic_regression(X_train, y_train, min_df = i)
    s = test_model(True, m, X_train, X_test, y_train, y_test)
    listTest.append(s[1])
    listTrain.append(s[0])
    print("--------------------\n")
    
    print("DT STATS ----------")
    m = train_decision_tree(X_train, y_train, min_df = i)
    s = test_model(True, m, X_train, X_test, y_train, y_test)
    listTest.append(s[1])
    listTrain.append(s[0])
    print("--------------------\n")
    
    print("SVM STATS ----------")
    m = train_svm(X_train, y_train, min_df = i)
    s = test_model(True, m, X_train, X_test, y_train, y_test)
    listTest.append(s[1])
    listTrain.append(s[0])
    print("--------------------\n")
    
    print("KNN STATS ----------")
    m = train_knn(X_train, y_train, amount_neighbors = 3, min_df = i)
    s = test_model(True, m, X_train, X_test, y_train, y_test)
    listTest.append(s[1])
    listTrain.append(s[0])
    print("--------------------\n")   
    
    
    feature_array = np.array(m['tfidf'].get_feature_names())
    print("Discovered = "  + str(len(feature_array)) + " features.")
    print("Ran on " + str(df.shape[0]) + " examples.")
      
    fig = plt.figure(figsize=(8,6))
    plt.scatter(name_list, listTest)
    plt.scatter(name_list, listTrain)
    plt.show()
    
#test_logistic_regression()
#test_decision_tree()    
#test_svm()    
#test_knn()    
#test_naive_bayes()
#compare_all()
