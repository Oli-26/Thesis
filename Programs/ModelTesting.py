    
import matplotlib.pyplot as plt
from LoadData import load_from_file

def test_naive_bayes():
    from NaiveBayes import naive_bayes_model
    ## Init variables
    number_of_examples = 20000
    verbose = False
    
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    list = []
    iMax = 25
    for i in range(1,iMax):
        print("running for i = " + str(i))
        x = naive_bayes_model(df, verbose, i)
        list.append((x[1][0], x[1][1]))
    test_acc = [b for  (a,b) in list]
    fig = plt.figure(figsize=(8,6))
    plt.scatter(range(1,iMax), test_acc)
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
        x = knn_model(df, verbose, amount_neighbors = i, min_words = 10)
        list.append(x[1])
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
        x = svm_model(df, verbose, min_words = i)
        list.append(x[1])
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
        x = decision_tree_model(df, verbose, min_words = i)
        list.append(x[1])
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
        x = logistic_regression_model(df, verbose, min_words = i)
        list.append(x[1])
    test_acc = [b for (a,b) in list]
    fig = plt.figure(figsize=(8,6))
    plt.scatter(range(10,iMax), test_acc)
    plt.show()

test_logistic_regression()
#test_decision_tree()    
#test_svm()    
#test_knn()    
#test_naive_bayes()